"""Evaluation harness: frozen eval sets, benchmark runs, promotion decisions.

The evaluation agent owns this module. It is the single source of truth on
model quality (PRD 7.9). Held-out sets are write-protected from every other
agent once frozen (FR-14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from internal_ai.audit import AuditLog
from internal_ai.data.records import Record
from internal_ai.data.splits import contamination_report
from internal_ai.data.transforms import PII_PATTERNS
from internal_ai.domain import AuditOutcome, EvalReport, EvalSet
from internal_ai.evals.metrics import (
    bm25_scores,
    cosine_similarity_matrix,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_recall_f1,
    rank_from_scores,
    recall_at_k,
)
from internal_ai.ids import content_hash, new_id, now_iso
from internal_ai.store import Store


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass(slots=True)
class EvalItem:
    query: str
    relevant_ids: list[str]
    kind: str = "retrieval"
    label: str | None = None
    meta: dict[str, Any] = None


@dataclass(slots=True)
class EvalOutcome:
    passed: bool
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    checks: dict[str, Any]
    notes: str


class EvalHarness:
    def __init__(self, store: Store, audit: AuditLog) -> None:
        self.store = store
        self.audit = audit

    # ------------------------------------------------------------ eval sets

    def create_eval_set(
        self,
        name: str,
        items: list[dict[str, Any]],
        version: int = 1,
        actor: str = "evaluation",
    ) -> EvalSet:
        payload = json_safe(items)
        eval_set = EvalSet(
            id=new_id("evs"),
            name=name,
            version=version,
            frozen=False,
            created_at=now_iso(),
            content_hash=content_hash(payload),
            path="",
            items=payload,
        )
        self.store.add_eval_set(eval_set)
        self.audit.record(
            actor=actor,
            action="eval_set.created",
            subject=eval_set.id,
            outcome=AuditOutcome.SUCCESS,
            details={"name": name, "version": version, "items": len(items)},
        )
        return eval_set

    def freeze_eval_set(
        self,
        eval_set_id: str,
        approver: str,
        gate_approved: bool,
        actor_kind: str = "human",
    ) -> EvalSet:
        if not gate_approved:
            raise PermissionError(
                "eval set freeze requires an approved eval_set_freeze gate (PRD 7.3, FR-14)"
            )
        eval_set = self.store.get_eval_set(eval_set_id)
        if eval_set is None:
            raise KeyError(f"unknown eval set: {eval_set_id}")
        if eval_set.frozen:
            raise PermissionError(f"eval set {eval_set_id} is already frozen and immutable")
        self.store.freeze_eval_set(eval_set_id, approver)
        self.audit.record(
            actor=approver,
            actor_kind=actor_kind,
            action="eval_set.frozen",
            subject=eval_set_id,
            details={"name": eval_set.name, "version": eval_set.version},
        )
        return self.store.get_eval_set(eval_set_id)

    def write_to_frozen_set(self, eval_set_id: str, actor: str) -> None:
        eval_set = self.store.get_eval_set(eval_set_id)
        if eval_set and eval_set.frozen:
            self.audit.record(
                actor=actor,
                action="eval_set.write_attempt",
                subject=eval_set_id,
                outcome=AuditOutcome.DENIED,
                details={"reason": "eval set is frozen and write-protected"},
            )
            raise PermissionError(
                f"eval set {eval_set_id} is frozen; changes require a new version and approval"
            )
        self.audit.record(
            actor=actor,
            action="eval_set.write_attempt",
            subject=eval_set_id,
            outcome=AuditOutcome.DENIED,
            details={"reason": "eval sets are written only through create_eval_set"},
        )
        raise PermissionError(
            f"eval set {eval_set_id} cannot be modified in place; create a new version"
        )

    # ----------------------------------------------------------- evaluation

    def evaluate_retrieval(
        self,
        model: Embedder,
        eval_set: EvalSet,
        corpus_records: list[Record],
        training_records: list[Record] | None = None,
        top_k: int = 10,
        margin: float = 0.0,
        actor: str = "evaluation",
    ) -> EvalOutcome:
        queries = [str(item["query"]) for item in eval_set.items]
        doc_ids = [record.record_key for record in corpus_records]
        doc_texts = [record.text for record in corpus_records]
        relevance = {
            str(index): {str(doc) for doc in item.get("relevant_ids", [])}
            for index, item in enumerate(eval_set.items)
        }
        query_vectors = model.embed(queries)
        doc_vectors = model.embed(doc_texts)
        similarity = cosine_similarity_matrix(query_vectors, doc_vectors)
        model_rankings = {
            str(index): rank_from_scores(list(similarity[index]), doc_ids)
            for index in range(len(queries))
        }
        baseline_rankings: dict[str, list[str]] = {}
        baseline_scores: list[float] = []
        for index, query in enumerate(queries):
            scores = bm25_scores(query, doc_texts)
            baseline_scores.extend(scores)
            baseline_rankings[str(index)] = rank_from_scores(scores, doc_ids)

        metrics = {
            "recall_at_1": recall_at_k(relevance, model_rankings, 1),
            "recall_at_5": recall_at_k(relevance, model_rankings, 5),
            "recall_at_10": recall_at_k(relevance, model_rankings, top_k),
            "mrr": mean_reciprocal_rank(relevance, model_rankings),
            "ndcg_at_10": ndcg_at_k(relevance, model_rankings, 10),
        }
        baseline_metrics = {
            "recall_at_1": recall_at_k(relevance, baseline_rankings, 1),
            "recall_at_5": recall_at_k(relevance, baseline_rankings, 5),
            "recall_at_10": recall_at_k(relevance, baseline_rankings, top_k),
            "mrr": mean_reciprocal_rank(relevance, baseline_rankings),
            "ndcg_at_10": ndcg_at_k(relevance, baseline_rankings, 10),
        }
        relevant_doc_ids = {doc for docs in relevance.values() for doc in docs}
        checks = self._safety_checks(
            model, corpus_records, training_records or [], relevant_doc_ids
        )
        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        checks["baseline_informativeness"] = {
            "passed": True,
            "baseline_mean_bm25_score": round(baseline_mean, 6),
            "note": (
                "baseline has no lexical signal on this eval set, so the comparison is "
                "easier than a production query mix would be"
                if baseline_mean < 1e-6
                else "baseline carries lexical signal on this eval set"
            ),
        }
        beats_baseline = metrics["recall_at_10"] >= baseline_metrics["recall_at_10"] + margin
        passed = beats_baseline and all(check["passed"] for check in checks.values())
        notes = (
            "model beats the keyword-search baseline"
            if beats_baseline
            else "model does not beat the keyword-search baseline by the required margin"
        )
        outcome = EvalOutcome(
            passed=passed,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            checks=checks,
            notes=notes,
        )
        self.audit.record(
            actor=actor,
            action="eval.run",
            subject=eval_set.id,
            outcome=AuditOutcome.SUCCESS if passed else AuditOutcome.FAILURE,
            details={
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "passed": passed,
            },
        )
        return outcome

    def evaluate_classifier(
        self,
        predictions: list[str],
        labels: list[str],
        positive: str = "positive",
        actor: str = "evaluation",
    ) -> EvalOutcome:
        metrics = precision_recall_f1(predictions, labels, positive=positive)
        checks = {
            "per_class_present": {"passed": len(set(labels)) > 0, "detail": sorted(set(labels))}
        }
        passed = metrics["f1"] > 0
        return EvalOutcome(
            passed=passed,
            metrics=metrics,
            baseline_metrics={},
            checks=checks,
            notes="per-class precision, recall and F1 computed",
        )

    def _safety_checks(
        self,
        model: Embedder,
        corpus_records: list[Record],
        training_records: list[Record],
        relevant_doc_ids: set[str],
    ) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        probes = [
            "email: test.user@example.com",
            "phone: +1 415 555 0134",
            "ssn: 123-45-6789",
            "aws key: AKIAIOSFODNN7EXAMPLE",
        ]
        try:
            doc_by_id = {record.record_key: record.text for record in corpus_records}
            doc_ids = list(doc_by_id)
            doc_vectors = model.embed([doc_by_id[doc_id] for doc_id in doc_ids])
            probe_vectors = model.embed(probes)
            similarity = cosine_similarity_matrix(probe_vectors, doc_vectors)
            regurgitated: list[dict[str, Any]] = []
            for index, probe in enumerate(probes):
                ranking = rank_from_scores(list(similarity[index]), doc_ids)[:10]
                for doc_id in ranking:
                    text = doc_by_id.get(doc_id, "")
                    for pattern in PII_PATTERNS.values():
                        match = pattern.search(text)
                        if match:
                            regurgitated.append(
                                {"probe": probe, "document": doc_id, "match": match.group(0)}
                            )
                            break
            checks["pii_regurgitation"] = {
                "passed": not regurgitated,
                "hits": regurgitated[:5],
            }
        except Exception as exc:
            checks["pii_regurgitation"] = {"passed": False, "error": str(exc)}
        if training_records and relevant_doc_ids:
            eval_documents = [
                record for record in corpus_records if record.record_key in relevant_doc_ids
            ]
            contamination = contamination_report(eval_documents, training_records)
            checks["contamination"] = contamination
        elif relevant_doc_ids:
            checks["contamination"] = {
                "passed": False,
                "note": (
                    "no training corpus was supplied, so overlap between the eval set "
                    "and training data could not be ruled out"
                ),
            }
        else:
            checks["contamination"] = {
                "passed": False,
                "note": "eval set has no relevance labels, contamination check could not run",
            }
        checks["permission_leak"] = permission_probe(corpus_records, relevant_doc_ids)
        return checks

    def record_report(
        self,
        model_id: str,
        eval_set_id: str,
        outcome: EvalOutcome,
        actor: str = "evaluation",
    ) -> EvalReport:
        report = EvalReport(
            id=new_id("rep"),
            model_id=model_id,
            eval_set_id=eval_set_id,
            created_at=now_iso(),
            metrics=outcome.metrics,
            baseline_metrics=outcome.baseline_metrics,
            checks=outcome.checks,
            passed=outcome.passed,
            notes=outcome.notes,
        )
        self.store.add_eval_report(report)
        self.audit.record(
            actor=actor,
            action="eval.report_created",
            subject=report.id,
            outcome=AuditOutcome.SUCCESS if report.passed else AuditOutcome.FAILURE,
            details={"model_id": model_id, "passed": report.passed},
        )
        return report


def permission_probe(corpus_records: list[Record], relevant_doc_ids: set[str]) -> dict[str, Any]:
    """Ask the serving permission rule whether an eval document leaks.

    A check that cannot fail is not a check. This uses ``Principal.can_read`` -
    the same rule the retrieval service applies - and fails on three things: a
    document with no access list (readable by everyone), a labelled document a
    stranger can still read, and a document nobody can read, which would make the
    eval measure something no user could ever reach.
    """
    from internal_ai.serving.service import Principal

    labelled = [record for record in corpus_records if record.record_key in relevant_doc_ids]
    if not labelled:
        return {
            "passed": False,
            "note": "no labelled document to probe",
            "probed": 0,
        }
    unscoped = [record.record_key for record in labelled if not record.acl]
    stranger = Principal(user_id="eval-probe", groups=frozenset())
    readable = [
        record.record_key
        for record in labelled
        if stranger.can_read(record.acl, record.sensitivity)
    ]
    granted = frozenset(group for record in labelled for group in record.acl)
    insider = Principal(user_id="eval-probe", groups=granted)
    unreachable = [
        record.record_key
        for record in labelled
        if not insider.can_read(record.acl, record.sensitivity)
    ]
    return {
        "passed": not unscoped and not readable and not unreachable,
        "probed": len(labelled),
        "without_access_list": unscoped[:5],
        "readable_without_a_grant": readable[:5],
        "unreachable_with_every_grant": unreachable[:5],
    }


def json_safe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for item in items:
        safe.append({str(key): value for key, value in item.items()})
    return safe
