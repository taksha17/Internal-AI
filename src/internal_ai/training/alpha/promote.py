"""Register an alpha adapter and ask humans to promote it. Serving waits on that gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.data.records import Record
from internal_ai.data.splits import contamination_report
from internal_ai.data.transforms import PII_PATTERNS
from internal_ai.domain import AuditOutcome, EvalReport, EvalSet, GateType, ModelVersion
from internal_ai.ids import content_hash, file_hash, new_id, now_iso
from internal_ai.services import governance
from internal_ai.services.actor import Actor
from internal_ai.services.alpha import reproducibility_beside
from internal_ai.services.permissions import Action, require
from internal_ai.training.alpha.card import alpha_card
from internal_ai.training.alpha.citations import LINE_NUMBER, AskHit, CitationReport, score_answer
from internal_ai.training.alpha.pin import FALLBACK, PRIMARY, RECIPE, LicenseClass, PinnedBase
from internal_ai.training.ownership import require_released_dataset
from internal_ai.training.provenance import provenance_mode
from internal_ai.workflows import latest_frozen_pass, model_card_markdown, promote_model


def register_alpha(
    platform: Platform,
    base: PinnedBase,
    adapter_path: Path,
    dataset_id: str,
    actor: Actor,
) -> ModelVersion:
    """Register an adapter as a candidate model.

    Registering is a governed act: only an ml lead may do it, only the base
    pinned in this package may be named, and only a released dataset may be
    claimed as the text it learned from.
    """
    require(platform.audit, actor, Action.REGISTER_MODEL, str(adapter_path))
    if base not in (PRIMARY, FALLBACK):
        raise PermissionError("alpha may load only the pinned base named in this package")
    if base.license_class != LicenseClass.PERMISSIVE:
        raise PermissionError(
            f"{base.repo_id} is {base.license} ({base.license_class}); registering an adapter "
            "on it needs an approved base_model_exception, and this install allows "
            f"{LicenseClass.PERMISSIVE}"
        )
    dataset = require_released_dataset(platform.store.get_dataset(dataset_id))
    digest = file_hash(adapter_path)
    card = alpha_card(
        base,
        RECIPE,
        dataset_id,
        dataset.content_hash,
        digest,
    )
    card["provenance_mode"] = provenance_mode(platform)
    record = reproducibility_beside(adapter_path)
    if record:
        card["reproducibility"] = record
        card["commit"] = record.get("commit")
        card["seed"] = record.get("seed")
        card["run_id"] = record.get("run_id")
        card["adapter_backend"] = record.get("adapter_backend")
    existing = platform.store.list_models(name=base.repo_id.split("/")[-1].lower())
    version = max((item.version for item in existing), default=0) + 1
    model = ModelVersion(
        id=new_id("mdl"),
        name=base.repo_id.split("/")[-1].lower(),
        version=version,
        content_hash=content_hash({"adapter": digest, "base": base.revision}),
        created_at=now_iso(),
        dataset_ids=[dataset_id] if dataset_id else [],
        config=card["training_config"],
        status="candidate",
        card=card,
        path=str(adapter_path),
    )
    platform.store.add_model(model)
    platform.audit.record(
        actor=actor.id,
        action="model.registered",
        subject=model.id,
        details={
            "track": "alpha",
            "base_repo_id": base.repo_id,
            "base_revision": base.revision,
            "adapter_sha256": digest,
        },
    )
    return model


def request_alpha_promotion(
    platform: Platform,
    actor: Actor,
    model_id: str,
    answer: str,
    hits: list[AskHit],
) -> Any:
    """Open the promotion gate. This does not decide it."""
    model = platform.store.get_model(model_id)
    if model is None:
        raise KeyError(f"unknown model: {model_id}")
    if not model.card.get("pretrained_weights_used"):
        raise ValueError("alpha promotion is only for an adapter on the pinned base")
    report = score_answer(answer, hits)
    return governance.request_gate(
        platform,
        actor,
        GateType.MODEL_PROMOTION,
        title=f"Promote alpha {model.name} v{model.version}",
        subject=model_id,
        evidence={
            "eval_results_vs_baseline": json.dumps(report.as_dict(), sort_keys=True),
            "model_card": model_card_markdown(model),
            "red_team_results": (
                "Citation check only: each line number must stay on the Ask hit it "
                f"belongs to. passed={report.passed}. The same check rejects line 36 "
                "on the Razorpay create-order route, because that line belongs to "
                "next.config.ts. No separate memorisation review was run. Alpha "
                "answers are not training data for the from-scratch track."
            ),
        },
    )


def _pii_leaks(answer: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Personal data in the answer that no cited line holds.

    An answer may repeat a person's details when the line it cites holds
    them. Anything else is the model putting personal data on its own.
    """
    allowed = "\n".join(str(hit.get("text", "")) for hit in hits)
    leaks: list[dict[str, str]] = []
    for kind, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(answer):
            if match.group(0) not in allowed:
                leaks.append({"kind": kind, "match": match.group(0)})
    return leaks


def _alpha_contamination(
    platform: Platform, model: ModelVersion, eval_set: EvalSet
) -> dict[str, Any]:
    """Was the eval set's evidence already in the text the adapter learned from?"""
    dataset_id = str(model.card.get("dataset_id") or "")
    rows = platform.store.dataset_records(dataset_id) if dataset_id else []
    training = [
        Record(
            id=str(row["id"]),
            record_key=str(row["record_key"]),
            text=str(row["payload"]),
            source_id=str(row["source_id"]),
            sensitivity=str(row["sensitivity"]),
            acl=list(row["acl"]),
            tokens=int(row["tokens"]),
            meta=dict(row["meta"]),
        )
        for row in rows
    ]
    if not training:
        return {
            "passed": False,
            "note": "no training corpus was supplied, so overlap with it could not be ruled out",
        }
    eval_records = [
        Record(
            id=f"{index}",
            record_key=f"{index}",
            text=str(hit.get("text", "")),
            source_id=eval_set.id,
        )
        for index, item in enumerate(eval_set.items)
        for hit in item.get("hits", [])
    ]
    if not eval_records:
        return {
            "passed": False,
            "note": (
                "the eval set cites no lines, so overlap with the training text "
                "could not be ruled out"
            ),
        }
    return contamination_report(eval_records, training)


def _score_answers(
    eval_set: EvalSet, answers: list[str], require_citation: bool = False
) -> tuple[int, list[str], list[dict[str, str]], list[float]]:
    """How many answers kept their citations, which ones did not, what leaked.

    The per-question flags are returned as well: promotion compares the adapter
    and the base question by question, and a mean alone cannot say whether the
    difference is larger than the noise it was measured in.

    ``require_citation`` is for the keyword baseline. The scorer asks whether a
    citation moved, so an answer that cites nothing passes it; a grep that found
    no citable line would then score full marks and the comparison would be
    worth nothing.
    """
    mismatches: list[str] = []
    leaks: list[dict[str, str]] = []
    kept: list[float] = []
    cited = 0
    for item, answer in zip(eval_set.items, answers, strict=True):
        hits = [AskHit(str(hit["path"]), int(hit["line"])) for hit in item.get("hits", [])]
        report = score_answer(answer, hits)
        mismatches.extend(report.mismatches)
        if report.passed and (not require_citation or LINE_NUMBER.search(answer)):
            cited += 1
        kept.append(1.0 if report.passed else 0.0)
        leaks.extend(_pii_leaks(answer, list(item.get("hits", []))))
    return cited, mismatches, leaks, kept


def keyword_baseline_answers(platform: Platform, eval_set: EvalSet, dataset_id: str) -> list[str]:
    """What a keyword search would answer, so the adapter has something dumb to beat.

    The corpus indexed here is the text the adapter learned from, never the whole
    dataset: indexing the held-out lines themselves would let a grep score full
    marks by finding the very answer it is supposed to be searching for.

    The answer is the top-ranked record's own key, which for a line-addressed
    corpus is its ``path:line``. Ties go to the earlier record so two runs of the
    same evaluation cannot disagree about what the baseline said.
    """
    from internal_ai.evals.metrics import bm25_scores

    rows = platform.store.dataset_records_by_split(dataset_id, "train")
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    if not rows:
        raise ValueError("a keyword baseline needs the corpus the adapter learned from")
    keys = [str(row["record_key"]) for row in rows]
    texts = [str(row["payload"]) for row in rows]
    answers: list[str] = []
    for item in eval_set.items:
        scores = bm25_scores(str(item.get("query", "")), texts)
        best = max(range(len(scores)), key=lambda index: (scores[index], -index))
        answers.append(f"The closest passage is {keys[best]}.")
    return answers


def evaluate_alpha(
    platform: Platform,
    model_id: str,
    eval_set_id: str,
    answers: list[str],
    baseline_answers: list[str],
    actor: str = "evaluation",
) -> EvalReport:
    """Score the alpha model against a frozen eval set and record the report.

    The platform holds no generator: the on-prem runner produces the answers
    and hands them back here to be scored. The set is frozen before the model
    is run against it, because a report on questions that can still be changed
    after the model has seen them is worth nothing.

    The same runner must also produce the un-adapted base's answers, and they
    are required rather than optional. An adapter that cannot beat the model it
    was layered onto is not an improvement, and a report that never asked the
    question would let one through by having nothing to compare against.
    """
    model = platform.store.get_model(model_id)
    if model is None:
        raise KeyError(f"unknown model: {model_id}")
    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise KeyError(f"unknown eval set: {eval_set_id}")
    if not eval_set.frozen:
        raise PermissionError("alpha evaluation runs against a frozen eval set (FR-14)")
    total = len(eval_set.items)
    for label, supplied in (("answers", answers), ("baseline answers", baseline_answers)):
        if len(supplied) != total:
            raise ValueError(
                f"the eval set has {total} questions and {len(supplied)} {label} were given"
            )
    cited, mismatches, leaks, kept = _score_answers(eval_set, answers)
    base_cited, base_mismatches, _, base_kept = _score_answers(eval_set, baseline_answers)
    rate = round(cited / total, 6) if total else 0.0
    base_rate = round(base_cited / total, 6) if total else 0.0
    if not model.dataset_ids:
        raise ValueError(
            "alpha evaluation needs the dataset the adapter learned from, so a keyword "
            "search can be scored against the same corpus"
        )
    keyword_answers = keyword_baseline_answers(platform, eval_set, str(model.dataset_ids[0]))
    keyword_cited, keyword_mismatches, _, keyword_kept = _score_answers(
        eval_set, keyword_answers, require_citation=True
    )
    keyword_rate = round(keyword_cited / total, 6) if total else 0.0
    if cited > base_cited:
        verdict = (
            f"the adapter kept {cited} of {total} citations and its un-adapted base kept "
            f"{base_cited}"
        )
    elif answers == baseline_answers:
        verdict = (
            "the adapter and its base returned the same answers, which usually means the "
            "adapter was not loaded; nothing here can promote it"
        )
    elif cited == base_cited:
        verdict = (
            f"the adapter kept {cited} of {total} citations and its base kept the same "
            f"{base_cited}; a set that cannot tell the two apart cannot promote the adapter"
        )
    else:
        verdict = (
            f"the adapter kept {cited} of {total} citations and its un-adapted base kept "
            f"{base_cited}; an adapter that loses to its base is not an improvement"
        )
    keyword_verdict = (
        f"the adapter kept {cited} of {total} citations and a keyword search kept {keyword_cited}"
        if cited > keyword_cited
        else (
            f"a keyword search kept {keyword_cited} of {total} citations and the adapter kept "
            f"{cited}; an adapter a grep already matches has earned nothing"
        )
    )
    checks = {
        "citations": {"passed": not mismatches, "mismatches": mismatches[:10]},
        "pii_regurgitation": {"passed": not leaks, "leaks": leaks[:10]},
        "contamination": _alpha_contamination(platform, model, eval_set),
        "beats_base": {
            "passed": cited > base_cited,
            "baseline": "base_model",
            "adapter_citation_pass_rate": rate,
            "base_citation_pass_rate": base_rate,
            "mismatches_on_base": base_mismatches[:10],
            "note": verdict,
        },
        "beats_keyword": {
            "passed": cited > keyword_cited,
            "baseline": "bm25",
            "adapter_citation_pass_rate": rate,
            "keyword_citation_pass_rate": keyword_rate,
            "mismatches_on_keyword": keyword_mismatches[:10],
            "note": keyword_verdict,
        },
    }
    metrics = {"citation_pass_rate": rate}
    baseline_metrics = {"citation_pass_rate": base_rate}
    passed = all(bool(check.get("passed")) for check in checks.values())
    eval_report = EvalReport(
        id=new_id("rep"),
        model_id=model_id,
        eval_set_id=eval_set_id,
        created_at=now_iso(),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        per_case={
            "citation_pass_rate": {
                "candidate": kept,
                "baseline": base_kept,
                "keyword": keyword_kept,
            }
        },
        checks=checks,
        passed=passed,
        notes=(
            "every answer kept its line numbers on the files the eval set named, and the "
            f"adapter kept more of them than its base ({cited} of {total} against {base_cited}) "
            f"and more than a keyword search kept ({keyword_cited})"
            if passed
            else "an answer moved a citation, leaked personal data, overlapped the training "
            f"text, or did not beat the un-adapted base ({cited} of {total} against {base_cited}) "
            f"and the keyword search ({keyword_cited})"
        ),
    )
    platform.store.add_eval_report(eval_report)
    platform.audit.record(
        actor=actor,
        action="eval.run",
        subject=eval_set_id,
        outcome=AuditOutcome.SUCCESS if passed else AuditOutcome.FAILURE,
        details={
            "model_id": model_id,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "passed": passed,
            "checks": {name: bool(check.get("passed")) for name, check in checks.items()},
        },
    )
    return eval_report


def serve_alpha(platform: Platform, model_id: str, runner_dir: Path) -> Path:
    """Write the runner manifest after the promotion gate is approved.

    The local runner (Ollama on the training PC) owns the KV cache. This
    repository does not implement one.

    Serving is a promotion, so it follows the same rule as every other model:
    an approved gate is not enough on its own. A passing report on a set that
    is still frozen is required too, and the promotion is recorded through the
    shared path so the card, the status and the audit all say the same thing.
    """
    model = platform.store.get_model(model_id)
    if model is None:
        raise KeyError(f"unknown model: {model_id}")
    gate = next(
        (
            item
            for item in platform.store.list_gates(status="approved")
            if str(item.gate_type) == str(GateType.MODEL_PROMOTION) and item.subject == model_id
        ),
        None,
    )
    if gate is None:
        raise PermissionError("alpha serving requires an approved model_promotion gate")
    report = latest_frozen_pass(platform, model_id)
    if report is None:
        raise PermissionError(
            "alpha serving requires a passing eval report on a frozen eval set (FR-15)"
        )
    if report.checks.get("beats_base", {}).get("passed") is not True:
        raise PermissionError(
            "alpha serving requires an eval report that beat the un-adapted base on the same "
            "frozen set"
        )
    if report.checks.get("beats_keyword", {}).get("passed") is not True:
        raise PermissionError(
            "alpha serving requires an eval report that beat a keyword search on the same frozen "
            "set; an adapter a grep already matches has earned nothing"
        )
    approver = next(
        (
            decision.approver
            for decision in reversed(platform.store.gate_decisions(gate.id))
            if decision.approved
        ),
        "ml_lead",
    )
    promote_model(platform, model_id, gate_approved=True, approver=approver)
    model = platform.store.get_model(model_id)
    runner_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model.id,
        "adapter": model.path,
        "base_repo_id": model.card.get("base_repo_id"),
        "base_revision": model.card.get("base_revision"),
        "adapter_sha256": model.card.get("adapter_sha256"),
        "eval_report_id": model.card.get("eval_report_id"),
        "kv_cache": "local-runner",
        "gate_id": gate.id,
    }
    path = runner_dir / "alpha-runner.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    platform.audit.record(
        actor="alpha",
        action="serving.prepared",
        subject=model_id,
        details={"manifest": str(path), "gate_id": gate.id},
    )
    return path


def citation_report(answer: str, hits: list[AskHit]) -> CitationReport:
    return score_answer(answer, hits)
