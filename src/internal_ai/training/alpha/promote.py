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
from internal_ai.services.permissions import Action, require
from internal_ai.training.alpha.card import alpha_card
from internal_ai.training.alpha.citations import AskHit, CitationReport, score_answer
from internal_ai.training.alpha.pin import FALLBACK, PRIMARY, RECIPE, LicenseClass, PinnedBase
from internal_ai.training.ownership import require_released_dataset
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


def evaluate_alpha(
    platform: Platform,
    model_id: str,
    eval_set_id: str,
    answers: list[str],
    actor: str = "evaluation",
) -> EvalReport:
    """Score the alpha model against a frozen eval set and record the report.

    The platform holds no generator: the on-prem runner produces the answers
    and hands them back here to be scored. The set is frozen before the model
    is run against it, because a report on questions that can still be changed
    after the model has seen them is worth nothing.
    """
    model = platform.store.get_model(model_id)
    if model is None:
        raise KeyError(f"unknown model: {model_id}")
    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise KeyError(f"unknown eval set: {eval_set_id}")
    if not eval_set.frozen:
        raise PermissionError("alpha evaluation runs against a frozen eval set (FR-14)")
    if len(answers) != len(eval_set.items):
        raise ValueError(
            f"the eval set has {len(eval_set.items)} questions "
            f"and {len(answers)} answers were given"
        )
    mismatches: list[str] = []
    leaks: list[dict[str, str]] = []
    cited = 0
    for item, answer in zip(eval_set.items, answers, strict=True):
        hits = [AskHit(str(hit["path"]), int(hit["line"])) for hit in item.get("hits", [])]
        report = score_answer(answer, hits)
        mismatches.extend(report.mismatches)
        if report.passed:
            cited += 1
        leaks.extend(_pii_leaks(answer, list(item.get("hits", []))))
    checks = {
        "citations": {"passed": not mismatches, "mismatches": mismatches[:10]},
        "pii_regurgitation": {"passed": not leaks, "leaks": leaks[:10]},
        "contamination": _alpha_contamination(platform, model, eval_set),
    }
    total = len(eval_set.items)
    metrics = {"citation_pass_rate": round(cited / total, 6) if total else 0.0}
    passed = all(bool(check.get("passed")) for check in checks.values())
    eval_report = EvalReport(
        id=new_id("rep"),
        model_id=model_id,
        eval_set_id=eval_set_id,
        created_at=now_iso(),
        metrics=metrics,
        baseline_metrics={},
        checks=checks,
        passed=passed,
        notes=(
            "every answer kept its line numbers on the files the eval set named"
            if passed
            else "an answer moved a citation, leaked personal data, or overlapped the training text"
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
    if latest_frozen_pass(platform, model_id) is None:
        raise PermissionError(
            "alpha serving requires a passing eval report on a frozen eval set (FR-15)"
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
