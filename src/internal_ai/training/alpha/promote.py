"""Register an alpha adapter and ask humans to promote it. Serving waits on that gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import GateType, ModelVersion
from internal_ai.ids import content_hash, file_hash, new_id, now_iso
from internal_ai.services import governance
from internal_ai.services.actor import Actor
from internal_ai.training.alpha.card import alpha_card
from internal_ai.training.alpha.citations import AskHit, CitationReport, score_answer
from internal_ai.training.alpha.pin import RECIPE, PinnedBase
from internal_ai.workflows import model_card_markdown


def register_alpha(
    platform: Platform,
    base: PinnedBase,
    adapter_path: Path,
    dataset_id: str,
    actor: str,
) -> ModelVersion:
    dataset = platform.store.get_dataset(dataset_id)
    digest = file_hash(adapter_path)
    card = alpha_card(
        base,
        RECIPE,
        dataset_id,
        dataset.content_hash if dataset else None,
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
        actor=actor,
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


def serve_alpha(platform: Platform, model_id: str, runner_dir: Path) -> Path:
    """Write the runner manifest after the promotion gate is approved.

    The local runner (Ollama on the training PC) owns the KV cache. This
    repository does not implement one.
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
    runner_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model.id,
        "adapter": model.path,
        "base_repo_id": model.card.get("base_repo_id"),
        "base_revision": model.card.get("base_revision"),
        "adapter_sha256": model.card.get("adapter_sha256"),
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
