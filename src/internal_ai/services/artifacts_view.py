"""The console's view of an artifact, and the check that it still is one.

A digest on a model card only means something if somebody can ask "and is it still
that?" without the shell. This page is that question: it lists what each card
records and re-runs the same ``require_intact`` the loader runs, so a directory that
drifted after promotion is visible before it answers a question.
"""

from __future__ import annotations

from typing import Any

from internal_ai.artifacts import (
    artifact_from_card,
    forget_verified,
    has_manifest,
    read_manifest,
    require_intact,
)
from internal_ai.context import Platform
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require


def rows(platform: Platform) -> list[dict[str, Any]]:
    """One row per model, from the card alone: nothing is hashed to draw this page."""
    listed: list[dict[str, Any]] = []
    for model in platform.store.list_models():
        card = dict(model.card or {})
        artifact = artifact_from_card(card)
        listed.append(
            {
                "id": model.id,
                "model": model.name,
                "version": model.version,
                "status": str(model.status),
                "recorded": artifact is not None,
                "digest": artifact.digest if artifact is not None else "",
                "files": len(artifact.files)
                if artifact is not None
                else int(
                    (card.get("artifact") or {}).get("files") or 0
                    if isinstance(card.get("artifact"), dict)
                    else 0
                ),
                "size": artifact.size if artifact is not None else 0,
                "signed": bool(artifact is not None and artifact.signature),
                "key_id": artifact.key_id if artifact is not None else "",
                "path": model.path,
            }
        )
    return listed


def verify(platform: Platform, actor: Actor, model_id: str) -> dict[str, Any]:
    """Re-check one model's directory against its card, signature first.

    ``forget_verified`` matters here: the loader caches a pass for the life of the
    process, and an operator asking now wants the answer now.
    """
    require(platform.audit, actor, Action.VERIFY_ARTIFACT, model_id)
    model = platform.store.get_model(model_id)
    if model is None:
        raise ValueError(f"no model {model_id}")
    card = dict(model.card or {})
    if not has_manifest(card):
        raise ValueError(
            f"{model.name} v{model.version} records no artifact manifest, so there "
            "is nothing to check; the card predates stored artifacts"
        )
    forget_verified()
    require_intact(platform.blobs, model.path, card, signer=platform.audit.signer)
    artifact = artifact_from_card(card)
    assert artifact is not None  # has_manifest just proved it
    files = read_manifest(platform.blobs, artifact.manifest_key)
    return {
        "model": model.name,
        "version": model.version,
        "digest": artifact.digest,
        "files": len(files),
        "size": sum(item.size for item in files),
        "signed": bool(artifact.signature),
    }
