"""The serving path may only ever answer with a model that was promoted."""

from __future__ import annotations

import pytest

from internal_ai.domain import ModelVersion
from internal_ai.serving.service import ModelRouter, served_model


def _model(model_id: str, name: str, status: str, created_at: str) -> ModelVersion:
    return ModelVersion(
        id=model_id,
        name=name,
        version=1,
        content_hash="h",
        created_at=created_at,
        status=status,
    )


def test_a_candidate_is_never_served(platform) -> None:
    platform.store.add_model(_model("mdl_candidate", "embedding-v1", "candidate", "2026-01-01"))
    router = ModelRouter(platform.store, platform.audit)
    assert router.resolve("embedding-v1") is None
    assert served_model(platform.store, router) is None


def test_a_candidate_is_never_activated(platform) -> None:
    platform.store.add_model(_model("mdl_candidate", "embedding-v1", "candidate", "2026-01-01"))
    router = ModelRouter(platform.store, platform.audit)
    with pytest.raises(PermissionError, match="promoted"):
        router.set_active("embedding-v1", "mdl_candidate", "admin", "deploy")
    assert router.resolve("embedding-v1") is None


def test_an_active_pointer_cannot_smuggle_a_candidate_into_service(platform) -> None:
    platform.store.add_model(_model("mdl_candidate", "embedding-v1", "candidate", "2026-01-01"))
    platform.store.add_model(_model("mdl_promoted", "embedding-v1", "promoted", "2026-01-02"))
    router = ModelRouter(platform.store, platform.audit)
    router.active["embedding-v1"] = "mdl_candidate"
    assert router.resolve("embedding-v1").id == "mdl_promoted"


def test_a_promoted_model_is_served(platform) -> None:
    platform.store.add_model(_model("mdl_promoted", "embedding-v1", "promoted", "2026-01-02"))
    router = ModelRouter(platform.store, platform.audit)
    assert router.resolve("embedding-v1").id == "mdl_promoted"
    assert served_model(platform.store, router).id == "mdl_promoted"
