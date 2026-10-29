"""Demote returns a model to candidate and leaves its weights where they are."""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.domain import ModelVersion
from internal_ai.services.permissions import Action, PermissionDenied, require
from internal_ai.workflows import demote_model


def test_demote_writes_one_audit_event_and_keeps_the_file(platform, tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    platform.store.add_model(
        ModelVersion(
            id="mdl_live",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            status="promoted",
            path=str(weights),
        )
    )

    model = demote_model(platform, "mdl_live", "ppl_marcus")

    assert model.status == "candidate"
    assert weights.read_bytes() == b"weights"
    events = platform.store.list_audit(action="model.demoted")
    assert len(events) == 1
    assert events[0].subject == "mdl_live"


def test_someone_who_cannot_promote_cannot_demote(platform) -> None:
    from conftest import enroll

    owner = enroll(platform, "Ada Owner", ["data_owner"])
    with pytest.raises(PermissionDenied):
        require(platform.audit, owner, Action.PROMOTE_MODEL, "mdl_live")
