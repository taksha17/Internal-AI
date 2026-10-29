"""A released feedback export is the corpus a later run may use."""

from __future__ import annotations

import pytest
from conftest import enroll

from internal_ai.data.pipeline import DataPipeline
from internal_ai.domain import DatasetVersion
from internal_ai.feedback import export_feedback, file_feedback
from internal_ai.ids import now_iso
from internal_ai.programme import customer_stages
from internal_ai.training.ownership import require_released_dataset


def test_an_unreleased_export_is_not_training_data(platform) -> None:
    owner = enroll(platform, "Sam Owner", ["data_owner"], ["engineering"])
    file_feedback(platform, owner, "where is the host", "src/app.py", 4, "ds_ask")
    dataset = export_feedback(platform, owner)
    assert dataset.released is False
    assert dataset.stats["kind"] == "feedback"
    assert dataset.stats["training_data"] is False


def test_the_release_gate_marks_it_as_training_data(platform) -> None:
    owner = enroll(platform, "Sam Owner", ["data_owner"], ["engineering"])
    file_feedback(platform, owner, "where is the host", "src/app.py", 4, "ds_ask")
    dataset = export_feedback(platform, owner)
    released = DataPipeline(platform.store, platform.audit, platform.settings.root).release(
        dataset.id, owner.id, gate_approved=True
    )
    assert released.released is True
    assert released.stats["kind"] == "feedback"
    assert released.stats["training_data"] is True
    events = platform.store.list_audit(action="dataset.training_data_marked", limit=5)
    assert events and events[-1].subject == dataset.id


def test_the_data_stage_links_the_newest_released_dataset(platform) -> None:
    owner = enroll(platform, "Sam Owner", ["data_owner"], ["engineering"])
    file_feedback(platform, owner, "where is the host", "src/app.py", 4, "ds_ask")
    dataset = export_feedback(platform, owner)
    stages = {stage["name"]: stage for stage in customer_stages(platform)}
    assert stages["Data"]["href"] == "/console/datasets"
    assert stages["Data"]["done"] is False
    DataPipeline(platform.store, platform.audit, platform.settings.root).release(
        dataset.id, owner.id, gate_approved=True
    )
    stages = {stage["name"]: stage for stage in customer_stages(platform)}
    assert stages["Data"]["href"] == f"/console/datasets/{dataset.id}"
    assert stages["Data"]["done"] is True


def test_training_refuses_a_set_marked_not_training_data(platform) -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_export",
            name="ask-feedback",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=10,
            created_at=now_iso(),
            released=True,
            stats={"kind": "feedback", "training_data": False},
        )
    )
    with pytest.raises(PermissionError, match="not training data"):
        require_released_dataset(platform.store.get_dataset("ds_export"))
