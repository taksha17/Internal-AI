"""A preference is a person's choice between two answers, not model output."""

from __future__ import annotations

import pytest
from conftest import enroll

from internal_ai.compute.advise import attach_backend
from internal_ai.data.pipeline import DataPipeline
from internal_ai.preferences import (
    export_preferences,
    file_preference,
    list_preferences,
    train_preference,
)


def _expert(platform, name: str = "Elena Expert"):
    return enroll(platform, name, ["domain_expert"], ["engineering"])


def _file(platform, expert, **overrides):
    values = {
        "question": "how do I reset the vpn",
        "answer_a": "open the profile and sign in again",
        "answer_b": "restart the laptop",
        "preferred": "a",
        "dataset_id": "ds_ask",
    }
    values.update(overrides)
    return file_preference(platform, expert, **values)


def test_a_model_author_is_refused_and_audited(platform) -> None:
    expert = _expert(platform)
    with pytest.raises(ValueError, match="not a preference a person filed"):
        _file(platform, expert, authored_by="model")
    events = platform.store.list_audit(action="preference.rejected", limit=5)
    assert events and events[-1].outcome == "denied"
    assert list_preferences(platform) == []


def test_an_alpha_source_is_refused(platform) -> None:
    expert = _expert(platform)
    with pytest.raises(ValueError, match="not a preference a person filed"):
        _file(platform, expert, source_id="alpha-run-7")
    assert list_preferences(platform) == []


def test_filing_needs_the_domain_expert_role(platform) -> None:
    author = enroll(platform, "Mo Model", ["ml_lead"], ["engineering"])
    with pytest.raises(PermissionError, match="domain_expert"):
        _file(platform, author)
    assert list_preferences(platform) == []


def test_a_preference_records_the_person_and_the_choice(platform) -> None:
    expert = _expert(platform)
    row = _file(platform, expert)
    assert row["authored_by"] == expert.id
    assert row["preferred"] == "a"
    assert list_preferences(platform)[0]["id"] == row["id"]
    events = platform.store.list_audit(action="preference.filed", limit=5)
    assert events and events[-1].subject == row["id"]


def test_the_export_is_unreleased_and_not_training_data(platform) -> None:
    expert = _expert(platform)
    _file(platform, expert)
    dataset = export_preferences(platform, expert)
    assert dataset.released is False
    assert dataset.stats["kind"] == "preference"
    assert dataset.stats["training_data"] is False
    assert dataset.record_count == 1


def test_training_refuses_before_the_set_is_released(platform) -> None:
    expert = _expert(platform)
    lead = enroll(platform, "Mo Lead", ["ml_lead"], ["engineering"])
    _file(platform, expert)
    dataset = export_preferences(platform, expert)
    with pytest.raises(PermissionError, match="release the preference set"):
        train_preference(platform, lead, dataset.id)


def test_release_marks_the_set_as_training_data_and_training_still_waits_for_a_backend(
    platform,
) -> None:
    expert = _expert(platform)
    lead = enroll(platform, "Mo Lead", ["ml_lead"], ["engineering"])
    _file(platform, expert)
    dataset = export_preferences(platform, expert)
    released = DataPipeline(platform.store, platform.audit, platform.settings.root).release(
        dataset.id, lead.id, gate_approved=True
    )
    assert released.released is True
    assert released.stats["kind"] == "preference"
    assert released.stats["training_data"] is True
    with pytest.raises(PermissionError, match="attach a compute backend"):
        train_preference(platform, lead, dataset.id)
    attach_backend(platform, "local", actor=lead.id)
    with pytest.raises(NotImplementedError, match="not implemented"):
        train_preference(platform, lead, dataset.id)
