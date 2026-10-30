"""A trace supplies the pair. The expert supplies the preference."""

from __future__ import annotations

import pytest
from conftest import enroll

from internal_ai.preferences import export_preferences, list_preferences
from internal_ai.services import triage as triage_service
from internal_ai.services.permissions import PermissionDenied
from internal_ai.serving.service import Principal
from internal_ai.serving.trace import FAILURE_CITATION, record_trace

ENGINEERING = frozenset({"engineering"})
QUESTION = "why did the ledger fail?"
GOOD = "See docs/ledger.md at line 12: the nightly replay reconciles the feed."
MOVED = "See docs/ledger.md at line 99."


def expert(platform, name: str = "Elena Expert", groups=("engineering",)):
    return enroll(platform, name, ["domain_expert"], list(groups))


def failed_trace(platform, user_id: str) -> str:
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=user_id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question=QUESTION,
        answer=MOVED,
        citations=[{"path": "docs/ledger.md", "line": 12}],
        failure=FAILURE_CITATION,
    )
    return trace.id


def test_an_expert_prefers_their_answer_to_the_failed_one(platform) -> None:
    elena = expert(platform)
    trace_id = failed_trace(platform, elena.id)

    row = triage_service.preference_from_trace(platform, elena, trace_id, GOOD)

    assert row["question"] == QUESTION
    assert row["answer_a"] == GOOD
    assert row["answer_b"] == MOVED
    assert row["preferred"] == "a"
    assert row["authored_by"] == elena.id
    assert [item["id"] for item in list_preferences(platform)] == [row["id"]]


def test_the_export_carries_the_trace_id(platform) -> None:
    elena = expert(platform)
    trace_id = failed_trace(platform, elena.id)
    triage_service.preference_from_trace(platform, elena, trace_id, GOOD)

    dataset = export_preferences(platform, elena)

    assert dataset.lineage["trace_ids"] == [trace_id]
    assert dataset.released is False
    assert dataset.stats["training_data"] is False


def test_an_expert_may_read_a_trace_after_the_role_was_added(platform) -> None:
    elena = expert(platform)
    trace_id = failed_trace(platform, elena.id)

    assert triage_service.sft_draft(platform, elena, trace_id).id == trace_id


def test_a_role_without_view_traces_is_still_refused(platform) -> None:
    owner = enroll(platform, "Omar Owner", ["data_owner"], ["engineering"])
    elena = expert(platform)
    trace_id = failed_trace(platform, elena.id)

    with pytest.raises(PermissionDenied):
        triage_service.preference_from_trace(platform, owner, trace_id, GOOD)
    assert list_preferences(platform) == []


def test_another_group_cannot_prefer_an_answer(platform) -> None:
    elena = expert(platform)
    fran = expert(platform, "Fran Expert", ("finance",))
    trace_id = failed_trace(platform, elena.id)

    with pytest.raises(PermissionError, match="group"):
        triage_service.preference_from_trace(platform, fran, trace_id, GOOD)
    assert list_preferences(platform) == []


def test_reading_a_trace_does_not_let_a_lead_file_the_preference(platform) -> None:
    elena = expert(platform)
    lead = enroll(platform, "Mo Lead", ["ml_lead"], ["engineering"])
    trace_id = failed_trace(platform, elena.id)

    with pytest.raises(PermissionError, match="domain_expert"):
        triage_service.preference_from_trace(platform, lead, trace_id, GOOD)
    assert list_preferences(platform) == []


def test_a_passing_answer_is_not_a_preference_pair(platform) -> None:
    elena = expert(platform)
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=elena.id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question=QUESTION,
        answer=GOOD,
        citations=[{"path": "docs/ledger.md", "line": 12}],
    )

    with pytest.raises(ValueError, match="only a failed answer"):
        triage_service.preference_from_trace(platform, elena, trace.id, MOVED)


def test_repeating_the_failed_answer_is_not_a_preference(platform) -> None:
    elena = expert(platform)
    trace_id = failed_trace(platform, elena.id)

    with pytest.raises(ValueError, match="not a correction"):
        triage_service.preference_from_trace(platform, elena, trace_id, MOVED)
    assert list_preferences(platform) == []


def test_an_uncited_correction_is_not_a_preference(platform) -> None:
    elena = expert(platform)
    trace_id = failed_trace(platform, elena.id)

    with pytest.raises(ValueError, match="must cite a line"):
        triage_service.preference_from_trace(
            platform, elena, trace_id, "See docs/other.md at line 12."
        )
