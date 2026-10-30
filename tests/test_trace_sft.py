"""A failed answer supplies the question. The person supplies the answer."""

from __future__ import annotations

import pytest

from internal_ai.services import triage as triage_service
from internal_ai.services.permissions import PermissionDenied
from internal_ai.serving.service import Principal
from internal_ai.serving.trace import FAILURE_CITATION, record_trace

ENGINEERING = frozenset({"engineering"})
QUESTION = "why did the ledger fail?"
CITED = [{"path": "docs/ledger.md", "line": 12, "text": "the ledger replays at midnight"}]
GOOD = "See docs/ledger.md at line 12: the nightly replay reconciles the feed."
MOVED = "See docs/ledger.md at line 99."


def failed_trace(platform, user_id: str, question: str = QUESTION) -> str:
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=user_id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question=question,
        answer=MOVED,
        citations=[{"path": "docs/ledger.md", "line": 12}],
        failure=FAILURE_CITATION,
    )
    return trace.id


def test_a_correction_becomes_the_first_version(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)

    dataset = triage_service.sft_example_from_trace(platform, lead, trace_id, GOOD)

    assert dataset.name == "trace-sft"
    assert dataset.version == 1
    assert dataset.released is False
    assert dataset.lineage["trace_ids"] == [trace_id]
    rows = platform.store.dataset_records(dataset.id)
    assert len(rows) == 1
    assert rows[0]["meta"]["instruction"] == QUESTION
    assert rows[0]["meta"]["response"] == GOOD
    assert rows[0]["meta"]["authored_by"] == lead.id
    events = platform.store.conn.execute(
        "SELECT subject, details FROM audit_events WHERE action = 'sft.example_from_trace'"
    ).fetchall()
    assert [row["subject"] for row in events] == [dataset.id]
    assert trace_id in events[0]["details"]


def test_the_next_example_makes_version_two_that_carries_the_first(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    first = failed_trace(platform, lead.id, "why did the ledger fail?")
    second = failed_trace(platform, lead.id, "who owns the ledger?")
    one = triage_service.sft_example_from_trace(platform, lead, first, GOOD)
    two = triage_service.sft_example_from_trace(platform, lead, second, GOOD)

    assert two.version == 2
    rows = platform.store.dataset_records(two.id)
    assert [row["meta"]["instruction"] for row in rows] == [
        "why did the ledger fail?",
        "who owns the ledger?",
    ]
    assert len(platform.store.dataset_records(one.id)) == 1


def test_the_failed_answer_itself_is_refused_as_a_correction(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)

    with pytest.raises(ValueError, match="not a correction"):
        triage_service.sft_example_from_trace(platform, lead, trace_id, MOVED)


def test_a_correction_that_moves_the_line_is_refused(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)

    with pytest.raises(ValueError, match="must cite a line"):
        triage_service.sft_example_from_trace(
            platform, lead, trace_id, "See docs/other.md at line 12."
        )


def test_an_empty_correction_is_refused(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)

    with pytest.raises(ValueError, match="needs the answer"):
        triage_service.sft_example_from_trace(platform, lead, trace_id, "   ")


def test_an_answer_that_passed_is_not_a_correction(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=lead.id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question=QUESTION,
        answer=GOOD,
        citations=CITED,
    )

    with pytest.raises(ValueError, match="only a failed answer"):
        triage_service.sft_example_from_trace(platform, lead, trace.id, GOOD)


def test_the_same_trace_is_not_added_twice(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)
    triage_service.sft_example_from_trace(platform, lead, trace_id, GOOD)

    with pytest.raises(ValueError, match="already has an example"):
        triage_service.sft_example_from_trace(platform, lead, trace_id, GOOD)


def test_another_group_cannot_correct_a_trace(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    outsider = enroll_person(platform, "Fran Lead", ["ml_lead"], groups=["finance"])
    trace_id = failed_trace(platform, owner.id)

    with pytest.raises(PermissionError, match="group"):
        triage_service.sft_example_from_trace(platform, outsider, trace_id, GOOD)


def test_building_a_dataset_is_required(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    expert = enroll_person(platform, "Eve Expert", ["domain_expert"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)

    with pytest.raises(PermissionDenied):
        triage_service.sft_example_from_trace(platform, expert, trace_id, GOOD)


def test_an_unknown_trace_is_a_key_error_not_a_new_dataset(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])

    with pytest.raises(KeyError):
        triage_service.sft_example_from_trace(platform, lead, "trc_missing", GOOD)
    assert platform.store.list_datasets(name="trace-sft") == []
