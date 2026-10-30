"""A failed answer becomes the next version of a regression set, never an edit."""

from __future__ import annotations

import pytest

from internal_ai.evals.harness import EvalHarness
from internal_ai.services import triage as triage_service
from internal_ai.services.permissions import PermissionDenied
from internal_ai.serving.service import Principal
from internal_ai.serving.trace import FAILURE_CITATION, record_trace

ENGINEERING = frozenset({"engineering"})


def failed_trace(platform, user_id: str, question: str = "why did the ledger fail?") -> str:
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=user_id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question=question,
        answer="See docs/ledger.md at line 99.",
        citations=[{"path": "docs/ledger.md", "line": 12}],
        failure=FAILURE_CITATION,
    )
    return trace.id


def test_a_failed_trace_becomes_the_first_version(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)

    eval_set = triage_service.eval_case_from_trace(platform, lead, trace_id)

    assert eval_set.name == "trace-regressions"
    assert eval_set.version == 1
    assert eval_set.frozen is False
    assert len(eval_set.items) == 1
    item = eval_set.items[0]
    assert item["query"] == "why did the ledger fail?"
    assert item["hits"] == [{"path": "docs/ledger.md", "line": 12}]
    assert item["trace_ids"] == [trace_id]
    assert item["failure"] == FAILURE_CITATION
    events = platform.store.conn.execute(
        "SELECT subject, details FROM audit_events WHERE action = 'eval.case_from_trace'"
    ).fetchall()
    assert [row["subject"] for row in events] == [eval_set.id]
    assert trace_id in events[0]["details"]


def test_the_next_trace_makes_version_two_that_carries_the_first(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    first = failed_trace(platform, lead.id, "why did the ledger fail?")
    second = failed_trace(platform, lead.id, "who owns the ledger?")
    one = triage_service.eval_case_from_trace(platform, lead, first)
    two = triage_service.eval_case_from_trace(platform, lead, second)

    assert two.version == 2
    assert [item["trace_ids"][0] for item in two.items] == [first, second]
    unchanged = platform.store.get_eval_set(one.id)
    assert len(unchanged.items) == 1


def test_a_frozen_set_is_copied_forward_not_edited(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    first = failed_trace(platform, lead.id, "why did the ledger fail?")
    one = triage_service.eval_case_from_trace(platform, lead, first)
    harness = EvalHarness(platform.store, platform.audit)
    harness.freeze_eval_set(one.id, "ml_lead", gate_approved=True)
    before = platform.store.get_eval_set(one.id)

    second = failed_trace(platform, lead.id, "who owns the ledger?")
    two = triage_service.eval_case_from_trace(platform, lead, second)

    after = platform.store.get_eval_set(one.id)
    assert after.frozen is True
    assert after.items == before.items
    assert after.content_hash == before.content_hash
    assert two.frozen is False
    assert len(two.items) == 2


def test_the_same_trace_is_not_added_twice(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)
    triage_service.eval_case_from_trace(platform, lead, trace_id)
    with pytest.raises(ValueError, match="already a case"):
        triage_service.eval_case_from_trace(platform, lead, trace_id)


def test_an_answer_that_passed_is_not_a_regression_case(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=lead.id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question="why did the ledger fail?",
        citations=[{"path": "docs/ledger.md", "line": 12}],
    )
    with pytest.raises(ValueError, match="only a failed answer"):
        triage_service.eval_case_from_trace(platform, lead, trace.id)


def test_a_trace_with_no_citation_cannot_be_asserted_against(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=lead.id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question="why did the ledger fail?",
        failure=FAILURE_CITATION,
    )
    with pytest.raises(ValueError, match="cites no line"):
        triage_service.eval_case_from_trace(platform, lead, trace.id)


def test_another_group_cannot_turn_a_trace_into_a_case(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    outsider = enroll_person(platform, "Fran Lead", ["ml_lead"], groups=["finance"])
    trace_id = failed_trace(platform, owner.id)
    with pytest.raises(PermissionError, match="group"):
        triage_service.eval_case_from_trace(platform, outsider, trace_id)


def test_the_capability_is_required(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    owner = enroll_person(platform, "Dana Owner", ["data_owner"], groups=["engineering"])
    trace_id = failed_trace(platform, lead.id)
    with pytest.raises(PermissionDenied):
        triage_service.eval_case_from_trace(platform, owner, trace_id)
