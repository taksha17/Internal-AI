"""A failed answer enters the learning pipeline through quarantine, never directly.

Proposing is intake: the trace is held, not used. Clearing takes a second
person. Conversions and supervision ingest refuse anything that skipped the
hold, and the model's failed answer is never training data.
"""

from __future__ import annotations

import pytest

from internal_ai.data.supervision import require_supervision_row
from internal_ai.services import triage as triage_service
from internal_ai.services.triage import clear_quarantined_trace, quarantine_trace
from internal_ai.serving.service import Principal
from internal_ai.serving.trace import (
    FAILURE_CITATION,
    record_trace,
    trace_quarantine_state,
)

ENGINEERING = frozenset({"engineering"})
QUESTION = "why did the ledger fail?"
GOOD = "See docs/ledger.md at line 12: the nightly replay reconciles the feed."
MOVED = "See docs/ledger.md at line 99."


def failed_trace(platform, user_id: str) -> str:  # type: ignore[no-untyped-def]
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


def people(platform, enroll_person):  # type: ignore[no-untyped-def]
    proposer = enroll_person(platform, "Ada Lead", ["ml_lead"], groups=["engineering"])
    clearer = enroll_person(
        platform, "Bo Lead", ["ml_lead", "domain_expert"], groups=["engineering"]
    )
    return proposer, clearer


def test_proposing_holds_the_trace_for_learning(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    proposer, _clearer = people(platform, enroll_person)
    trace_id = failed_trace(platform, proposer.id)

    assert trace_quarantine_state(platform.store, trace_id) is None
    quarantine_trace(platform, proposer, trace_id)
    assert trace_quarantine_state(platform.store, trace_id) == "proposed"
    events = platform.store.list_audit(action="trace.quarantined", subject=trace_id)
    assert len(events) == 1


def test_clearing_takes_a_second_person(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    proposer, clearer = people(platform, enroll_person)
    trace_id = failed_trace(platform, proposer.id)
    quarantine_trace(platform, proposer, trace_id)

    with pytest.raises(PermissionError, match="different person"):
        clear_quarantined_trace(platform, proposer, trace_id)
    clear_quarantined_trace(platform, clearer, trace_id)
    assert trace_quarantine_state(platform.store, trace_id) == "cleared"


def test_a_conversion_needs_cleared_quarantine(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    proposer, clearer = people(platform, enroll_person)
    trace_id = failed_trace(platform, proposer.id)

    with pytest.raises(ValueError, match="quarantine"):
        triage_service.sft_example_from_trace(platform, clearer, trace_id, GOOD)
    with pytest.raises(ValueError, match="quarantine"):
        triage_service.eval_case_from_trace(platform, clearer, trace_id)
    with pytest.raises(ValueError, match="quarantine"):
        triage_service.preference_from_trace(platform, clearer, trace_id, GOOD)

    quarantine_trace(platform, proposer, trace_id)
    with pytest.raises(ValueError, match="quarantine"):
        triage_service.sft_example_from_trace(platform, clearer, trace_id, GOOD)

    clear_quarantined_trace(platform, clearer, trace_id)
    dataset = triage_service.sft_example_from_trace(platform, clearer, trace_id, GOOD)
    assert dataset.name == "trace-sft"


def test_a_trace_source_that_skipped_quarantine_is_not_supervision(  # type: ignore[no-untyped-def]
    platform, enroll_person
) -> None:
    proposer, clearer = people(platform, enroll_person)
    trace_id = failed_trace(platform, proposer.id)
    row = {
        "instruction": QUESTION,
        "response": GOOD,
        "authored_by": clearer.id,
        "source_id": trace_id,
    }
    with pytest.raises(ValueError, match="quarantine"):
        require_supervision_row(platform, row)

    quarantine_trace(platform, proposer, trace_id)
    clear_quarantined_trace(platform, clearer, trace_id)
    require_supervision_row(platform, row)


def test_an_unknown_trace_source_is_not_supervision(  # type: ignore[no-untyped-def]
    platform, enroll_person
) -> None:
    _proposer, clearer = people(platform, enroll_person)
    row = {
        "instruction": QUESTION,
        "response": GOOD,
        "authored_by": clearer.id,
        "source_id": "trc_missing",
    }
    with pytest.raises(ValueError, match="quarantine"):
        require_supervision_row(platform, row)


def test_a_passing_answer_is_not_quarantine_material(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    proposer, _clearer = people(platform, enroll_person)
    trace = record_trace(
        platform.store,
        platform.audit,
        Principal(user_id=proposer.id, groups=ENGINEERING),
        endpoint="/v1/answer",
        question=QUESTION,
        answer=GOOD,
        citations=[{"path": "docs/ledger.md", "line": 12, "text": "replay"}],
    )
    with pytest.raises(ValueError, match="only a failed answer"):
        quarantine_trace(platform, proposer, trace.id)
