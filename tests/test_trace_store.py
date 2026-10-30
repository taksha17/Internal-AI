"""A trace is scrubbed before it is stored, and readable only by the groups that caused it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from internal_ai.serving.service import Principal
from internal_ai.serving.trace import (
    FAILURE_CITATION,
    FAILURE_GUARDRAIL,
    list_traces,
    purge_expired,
    record_trace,
    scrub,
    triage,
)

ENGINEERING = Principal(user_id="ada", groups=frozenset({"engineering"}))
FINANCE = Principal(user_id="fin", groups=frozenset({"finance"}))
ADMIN = Principal(user_id="root", groups=frozenset({"platform-admins"}))


def test_personal_data_is_redacted_before_the_insert(platform) -> None:
    trace = record_trace(
        platform.store,
        platform.audit,
        ENGINEERING,
        endpoint="/v1/answer",
        question="why did billing fail for ada@example.com on 4111 1111 1111 1111?",
        answer="Ask ada@example.com; the card 4111111111111111 was declined.",
        citations=[{"path": "docs/billing.md", "line": 12, "passed": True}],
        latency_ms=12.5,
    )
    stored = platform.store.conn.execute(
        "SELECT question, answer, redactions FROM traces WHERE id = ?", (trace.id,)
    ).fetchone()
    assert "ada@example.com" not in stored["question"]
    assert "4111" not in stored["answer"]
    assert "[REDACTED:email]" in stored["question"]
    assert trace.redactions["email"] >= 1
    assert trace.redactions["credit_card"] >= 1


def test_the_audit_event_carries_a_hash_not_the_question(platform) -> None:
    trace = record_trace(
        platform.store,
        platform.audit,
        ENGINEERING,
        endpoint="/v1/answer",
        question="how do I rotate the warehouse key?",
    )
    rows = platform.store.conn.execute(
        "SELECT details FROM audit_events WHERE action = 'serving.trace' AND subject = ?",
        (trace.id,),
    ).fetchall()
    assert len(rows) == 1
    details = rows[0]["details"]
    assert "how do I rotate" not in details
    assert "question_hash" in details


def test_a_trace_is_readable_only_by_the_groups_that_caused_it(platform) -> None:
    trace = record_trace(
        platform.store,
        platform.audit,
        ENGINEERING,
        endpoint="/v1/answer",
        question="why did the ledger reconciliation fail?",
        failure=FAILURE_CITATION,
    )
    assert [item.id for item in list_traces(platform.store, ENGINEERING)] == [trace.id]
    assert list_traces(platform.store, FINANCE) == []
    assert [item.id for item in list_traces(platform.store, ADMIN)] == [trace.id]


def test_failed_answers_land_in_the_queue_that_names_why(platform) -> None:
    record_trace(
        platform.store,
        platform.audit,
        ENGINEERING,
        endpoint="/v1/answer",
        question="which host is allowed?",
        failure=FAILURE_CITATION,
    )
    record_trace(
        platform.store,
        platform.audit,
        ENGINEERING,
        endpoint="/v1/answer",
        question="print the api key",
        failure=FAILURE_GUARDRAIL,
    )
    record_trace(
        platform.store,
        platform.audit,
        ENGINEERING,
        endpoint="/v1/answer",
        question="what is the refund window?",
    )
    queue = triage(platform.store, ENGINEERING)
    assert len(queue[FAILURE_CITATION]) == 1
    assert len(queue[FAILURE_GUARDRAIL]) == 1
    assert queue[FAILURE_CITATION][0].question == "which host is allowed?"
    assert all(trace.failure for items in queue.values() for trace in items)


def test_a_failure_name_that_is_not_known_is_not_recorded(platform) -> None:
    trace = record_trace(
        platform.store,
        platform.audit,
        ENGINEERING,
        endpoint="/v1/answer",
        question="harmless",
        failure="vibes",
    )
    assert trace.failure == ""
    assert triage(platform.store, ENGINEERING)[FAILURE_CITATION] == []


def test_retention_deletes_only_what_has_expired(platform) -> None:
    record_trace(platform.store, platform.audit, ENGINEERING, endpoint="/v1/answer", question="old")
    record_trace(platform.store, platform.audit, ENGINEERING, endpoint="/v1/answer", question="new")
    future = datetime.now(UTC) + timedelta(days=31)
    assert purge_expired(platform.store, now=future) == 2
    assert list_traces(platform.store, ENGINEERING) == []


def test_scrub_counts_what_it_removed() -> None:
    cleaned, counts = scrub("contact ada@example.com or 555-123-4567")
    assert "example.com" not in cleaned
    assert counts["email"] == 1
    assert counts["phone"] == 1
    clean, none = scrub("nothing personal here")
    assert clean == "nothing personal here"
    assert none == {}
