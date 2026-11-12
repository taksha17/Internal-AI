"""The console's harness page runs the layer serving uses, and shows its record.

The page is a third client of the same harness: it asks with this install's own
engine, passes the evidence a person typed, and leaves the digest, latency, cost
and pass or fail in the audit log. The denials matter as much as the answer — a
kill switch stops the call before it is paid for, and a citation with no text
cannot be checked, so it is refused by line rather than guessed at.
"""

from __future__ import annotations

from pathlib import Path

from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.domain import BudgetScope

QUESTION = "where is ticket 42?"
EVIDENCE = "notes/tickets.txt:3 | Ticket 42 was closed on Monday after the reset."


def test_the_console_answers_through_the_harness_and_shows_the_record(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        person = enroll(platform, "Ada Analyst", ["ml_lead"])
        assert sign_in(client, person.name).status_code == 303

        page = client.get("/console/harness")
        assert page.status_code == 200
        assert "Ask the harness" in page.text
        assert 'href="/console/harness"' in page.text

        posted = client.post(
            "/console/harness",
            data={"csrf_token": csrf_from(page.text), "question": QUESTION, "evidence": EVIDENCE},
        )
        assert posted.status_code == 200
        assert "<h2>Answer</h2>" in posted.text
        assert "Every citation names a line you supplied" in posted.text

        events = platform.store.list_audit(action="harness.answer", actor=person.id)
        assert len(events) == 1
        digest = events[0].details["prompt_sha256"]
        assert digest in posted.text
        assert events[0].details["window_tokens"] == 8192
        assert "Recent harness answers" in posted.text


def test_an_engaged_kill_switch_stops_the_console_before_it_spends(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        person = enroll(platform, "Ada Analyst", ["ml_lead"])
        platform.budget.engage_kill_switch()
        assert sign_in(client, person.name).status_code == 303

        page = client.get("/console/harness")
        posted = client.post(
            "/console/harness",
            data={"csrf_token": csrf_from(page.text), "question": QUESTION, "evidence": EVIDENCE},
        )
        assert posted.status_code == 200
        assert "flash-error" in posted.text
        assert "<h2>Answer</h2>" not in posted.text

        events = platform.store.list_audit(action="harness.answer", actor=person.id)
        assert [str(event.outcome) for event in events] == ["denied"]
        assert platform.store.spend_by(BudgetScope.AGENT, person.id) == 0.0


def test_evidence_without_its_text_is_refused_by_its_line_number(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        person = enroll(platform, "Ada Analyst", ["ml_lead"])
        assert sign_in(client, person.name).status_code == 303

        page = client.get("/console/harness")
        posted = client.post(
            "/console/harness",
            data={
                "csrf_token": csrf_from(page.text),
                "question": QUESTION,
                "evidence": "notes/tickets.txt:3",
            },
        )
        assert "evidence line 1" in posted.text
        assert not platform.store.list_audit(action="harness.answer")

        empty = client.post(
            "/console/harness",
            data={"csrf_token": csrf_from(posted.text), "question": QUESTION, "evidence": ""},
        )
        assert "at least one evidence line" in empty.text
        assert not platform.store.list_audit(action="harness.answer")
