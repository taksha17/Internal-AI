"""The Inbox is the one place open work is counted.

A gate belongs to whoever may decide it, and a gate this person asked for is
never theirs to decide. The rail badge, the page and the sections are read from
one collector, so the number on the rail is the number of things the page lists.
An older failure is not counted, because nobody can ever clear it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.alerts import AlertEngine
from internal_ai.api import create_app
from internal_ai.context import Platform
from internal_ai.domain import GateType
from internal_ai.services import governance
from internal_ai.services.actor import Actor


def release_gate(platform: Platform, requester: Actor) -> Any:
    return governance.request_gate(
        platform,
        requester,
        GateType.DATASET_RELEASE,
        title="Release support-tickets v1",
        subject="ds_support_tickets",
        evidence={
            "quality_report": "labels reviewed",
            "pii_audit_sample": "no PII in the sample",
            "lineage": "built from approved source src_tickets",
        },
    )


def alert(platform: Platform) -> Any:
    return AlertEngine(platform.store, platform.audit, platform.budget).raise_alert(
        "spend_approaching_cap",
        "medium",
        "weekly:global",
        "spend is 90% of the weekly cap",
    )


@contextmanager
def console(tmp_path: Path) -> Iterator[tuple[TestClient, Any, Actor, Actor]]:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        platform = app.state.platform
        lead = enroll(platform, "Omar Ops", ["ml_lead"])
        requester = enroll(platform, "Lena Lead", ["program_lead"])
        yield client, app, lead, requester


def sections(page: str) -> tuple[str, str]:
    """The page split where the second half starts, to see which card holds a row."""
    return page.split("Blocked on someone else", 1)


def test_a_gate_is_theirs_to_decide_or_someone_elses(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, lead, requester):
        gate = release_gate(app.state.platform, requester)

        assert sign_in(client, lead.name).status_code == 303
        page = client.get("/console/inbox").text
        mine, blocked = sections(page)
        assert gate.title in mine
        assert "No gate needs you right now" not in page
        assert f'href="/console/approvals#gate-{gate.id}"' in mine

        client.cookies.clear()
        assert sign_in(client, requester.name).status_code == 303
        page = client.get("/console/inbox").text
        mine, blocked = sections(page)
        assert gate.title not in mine
        assert gate.title in blocked
        assert "you asked for this one" in blocked
        assert "ml_lead or compliance" in blocked


def test_a_person_without_the_role_is_offered_no_decision(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, _lead, requester):
        gate = release_gate(app.state.platform, requester)
        viewer = enroll(app.state.platform, "Vic Viewer", ["security_compliance"])
        assert sign_in(client, viewer.name).status_code == 303

        page = client.get("/console/inbox").text
        mine, blocked = sections(page)
        assert gate.title in blocked
        assert gate.title not in mine
        assert f'action="/console/approvals/{gate.id}"' not in page
        assert "No gate needs you right now" in page


def test_the_rail_badge_counts_the_whole_inbox(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, lead, requester):
        release_gate(app.state.platform, requester)
        alert(app.state.platform)
        app.state.jobs.submit(
            actor=lead,
            kind="build_dataset",
            title="build support-tickets",
            params={},
            work=lambda: (_ for _ in ()).throw(RuntimeError("disk full")),
            agent="data_engineer",
        )
        assert sign_in(client, lead.name).status_code == 303

        page = client.get("/console/home").text
        assert '<a href="/console/inbox"' in page
        assert '<span class="count">3</span>' in page
        assert page.count('class="count"') == 1
        assert "Open the inbox" in page

        inbox = client.get("/console/inbox").text
        assert "1 waiting" in inbox
        assert "1 open" in inbox
        assert "1 failed" in inbox


def test_acknowledging_from_the_inbox_comes_back_to_the_inbox(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, lead, _requester):
        item = alert(app.state.platform)
        assert sign_in(client, lead.name).status_code == 303
        page = client.get("/console/inbox")
        assert item.message in page.text

        posted = client.post(
            f"/console/alerts/{item.id}/acknowledge",
            data={"csrf_token": csrf_from(page.text), "back": "/console/inbox"},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        assert posted.headers["location"] == "/console/inbox"

        after = client.get("/console/inbox").text
        assert item.message not in after
        assert "Nothing is waiting" in after
        assert '<span class="count">' not in after


def test_a_failure_older_than_a_week_leaves_the_inbox(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, lead, _requester):
        app.state.jobs.submit(
            actor=lead,
            kind="train_model",
            title="train tickets-embed",
            params={},
            work=lambda: (_ for _ in ()).throw(RuntimeError("disk full")),
            agent="ml_lead",
        )
        old = datetime.now(UTC) - timedelta(days=30)
        app.state.platform.store.conn.execute(
            "INSERT INTO console_jobs (id, kind, title, requested_by, state, params, error,"
            " created_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "cjob_old",
                "build_dataset",
                "build the old corpus",
                lead.id,
                "failed",
                "{}",
                "KeyError: source",
                old.isoformat(timespec="seconds"),
                old.isoformat(timespec="seconds"),
            ),
        )
        assert sign_in(client, lead.name).status_code == 303

        page = client.get("/console/inbox").text
        assert "train tickets-embed" in page
        assert "RuntimeError: disk full" in page
        assert "build the old corpus" not in page
        assert "1 older failure stay on" in page
        assert 'href="/console/jobs?state=failed"' in page


def test_an_empty_inbox_says_so_once(tmp_path: Path) -> None:
    with console(tmp_path) as (client, _app, lead, _requester):
        assert sign_in(client, lead.name).status_code == 303
        page = client.get("/console/inbox").text
        assert "Nothing is waiting" in page
        assert 'href="/console/home"' in page
        assert "Needs your decision" not in page
