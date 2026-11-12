"""An approved web-access gate can be used from the console.

Requesting and approving already work through the generic gate pages; this
covers the missing half, exercising the gate: the fetch page lists approved
gates with their allowlists, a signed-in person fetches an allowlisted URL
through one, and anything else is refused without touching the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.domain import GateStatus, GateType

EVIDENCE = {
    "url_allowlist": ["docs.example.com"],
    "purpose": "read the vendor SDK changelog before pinning a dependency",
    "redaction_plan": "store the changelog text, never credentials or cookies",
}


@contextmanager
def console(tmp_path: Path) -> Iterator[tuple[TestClient, Any, Any, Any]]:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        platform = app.state.platform
        requester = enroll(platform, "Rae Requester", ["ml_lead"])
        programme = enroll(platform, "Pat Programme", ["program_lead"])
        security = enroll(platform, "Sam Security", ["security"])
        yield client, platform, requester, (programme, security)


def approved_gate(platform, requester, programme, security):  # type: ignore[no-untyped-def]
    gate = platform.gates.request(
        GateType.WEB_ACCESS,
        "Fetch vendor SDK docs",
        "vendor-sdk-docs",
        requester.id,
        requested_by_kind="human",
        evidence=dict(EVIDENCE),
    )
    platform.gates.decide(gate.id, programme.id, approved=True, note="risk accepted")
    platform.gates.decide(gate.id, security.id, approved=True, note="boundary reviewed")
    gate = platform.store.get_gate(gate.id)
    assert gate.status == GateStatus.APPROVED
    return gate


def test_the_fetch_page_lists_approved_gates_with_their_allowlists(
    tmp_path: Path,
) -> None:
    with console(tmp_path) as (client, platform, requester, approvers):
        programme, security = approvers
        gate = approved_gate(platform, requester, programme, security)
        assert sign_in(client, requester.name).status_code == 303

        page = client.get("/console/web")
        assert page.status_code == 200
        assert gate.id in page.text
        assert "docs.example.com" in page.text
        assert "Fetch vendor SDK docs" in page.text


def test_a_signed_out_visitor_is_sent_to_sign_in(tmp_path: Path) -> None:
    with console(tmp_path) as (client, _platform, _requester, _approvers):
        response = client.get("/console/web", follow_redirects=False)
        assert response.status_code == 303


def test_fetching_through_an_approved_gate_shows_the_bytes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"changelog text")
    with console(tmp_path) as (client, platform, requester, approvers):
        programme, security = approvers
        gate = approved_gate(platform, requester, programme, security)
        assert sign_in(client, requester.name).status_code == 303

        page = client.post(
            "/console/web/fetch",
            data={
                "csrf_token": session_csrf(client),
                "gate_id": gate.id,
                "url": "https://docs.example.com/changelog",
            },
        )
        assert page.status_code == 200
        assert "changelog text" in page.text
        assert "docs.example.com" in page.text
        events = platform.store.list_audit(action="service.web_fetch", subject=gate.id)
        assert events, "a console fetch must leave a fetch log"


def test_fetching_off_the_allowlist_is_refused_without_a_fetch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setattr(
        "internal_ai.tools._fetch_url_bytes",
        lambda url: (_ for _ in ()).throw(AssertionError("must never be fetched")),
    )
    with console(tmp_path) as (client, platform, requester, approvers):
        programme, security = approvers
        gate = approved_gate(platform, requester, programme, security)
        assert sign_in(client, requester.name).status_code == 303

        page = client.post(
            "/console/web/fetch",
            data={
                "csrf_token": session_csrf(client),
                "gate_id": gate.id,
                "url": "https://evil.example.net/",
            },
        )
        assert page.status_code == 200
        assert "allowlist" in page.text


def test_fetching_under_a_pending_gate_is_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setattr(
        "internal_ai.tools._fetch_url_bytes",
        lambda url: (_ for _ in ()).throw(AssertionError("must never be fetched")),
    )
    with console(tmp_path) as (client, platform, requester, _approvers):
        gate = platform.gates.request(
            GateType.WEB_ACCESS,
            "Fetch vendor SDK docs",
            "vendor-sdk-docs",
            requester.id,
            requested_by_kind="human",
            evidence=dict(EVIDENCE),
        )
        assert sign_in(client, requester.name).status_code == 303

        page = client.post(
            "/console/web/fetch",
            data={
                "csrf_token": session_csrf(client),
                "gate_id": gate.id,
                "url": "https://docs.example.com/",
            },
        )
        assert page.status_code == 200
        assert "gate" in page.text
