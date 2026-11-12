"""People can use an approved web-access gate without an agent run.

The gate's dual-control approval is the authorization: the service enforces
the same refusals as the agent fetch path (offline, no approved gate, host
off the allowlist) and writes the same fetch log. A gate requested through
the console carries the allowlist as pasted text, so the reader accepts one
host per line as well as a list.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from internal_ai.domain import GateStatus, GateType
from internal_ai.services import webfetch
from internal_ai.services.permissions import ACTION_ROLES, Action, PermissionDenied
from internal_ai.tools import ToolDenied

ALLOWLIST = ["docs.example.com", "files.example.com"]
EVIDENCE = {
    "url_allowlist": ALLOWLIST,
    "purpose": "read the vendor SDK changelog before pinning a dependency",
    "redaction_plan": "store the changelog text, never credentials or cookies",
}


def request_web_gate(platform, requester):  # type: ignore[no-untyped-def]
    return platform.gates.request(
        GateType.WEB_ACCESS,
        "Fetch vendor SDK docs",
        "vendor-sdk-docs",
        requester.id,
        requested_by_kind="human",
        evidence=dict(EVIDENCE),
    )


def approve_web_gate(platform, gate, programme, security):  # type: ignore[no-untyped-def]
    platform.gates.decide(gate.id, programme.id, approved=True, note="risk accepted")
    platform.gates.decide(gate.id, security.id, approved=True, note="boundary reviewed")
    return platform.store.get_gate(gate.id)


def people(platform, enroll_person):  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    security = enroll_person(platform, "Sam Security", ["security"])
    return requester, programme, security


def approved_gate(platform, enroll_person):  # type: ignore[no-untyped-def]
    requester, programme, security = people(platform, enroll_person)
    gate = approve_web_gate(platform, request_web_gate(platform, requester), programme, security)
    assert gate.status == GateStatus.APPROVED
    return gate, requester


def test_fetching_is_a_named_permission_any_signed_in_person_holds() -> None:
    assert ACTION_ROLES[Action.USE_WEB_GATE] == frozenset()


def test_an_agent_actor_may_not_fetch(platform, enroll_person, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"must never be fetched")
    gate, requester = approved_gate(platform, enroll_person)
    agent = dataclasses.replace(requester, kind="agent")
    with pytest.raises(PermissionDenied):
        webfetch.fetch_url(platform, agent, gate.id, "https://docs.example.com/")
    denials = platform.store.list_audit(action="console.denied", subject=gate.id)
    assert denials, "a refused fetch must leave a denial in the log"


def test_an_unknown_gate_is_refused(platform, enroll_person, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"must never be fetched")
    requester, _programme, _security = people(platform, enroll_person)
    with pytest.raises(ToolDenied, match="gate"):
        webfetch.fetch_url(platform, requester, "gate_missing", "https://docs.example.com/")


def test_a_pending_gate_is_refused(platform, enroll_person, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"must never be fetched")
    requester, _programme, _security = people(platform, enroll_person)
    gate = request_web_gate(platform, requester)
    with pytest.raises(ToolDenied, match="gate"):
        webfetch.fetch_url(platform, requester, gate.id, "https://docs.example.com/")


def test_a_host_off_the_allowlist_is_refused_and_logged(
    platform, enroll_person, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"must never be fetched")
    gate, requester = approved_gate(platform, enroll_person)
    with pytest.raises(ToolDenied, match="allowlist"):
        webfetch.fetch_url(platform, requester, gate.id, "https://evil.example.net/")
    denials = platform.store.list_audit(action="service.web_fetch", subject=gate.id)
    assert any(event.outcome == "denied" for event in denials)


def test_fetch_stays_offline_by_default(platform, enroll_person, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gate, requester = approved_gate(platform, enroll_person)
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"must never be fetched")
    with pytest.raises(ToolDenied, match="offline"):
        webfetch.fetch_url(platform, requester, gate.id, "https://docs.example.com/")


def test_a_console_pasted_allowlist_with_one_host_per_line_works(
    platform, enroll_person, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"changelog text")
    requester, programme, security = people(platform, enroll_person)
    gate = platform.gates.request(
        GateType.WEB_ACCESS,
        "Fetch vendor SDK docs",
        "vendor-sdk-docs",
        requester.id,
        requested_by_kind="human",
        evidence={**EVIDENCE, "url_allowlist": "docs.example.com\nfiles.example.com"},
    )
    gate = approve_web_gate(platform, gate, programme, security)
    result = webfetch.fetch_url(platform, requester, gate.id, "https://files.example.com/notes")
    assert result.body == b"changelog text"


def test_an_approved_fetch_returns_bytes_and_logs_host_not_query(
    platform, enroll_person, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"changelog text")
    gate, requester = approved_gate(platform, enroll_person)
    result = webfetch.fetch_url(
        platform,
        requester,
        gate.id,
        "https://docs.example.com/changelog?session=secret",
    )
    assert result.body == b"changelog text"
    assert result.host == "docs.example.com"
    assert result.size_bytes == len(b"changelog text")
    assert result.sha256 == hashlib.sha256(b"changelog text").hexdigest()
    events = platform.store.list_audit(action="service.web_fetch", subject=gate.id)
    assert events, "an approved fetch must leave a fetch log"
    assert "secret" not in str(events[0].inputs)
    assert "docs.example.com" in str(events[0].inputs)
