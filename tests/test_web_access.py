"""Web access is a human-decided exception, never an agent default.

The platform ships with no web path: no shipped role holds NET_EGRESS, and
the fetch tool refuses unless the install is online, a WEB_ACCESS gate is
approved, and the host is on that gate's allowlist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.domain import Capability, GateStatus, GateType, RoleSpec
from internal_ai.gates import GATE_CATALOG, GateError
from internal_ai.roles import all_roles
from internal_ai.tools import ToolDenied, ToolRegistry, approved_web_allowlist

ALLOWLIST = ["docs.example.com"]
EVIDENCE = {
    "url_allowlist": ALLOWLIST,
    "purpose": "read the vendor SDK changelog before pinning a dependency",
    "redaction_plan": "store only the changelog text, never credentials or cookies",
}


def net_role() -> RoleSpec:
    return RoleSpec(
        name="net_probe",
        mission="probe",
        sub_agents=(),
        inputs=(),
        outputs=(),
        capabilities=frozenset({Capability.NET_EGRESS}),
        gates=(),
        kpis=(),
    )


def wired_registry(platform, tmp_path: Path) -> ToolRegistry:  # type: ignore[no-untyped-def]
    registry = ToolRegistry(platform.audit, tmp_path / "work", tmp_path / "artifacts")
    registry.set_web_gate_check(lambda gate_id: approved_web_allowlist(platform.store, gate_id))
    return registry


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


def test_no_shipped_role_may_reach_the_network(platform, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    registry = wired_registry(platform, tmp_path)
    assert "web_fetch" in registry.names()
    for role in all_roles():
        assert not role.can(Capability.NET_EGRESS), role.name
        with pytest.raises(ToolDenied):
            registry.invoke(
                role,
                "web_fetch",
                {"url": "https://docs.example.com/", "gate_id": "gate_none"},
            )


def test_web_fetch_needs_an_approved_gate(platform, tmp_path: Path, enroll_person) -> None:  # type: ignore[no-untyped-def]
    registry = wired_registry(platform, tmp_path)
    with pytest.raises(ToolDenied, match="gate"):
        registry.invoke(
            net_role(),
            "web_fetch",
            {"url": "https://docs.example.com/", "gate_id": "gate_missing"},
        )


def test_web_fetch_needs_its_host_on_the_allowlist(  # type: ignore[no-untyped-def]
    platform, tmp_path: Path, enroll_person, monkeypatch
) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    requester = enroll_person(platform, "Rae Requester", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    security = enroll_person(platform, "Sam Security", ["security"])
    gate = approve_web_gate(platform, request_web_gate(platform, requester), programme, security)
    assert gate.status == GateStatus.APPROVED
    registry = wired_registry(platform, tmp_path)
    with pytest.raises(ToolDenied, match="allowlist"):
        registry.invoke(
            net_role(),
            "web_fetch",
            {"url": "https://evil.example.net/", "gate_id": gate.id},
        )


def test_web_fetch_stays_offline_by_default(  # type: ignore[no-untyped-def]
    platform, tmp_path: Path, enroll_person, monkeypatch
) -> None:
    requester = enroll_person(platform, "Rae Requester", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    security = enroll_person(platform, "Sam Security", ["security"])
    gate = approve_web_gate(platform, request_web_gate(platform, requester), programme, security)
    registry = wired_registry(platform, tmp_path)
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"must never be fetched")
    with pytest.raises(ToolDenied, match="offline"):
        registry.invoke(
            net_role(),
            "web_fetch",
            {"url": "https://docs.example.com/", "gate_id": gate.id},
        )


def test_approved_allowlisted_fetch_is_logged(  # type: ignore[no-untyped-def]
    platform, tmp_path: Path, enroll_person, monkeypatch
) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setattr("internal_ai.tools._fetch_url_bytes", lambda url: b"changelog text")
    requester = enroll_person(platform, "Rae Requester", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    security = enroll_person(platform, "Sam Security", ["security"])
    gate = approve_web_gate(platform, request_web_gate(platform, requester), programme, security)
    registry = wired_registry(platform, tmp_path)
    body = registry.invoke(
        net_role(),
        "web_fetch",
        {"url": "https://docs.example.com/changelog", "gate_id": gate.id},
    )
    assert body == b"changelog text"
    events = platform.store.list_audit(action="tool.web_fetch", subject="web_fetch")
    assert events, "an approved fetch must leave a fetch log"
    assert gate.id in str(events[0].inputs)


def test_web_access_gate_names_its_deciders() -> None:
    entry = GATE_CATALOG[GateType.WEB_ACCESS]
    assert str(GateType.WEB_ACCESS) == "web_access"
    assert entry.approvers == ("program_lead", "security")
    assert set(entry.evidence) == {"url_allowlist", "purpose", "redaction_plan"}


def test_an_agent_may_not_decide_web_access(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["ml_lead"])
    gate = request_web_gate(platform, requester)
    with pytest.raises(GateError, match="humans"):
        platform.gates.decide(gate.id, "some_agent", approved=True, approver_kind="agent")


def test_the_requester_may_not_approve_web_access(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    gate = request_web_gate(platform, requester)
    with pytest.raises(GateError, match="dual control"):
        platform.gates.decide(gate.id, requester.id, approved=True)
