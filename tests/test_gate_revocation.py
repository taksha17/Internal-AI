"""An approved gate can be withdrawn by a person, and by nothing else.

Approval is dual control: every required role, from distinct people. Withdrawal
only removes authority, so one qualified person is enough — but that person has
to be someone who could have decided the gate, or the one who asked for it.
"""

from __future__ import annotations

import pytest

from internal_ai.domain import GateStatus, GateType
from internal_ai.gates import GateError
from internal_ai.services import governance
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import PermissionDenied

EVIDENCE = {
    "source_description": "a local records index maintained by the data team",
    "sensitivity_class": "internal",
    "purpose": "answer questions about open tickets",
    "legal_status": "internal-only, owned by the company",
}


def approved_gate(platform, requester) -> str:  # type: ignore[no-untyped-def]
    gate = platform.gates.request(
        GateType.NEW_DATA_SOURCE,
        "Admit the records server",
        "records",
        requester.id,
        requested_by_kind="human",
        evidence=dict(EVIDENCE),
    )
    platform.gates.decide(gate.id, "data_owner", approved=True, note="reviewed")
    platform.gates.decide(gate.id, "compliance", approved=True, note="reviewed")
    assert platform.store.get_gate(gate.id).status == GateStatus.APPROVED
    return gate.id


def test_an_approver_can_withdraw_an_approval(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    owner = enroll_person(platform, "Dana Owner", ["data_owner"])
    gate_id = approved_gate(platform, requester)

    revoked = governance.revoke_gate(
        platform, owner, gate_id, "the index was found to hold payroll"
    )

    assert revoked.status == GateStatus.REVOKED
    assert platform.gates.is_approved(gate_id) is False
    assert "the index was found to hold payroll" in revoked.decision_note
    assert platform.store.list_audit(action="gate.revoked", subject=gate_id)


def test_the_requester_may_withdraw_their_own_gate(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    gate_id = approved_gate(platform, requester)

    revoked = governance.revoke_gate(platform, requester, gate_id, "the team stopped using it")

    assert revoked.status == GateStatus.REVOKED


def test_a_person_with_no_role_on_the_gate_cannot_withdraw(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    bystander = enroll_person(platform, "Bea Bystander", ["finance"])
    gate_id = approved_gate(platform, requester)

    with pytest.raises(PermissionDenied, match="data_owner"):
        governance.revoke_gate(platform, bystander, gate_id, "I do not like it")

    assert platform.store.get_gate(gate_id).status == GateStatus.APPROVED
    assert platform.store.list_audit(action="console.denied", subject=gate_id)


def test_an_agent_cannot_withdraw_even_with_the_role_name(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    gate_id = approved_gate(platform, requester)
    agent = Actor(
        id="agent:harness",
        name="harness",
        roles=frozenset({"data_owner", "compliance"}),
        groups=frozenset(),
        authenticated=True,
        kind="agent",
    )

    with pytest.raises(PermissionDenied, match="people"):
        governance.revoke_gate(platform, agent, gate_id, "the harness changed its mind")

    assert platform.store.get_gate(gate_id).status == GateStatus.APPROVED


def test_a_withdrawal_needs_a_typed_reason(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    owner = enroll_person(platform, "Dana Owner", ["data_owner"])
    gate_id = approved_gate(platform, requester)

    with pytest.raises(ValueError, match="typed reason"):
        governance.revoke_gate(platform, owner, gate_id, "  ")

    assert platform.store.get_gate(gate_id).status == GateStatus.APPROVED


def test_a_pending_gate_is_cancelled_not_revoked(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    gate = platform.gates.request(
        GateType.NEW_DATA_SOURCE,
        "Admit another server",
        "other",
        requester.id,
        requested_by_kind="human",
        evidence=dict(EVIDENCE),
    )

    with pytest.raises(GateError, match="cancel it"):
        governance.revoke_gate(platform, requester, gate.id, "never mind")

    assert platform.store.get_gate(gate.id).status == GateStatus.PENDING


def test_an_approval_cannot_be_withdrawn_twice(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    requester = enroll_person(platform, "Rae Requester", ["program_lead"])
    owner = enroll_person(platform, "Dana Owner", ["data_owner"])
    gate_id = approved_gate(platform, requester)
    governance.revoke_gate(platform, owner, gate_id, "withdrawn")

    with pytest.raises(GateError, match="only an approval can be revoked"):
        governance.revoke_gate(platform, owner, gate_id, "withdrawn again")
