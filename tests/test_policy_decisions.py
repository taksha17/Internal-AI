"""A proposed policy value does not apply until the gate that proposes it is decided."""

from __future__ import annotations

import pytest

from internal_ai.services import governance


def pending_gate_id(platform, policy_id: str) -> str:
    return platform.store.kv_get(f"policy_gate:{policy_id}", "") or ""


def approve_gate(platform, gate_id: str) -> None:
    """Close the gate the way an approved decision leaves it.

    The gate's own role checks are covered by the gate tests; this test is about
    what a *pending* gate does to the policy, so it only needs the status.
    """
    stored = platform.store.get_gate(gate_id)
    assert stored is not None
    stored.status = type(stored.status)("approved")
    platform.store.update_gate(stored)


def test_a_proposal_leaves_the_policy_unchanged(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pat Lead", ["program_lead"])
    before = platform.policy.effective("P2")
    decision, gate_id = platform.policy.propose(
        "P2", "banned", requested_by=actor.id, rationale="no external model calls"
    )
    assert decision.status == "pending_approval"
    assert decision.current_value == before
    assert platform.policy.effective("P2") == before
    assert pending_gate_id(platform, "P2") == gate_id


def test_confirming_an_unapproved_gate_changes_nothing(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pat Lead", ["program_lead"])
    before = platform.policy.effective("P2")
    platform.policy.propose("P2", "banned", requested_by=actor.id)
    try:
        platform.policy.confirm("P2", decided_by=actor.id)
    except ValueError as exc:
        assert "not approved" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("confirm accepted an undecided gate")
    assert platform.policy.effective("P2") == before


def test_an_approved_gate_applies_the_proposed_value(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pat Lead", ["program_lead"])
    _decision, gate_id = platform.policy.propose("P2", "banned", requested_by=actor.id)
    approve_gate(platform, gate_id)
    decided = platform.policy.confirm("P2", decided_by=actor.id)
    assert decided.current_value == "banned"
    assert decided.status == "decided"
    assert decided.decided_by == actor.id
    assert platform.policy.effective("P2") == "banned"


def test_a_second_proposal_cannot_apply_without_its_own_gate(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pat Lead", ["program_lead"])
    _first, first_gate = platform.policy.propose("P2", "banned", requested_by=actor.id)
    approve_gate(platform, first_gate)
    decided = platform.policy.confirm("P2", decided_by=actor.id)
    assert decided.current_value == "banned"
    _second, second_gate = platform.policy.propose("P2", "allowed", requested_by=actor.id)
    assert second_gate != first_gate
    with pytest.raises(ValueError, match="not approved"):
        platform.policy.confirm("P2", decided_by=actor.id)
    assert platform.policy.effective("P2") == "banned"


def test_confirming_without_a_proposal_refuses_and_invents_nothing(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pat Lead", ["program_lead"])
    before = platform.policy.effective("P3")
    try:
        platform.policy.confirm("P3", decided_by=actor.id)
    except ValueError as exc:
        assert "no gate" in str(exc) or "no pending proposal" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("confirm decided a policy nobody proposed")
    assert platform.policy.effective("P3") == before


def test_the_governed_path_still_proposes_and_decides(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Rita Legal", ["program_lead"])
    before = platform.policy.effective("P2")
    decision, gate_id = governance.propose_policy(platform, actor, "P2", "banned", "regulator")
    assert decision.status == "pending_approval"
    assert platform.policy.effective("P2") == before
    approve_gate(platform, gate_id)
    confirmed = governance.confirm_policy(platform, actor, "P2")
    assert confirmed.current_value == "banned"
    proposed = platform.store.conn.execute(
        "SELECT action FROM audit_events WHERE action = 'policy.proposed'"
    ).fetchall()
    decided = platform.store.conn.execute(
        "SELECT action FROM audit_events WHERE action = 'policy.decided'"
    ).fetchall()
    assert proposed and decided
