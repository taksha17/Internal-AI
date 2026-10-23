"""Task lifecycle, gates, audit chain and budget enforcement."""

from __future__ import annotations

import sqlite3

import pytest

from internal_ai.domain import AuditOutcome, BudgetScope, GateType, Task, TaskKind, TaskState
from internal_ai.gates import GateError
from internal_ai.store import TransitionError


def make_task(**kwargs) -> Task:
    defaults = {
        "id": "tsk_1",
        "title": "Build corpus",
        "description": "",
        "kind": TaskKind.DATA,
        "owner_role": "data_engineer",
    }
    defaults.update(kwargs)
    return Task(**defaults)


def test_task_state_transitions_are_enforced(forge) -> None:
    task = make_task()
    forge.store.add_task(task)
    forge.store.set_task_state(task, TaskState.APPROVED, "program_lead", "human")
    assert task.state == TaskState.APPROVED
    with pytest.raises(TransitionError):
        forge.store.set_task_state(task, TaskState.ACCEPTED, "program_lead", "human")


def test_terminal_states_cannot_move(forge) -> None:
    task = make_task()
    forge.store.add_task(task)
    forge.store.set_task_state(task, TaskState.APPROVED, "program_lead", "human")
    forge.store.set_task_state(task, TaskState.IN_PROGRESS, "data_engineer", "agent")
    forge.store.set_task_state(task, TaskState.IN_REVIEW, "data_engineer", "agent")
    forge.store.set_task_state(task, TaskState.ACCEPTED, "evaluation", "agent")
    with pytest.raises(TransitionError):
        forge.store.set_task_state(task, TaskState.IN_PROGRESS, "evaluation", "agent")


def test_task_events_record_the_audit_trail(forge) -> None:
    task = make_task()
    forge.store.add_task(task)
    forge.store.set_task_state(task, TaskState.APPROVED, "program_lead", "human")
    events = forge.store.task_events(task.id)
    assert [event.to_state for event in events] == ["proposed", "approved"]
    assert events[-1].actor == "program_lead"


def test_gate_requires_all_named_approvers(forge) -> None:
    gate = forge.gates.request(
        GateType.MODEL_PROMOTION,
        title="Promote model",
        subject="mdl_1",
        requested_by="evaluation",
        evidence={
            "eval_results_vs_baseline": "report",
            "model_card": "card",
            "red_team_results": "none",
        },
    )
    assert forge.gates.is_approved(gate.id) is False
    forge.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="human")
    assert forge.gates.is_approved(gate.id) is False
    forge.gates.decide(gate.id, "domain_expert", approved=True, approver_kind="human")
    assert forge.gates.is_approved(gate.id) is True


def test_gate_blocks_dual_control_violation(forge) -> None:
    gate = forge.gates.request(
        GateType.DATASET_RELEASE,
        title="Release dataset",
        subject="ds_1",
        requested_by="ml_lead",
        evidence={"quality_report": "q", "pii_audit_sample": "p", "lineage": "l"},
    )
    with pytest.raises(GateError, match="dual control"):
        forge.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="human")


def test_gate_rejects_agents_and_missing_evidence(forge) -> None:
    gate = forge.gates.request(
        GateType.EVAL_SET_FREEZE,
        title="Freeze eval",
        subject="evs_1",
        requested_by="data_scientist",
        evidence={"coverage_analysis": "c"},
    )
    with pytest.raises(GateError, match="only be decided by humans"):
        forge.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="agent")
    with pytest.raises(GateError, match="missing required evidence"):
        forge.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="human")


def test_audit_chain_verifies_and_rejects_tampering(forge) -> None:
    for index in range(5):
        forge.audit.record(
            actor="orchestrator",
            action=f"action.{index}",
            subject="subject",
            outcome=AuditOutcome.SUCCESS,
        )
    verification = forge.audit.verify()
    assert verification.ok
    assert verification.checked == 5
    with pytest.raises(sqlite3.IntegrityError):
        forge.store.conn.execute("UPDATE audit_events SET action = 'tampered' WHERE seq = 3")
    assert forge.audit.verify().ok
    assert forge.audit.recent(limit=1)[0].action == "action.4"


def test_audit_log_rejects_deletes(forge) -> None:
    forge.audit.record(actor="a", action="x", subject="y")
    with pytest.raises(sqlite3.IntegrityError):
        forge.store.conn.execute("DELETE FROM audit_events")


def test_budget_caps_apply_across_scopes(forge) -> None:
    forge.budget.set_limit(BudgetScope.GLOBAL, "global", 1.0)
    forge.budget.record_spend(0.6, agent="data_engineer")
    decision = forge.budget.check("data_engineer")
    assert decision.allowed
    forge.budget.record_spend(0.6, agent="data_engineer")
    decision = forge.budget.check("data_engineer")
    assert not decision.allowed
    assert "global" in decision.reason


def test_kill_switch_stops_dispatch(forge) -> None:
    forge.budget.engage_kill_switch()
    decision = forge.budget.check("data_engineer")
    assert not decision.allowed
    assert "kill switch" in decision.reason
    forge.budget.release_kill_switch()
    assert forge.budget.check("data_engineer").allowed


def test_people_replace_swaps_placeholder_for_real_human(forge) -> None:
    from internal_ai.people import PeopleRegistry

    registry = PeopleRegistry(forge.store)
    record = registry.replace("ppl_marcus_delgado", "Taksha", "ML lead", ["ml_lead"])
    assert record["placeholder"] is False
    assert record["roles"] == ["ml_lead"]
    people = {person["id"]: person for person in registry.list_people()}
    ml_lead = people["ppl_marcus_delgado"]
    assert ml_lead["name"] == "Taksha"
    assert ml_lead["placeholder"] is False
    assert ml_lead["roles"] == ["ml_lead"]
    untouched = people["ppl_priya_raghunathan"]
    assert untouched["placeholder"] is True
