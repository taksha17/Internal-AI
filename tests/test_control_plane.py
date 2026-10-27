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


def test_task_state_transitions_are_enforced(platform) -> None:
    task = make_task()
    platform.store.add_task(task)
    platform.store.set_task_state(task, TaskState.APPROVED, "program_lead", "human")
    assert task.state == TaskState.APPROVED
    with pytest.raises(TransitionError):
        platform.store.set_task_state(task, TaskState.ACCEPTED, "program_lead", "human")


def test_terminal_states_cannot_move(platform) -> None:
    task = make_task()
    platform.store.add_task(task)
    platform.store.set_task_state(task, TaskState.APPROVED, "program_lead", "human")
    platform.store.set_task_state(task, TaskState.IN_PROGRESS, "data_engineer", "agent")
    platform.store.set_task_state(task, TaskState.IN_REVIEW, "data_engineer", "agent")
    platform.store.set_task_state(task, TaskState.ACCEPTED, "evaluation", "agent")
    with pytest.raises(TransitionError):
        platform.store.set_task_state(task, TaskState.IN_PROGRESS, "evaluation", "agent")


def test_task_events_record_the_audit_trail(platform) -> None:
    task = make_task()
    platform.store.add_task(task)
    platform.store.set_task_state(task, TaskState.APPROVED, "program_lead", "human")
    events = platform.store.task_events(task.id)
    assert [event.to_state for event in events] == ["proposed", "approved"]
    assert events[-1].actor == "program_lead"


def test_gate_requires_all_named_approvers(platform) -> None:
    gate = platform.gates.request(
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
    assert platform.gates.is_approved(gate.id) is False
    platform.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="human")
    assert platform.gates.is_approved(gate.id) is False
    platform.gates.decide(gate.id, "domain_expert", approved=True, approver_kind="human")
    assert platform.gates.is_approved(gate.id) is True


def test_gate_blocks_dual_control_violation(platform) -> None:
    gate = platform.gates.request(
        GateType.DATASET_RELEASE,
        title="Release dataset",
        subject="ds_1",
        requested_by="ml_lead",
        evidence={"quality_report": "q", "pii_audit_sample": "p", "lineage": "l"},
    )
    with pytest.raises(GateError, match="dual control"):
        platform.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="human")


def test_gate_rejects_agents_and_missing_evidence(platform) -> None:
    gate = platform.gates.request(
        GateType.EVAL_SET_FREEZE,
        title="Freeze eval",
        subject="evs_1",
        requested_by="data_scientist",
        evidence={"coverage_analysis": "c"},
    )
    with pytest.raises(GateError, match="only be decided by humans"):
        platform.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="agent")
    with pytest.raises(GateError, match="missing required evidence"):
        platform.gates.decide(gate.id, "ml_lead", approved=True, approver_kind="human")


def test_a_live_gate_blocks_a_second_request_but_a_rejection_reopens(platform) -> None:
    from internal_ai.services import build

    build._no_open_gate(platform, GateType.MODEL_PROMOTION, "mdl_9")
    gate = platform.gates.request(
        GateType.MODEL_PROMOTION,
        title="Promote model",
        subject="mdl_9",
        requested_by="evaluation",
        evidence={
            "eval_results_vs_baseline": "report",
            "model_card": "card",
            "red_team_results": "none",
        },
    )
    with pytest.raises(ValueError, match="already pending"):
        build._no_open_gate(platform, GateType.MODEL_PROMOTION, "mdl_9")
    platform.gates.decide(gate.id, "ml_lead", approved=False, approver_kind="human")
    build._no_open_gate(platform, GateType.MODEL_PROMOTION, "mdl_9")


def test_audit_chain_verifies_and_rejects_tampering(platform) -> None:
    for index in range(5):
        platform.audit.record(
            actor="orchestrator",
            action=f"action.{index}",
            subject="subject",
            outcome=AuditOutcome.SUCCESS,
        )
    verification = platform.audit.verify()
    assert verification.ok
    assert verification.checked == 5
    with pytest.raises(sqlite3.IntegrityError):
        platform.store.conn.execute("UPDATE audit_events SET action = 'tampered' WHERE seq = 3")
    assert platform.audit.verify().ok
    assert platform.audit.recent(limit=1)[0].action == "action.4"


def test_audit_log_rejects_deletes(platform) -> None:
    platform.audit.record(actor="a", action="x", subject="y")
    with pytest.raises(sqlite3.IntegrityError):
        platform.store.conn.execute("DELETE FROM audit_events")


def test_budget_caps_apply_across_scopes(platform) -> None:
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 1.0)
    platform.budget.record_spend(0.6, agent="data_engineer")
    decision = platform.budget.check("data_engineer")
    assert decision.allowed
    platform.budget.record_spend(0.6, agent="data_engineer")
    decision = platform.budget.check("data_engineer")
    assert not decision.allowed
    assert "global" in decision.reason


def test_kill_switch_stops_dispatch(platform) -> None:
    platform.budget.engage_kill_switch()
    decision = platform.budget.check("data_engineer")
    assert not decision.allowed
    assert "kill switch" in decision.reason
    platform.budget.release_kill_switch()
    assert platform.budget.check("data_engineer").allowed


def test_people_replace_swaps_placeholder_for_real_human(platform) -> None:
    from internal_ai.people import PeopleRegistry

    registry = PeopleRegistry(platform.store)
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
