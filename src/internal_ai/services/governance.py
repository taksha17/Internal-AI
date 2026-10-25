"""Governance actions: plans, gates, budgets, kill switch, policy, people."""

from __future__ import annotations

from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import BudgetScope, GateRequest, GateType, Task, TaskState
from internal_ai.gates import GATE_CATALOG, GateError
from internal_ai.planner import LLMPlanner, TemplatePlanner
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, deny, require

KILL_SWITCH_ENGAGED_BY = "kill_switch_engaged_by"
KILL_SWITCH_REASON_KEY = "kill_switch_reason"


def _note_required(actor: Actor, note: str, what: str) -> None:
    if actor.authenticated and not note.strip():
        raise ValueError(f"{what} needs a typed reason")


def submit_plan(
    platform: Platform,
    actor: Actor,
    goal: str,
    milestone: str | None = None,
    planner: str = "template",
    approve: bool = False,
) -> list[Task]:
    require(platform.audit, actor, Action.SUBMIT_PLAN, "plan")
    if approve:
        require(platform.audit, actor, Action.APPROVE_PLAN, "plan")
    chosen = (
        TemplatePlanner()
        if planner == "template"
        else LLMPlanner(llm=platform.runtime.llm, fallback=TemplatePlanner())
    )
    drafts = chosen.plan(goal, {"milestone": milestone})
    return platform.orchestrator.submit_plan(drafts, approved_by=actor.id if approve else None)


def approve_plan(platform: Platform, actor: Actor) -> int:
    require(platform.audit, actor, Action.APPROVE_PLAN, "plan")
    return platform.orchestrator.approve_plan(actor.id)


def run_plan(
    platform: Platform, actor: Actor, max_iterations: int = 50, approve: bool = False
) -> dict[str, Any]:
    require(platform.audit, actor, Action.RUN_PLAN, "orchestrator")
    if approve:
        approve_plan(platform, actor)
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="orchestrator.run_requested",
        subject="orchestrator",
        details={"max_iterations": max_iterations},
    )
    summary = platform.orchestrator.run(max_iterations=max_iterations)
    summary.pop("outcomes", None)
    return summary


def transition_task(
    platform: Platform, actor: Actor, task_id: str, state: str, note: str = "manual transition"
) -> Task:
    require(platform.audit, actor, Action.TRANSITION_TASK, task_id)
    task = platform.store.get_task(task_id)
    if task is None:
        raise KeyError(f"unknown task: {task_id}")
    target = TaskState(state)
    return platform.store.set_task_state(task, target, actor.id, actor.kind, note or "manual")


def request_gate(
    platform: Platform,
    actor: Actor,
    gate_type: GateType | str,
    title: str,
    subject: str,
    evidence: dict[str, Any] | None = None,
) -> GateRequest:
    require(platform.audit, actor, Action.REQUEST_GATE, subject)
    resolved = GateType(gate_type)
    entry = GATE_CATALOG[resolved]
    supplied = dict(evidence or {})
    missing = [key for key in entry.evidence if not str(supplied.get(key, "")).strip()]
    if missing:
        raise ValueError(f"missing required evidence: {', '.join(missing)}")
    gate = platform.gates.request(
        resolved,
        title=title,
        subject=subject,
        requested_by=actor.id,
        requested_by_kind=actor.kind,
        evidence=supplied,
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="gate.requested",
        subject=gate.id,
        details={"gate_type": str(resolved), "subject": subject},
    )
    return gate


def can_decide(platform: Platform, actor: Actor, gate: GateRequest) -> tuple[bool, str]:
    """Whether this person may decide this gate, and why not when they may not."""
    if actor.kind != "human":
        return False, "gates may only be decided by humans"
    if actor.id == gate.requested_by:
        return False, "you requested this gate, so someone else has to decide it"
    if not actor.holds(*gate.required_approvers):
        return False, "needs " + " or ".join(gate.required_approvers)
    decided = {item.approver for item in platform.store.gate_decisions(gate.id)}
    if actor.id in decided:
        return False, "you have already decided this gate"
    return True, ""


def decide_gate(
    platform: Platform, actor: Actor, gate_id: str, approved: bool, note: str = ""
) -> GateRequest:
    gate = platform.store.get_gate(gate_id)
    if gate is None:
        raise GateError(f"unknown gate: {gate_id}")
    if actor.id == gate.requested_by:
        deny(platform.audit, actor, "decide_gate", gate_id, "requester cannot decide own gate")
        raise GateError(
            f"dual control violation: {actor.id} requested gate {gate_id} and cannot approve it"
        )
    if actor.authenticated:
        ok, why = can_decide(platform, actor, gate)
        if not ok:
            raise deny(platform.audit, actor, "decide_gate", gate_id, why)
    _note_required(actor, note, "a gate decision")
    decided = platform.gates.decide(
        gate_id, actor.id, approved=approved, approver_kind=actor.kind, note=note
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="gate.approved" if approved else "gate.rejected",
        subject=decided.id,
        details={"gate_type": str(decided.gate_type), "note": note, "status": str(decided.status)},
    )
    platform.orchestrator.release_unblocked()
    return decided


def set_budget_limit(
    platform: Platform,
    actor: Actor,
    scope: BudgetScope | str,
    key: str,
    limit_usd: float,
    gate_id: str | None = None,
) -> dict[str, Any]:
    """Set a cap. Lowering or adding one is immediate; raising one needs a gate."""
    resolved = BudgetScope(scope)
    subject = f"{resolved}:{key}"
    require(platform.audit, actor, Action.SET_BUDGET, subject)
    if limit_usd < 0:
        raise ValueError("a budget limit cannot be negative")
    current = platform.budget.status(resolved, key)
    if current is not None and limit_usd > current.limit_usd:
        gate = platform.store.get_gate(gate_id) if gate_id else None
        if (
            gate is None
            or str(gate.gate_type) != str(GateType.BUDGET_INCREASE)
            or gate.subject != subject
            or not platform.gates.is_approved(gate.id)
        ):
            raise deny(
                platform.audit,
                actor,
                "set_budget",
                subject,
                f"raising {subject} from {current.limit_usd:.2f} to {limit_usd:.2f} USD needs an "
                f"approved budget_increase gate with subject {subject}",
            )
    platform.budget.set_limit(resolved, key, limit_usd)
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="budget.limit_set",
        subject=subject,
        details={
            "limit_usd": limit_usd,
            "previous_usd": current.limit_usd if current else None,
            "gate_id": gate_id,
        },
    )
    return {"scope": str(resolved), "key": key, "limit_usd": limit_usd}


def kill_switch_state(platform: Platform) -> dict[str, Any]:
    return {
        "engaged": platform.budget.kill_switch_engaged(),
        "engaged_by": platform.store.kv_get(KILL_SWITCH_ENGAGED_BY, "") or None,
        "reason": platform.store.kv_get(KILL_SWITCH_REASON_KEY, "") or None,
    }


def engage_kill_switch(platform: Platform, actor: Actor, reason: str = "") -> dict[str, Any]:
    require(platform.audit, actor, Action.ENGAGE_KILL_SWITCH, "global")
    _note_required(actor, reason, "engaging the kill switch")
    platform.budget.engage_kill_switch()
    platform.store.kv_set(KILL_SWITCH_ENGAGED_BY, actor.id if actor.authenticated else "")
    platform.store.kv_set(KILL_SWITCH_REASON_KEY, reason)
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="budget.kill_switch_engaged",
        subject="global",
        details={"reason": reason},
    )
    return kill_switch_state(platform)


def release_kill_switch(platform: Platform, actor: Actor, reason: str = "") -> dict[str, Any]:
    """Release needs a second person when the switch was engaged by a signed-in one."""
    require(platform.audit, actor, Action.RELEASE_KILL_SWITCH, "global")
    _note_required(actor, reason, "releasing the kill switch")
    engaged_by = platform.store.kv_get(KILL_SWITCH_ENGAGED_BY, "") or ""
    if engaged_by and engaged_by == actor.id:
        raise deny(
            platform.audit,
            actor,
            "release_kill_switch",
            "global",
            "the person who engaged the kill switch cannot release it; ask a second person",
        )
    platform.budget.release_kill_switch()
    platform.store.kv_set(KILL_SWITCH_ENGAGED_BY, "")
    platform.store.kv_set(KILL_SWITCH_REASON_KEY, "")
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="budget.kill_switch_released",
        subject="global",
        details={"reason": reason, "engaged_by": engaged_by or None},
    )
    platform.orchestrator.release_unblocked()
    return kill_switch_state(platform)


def propose_policy(
    platform: Platform, actor: Actor, policy_id: str, value: str, rationale: str = ""
) -> tuple[Any, str]:
    require(platform.audit, actor, Action.DECIDE_POLICY, policy_id)
    decision, gate_id = platform.policy.propose(
        policy_id, value, requested_by=actor.id, rationale=rationale
    )
    platform.store.kv_set(f"policy_gate:{policy_id}", gate_id)
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="policy.proposed",
        subject=policy_id,
        details={"value": value, "gate_id": gate_id},
    )
    return decision, gate_id


def confirm_policy(
    platform: Platform, actor: Actor, policy_id: str, gate_id: str | None = None
) -> Any:
    require(platform.audit, actor, Action.DECIDE_POLICY, policy_id)
    decision = platform.policy.confirm(policy_id, decided_by=actor.id, gate_id=gate_id)
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="policy.decided",
        subject=policy_id,
        details={"value": decision.current_value},
    )
    return decision


def acknowledge_alert(platform: Platform, actor: Actor, alert_id: str) -> Any:
    from internal_ai.alerts import AlertEngine

    require(platform.audit, actor, Action.ACKNOWLEDGE_ALERT, alert_id)
    engine = AlertEngine(platform.store, platform.audit, platform.budget)
    known = {alert.id for alert in engine.list_alerts()}
    if alert_id not in known:
        raise KeyError(f"unknown alert: {alert_id}")
    return engine.acknowledge(alert_id, actor.id)


def replace_person(
    platform: Platform,
    actor: Actor,
    person_id: str,
    name: str,
    title: str,
    roles: list[str],
    groups: list[str] | None = None,
) -> dict[str, Any]:
    from internal_ai.people import PeopleRegistry

    require(platform.audit, actor, Action.MANAGE_PEOPLE, person_id)
    registry = PeopleRegistry(platform.store)
    if registry.get(person_id) is None:
        raise KeyError(f"unknown person: {person_id}")
    record = registry.replace(person_id, name, title, roles)
    if groups is not None:
        registry.set_groups(person_id, groups)
        record["groups"] = groups
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="people.replaced",
        subject=person_id,
        details={"name": name, "roles": ",".join(roles), "groups": groups},
    )
    return record


def add_person(
    platform: Platform,
    actor: Actor,
    name: str,
    title: str,
    roles: list[str],
    groups: list[str] | None = None,
) -> dict[str, Any]:
    from internal_ai.people import PeopleRegistry

    require(platform.audit, actor, Action.MANAGE_PEOPLE, name)
    registry = PeopleRegistry(platform.store)
    if registry.get(name) is not None:
        raise ValueError(f"a person named {name} already exists")
    record = registry.new_person(name, title, roles, groups)
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="people.added",
        subject=record["id"],
        details={"name": name, "roles": ",".join(roles), "groups": groups or []},
    )
    return record
