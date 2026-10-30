"""Which human roles may take which action.

The map is the specification. An action mapped to `ANYONE` is open to every
signed-in person; every other action names the roles that may take it. Gate
decisions are checked against the roles the gate itself requires, so they are
not in the map.
"""

from __future__ import annotations

from enum import StrEnum

from internal_ai.audit import AuditLog
from internal_ai.domain import AuditOutcome
from internal_ai.services.actor import Actor


class Action(StrEnum):
    ENGAGE_KILL_SWITCH = "engage_kill_switch"
    RELEASE_KILL_SWITCH = "release_kill_switch"
    ACKNOWLEDGE_ALERT = "acknowledge_alert"
    REQUEST_GATE = "request_gate"
    ASK = "ask"
    SUBMIT_PLAN = "submit_plan"
    APPROVE_PLAN = "approve_plan"
    RUN_PLAN = "run_plan"
    TRANSITION_TASK = "transition_task"
    SET_BUDGET = "set_budget"
    REGISTER_SOURCE = "register_source"
    BUILD_DATASET = "build_dataset"
    RELEASE_DATASET = "release_dataset"
    WITHDRAW_SOURCE = "withdraw_source"
    CREATE_EVAL_SET = "create_eval_set"
    FREEZE_EVAL_SET = "freeze_eval_set"
    TRAIN_MODEL = "train_model"
    EVALUATE_MODEL = "evaluate_model"
    REGISTER_MODEL = "register_model"
    PROMOTE_MODEL = "promote_model"
    SET_SERVING_POLICY = "set_serving_policy"
    DECIDE_POLICY = "decide_policy"
    WRITE_BRIEF = "write_brief"
    WRITE_RECIPE = "write_recipe"
    MANAGE_PEOPLE = "manage_people"
    VIEW_TRACES = "view_traces"


ANYONE: frozenset[str] = frozenset()

ACTION_ROLES: dict[Action, frozenset[str]] = {
    Action.ENGAGE_KILL_SWITCH: ANYONE,
    Action.RELEASE_KILL_SWITCH: ANYONE,
    Action.ACKNOWLEDGE_ALERT: ANYONE,
    Action.REQUEST_GATE: ANYONE,
    Action.ASK: ANYONE,
    Action.SUBMIT_PLAN: frozenset({"program_lead", "engineering_manager"}),
    Action.APPROVE_PLAN: frozenset({"program_lead", "engineering_manager"}),
    Action.RUN_PLAN: frozenset({"program_lead", "engineering_manager"}),
    Action.TRANSITION_TASK: frozenset({"program_lead", "engineering_manager"}),
    Action.SET_BUDGET: frozenset({"program_lead", "finance"}),
    Action.REGISTER_SOURCE: frozenset({"data_owner", "ml_lead"}),
    Action.BUILD_DATASET: frozenset({"data_owner", "ml_lead"}),
    Action.RELEASE_DATASET: frozenset({"ml_lead", "data_owner"}),
    Action.WITHDRAW_SOURCE: frozenset({"data_owner", "compliance"}),
    Action.CREATE_EVAL_SET: frozenset({"ml_lead", "domain_expert"}),
    Action.FREEZE_EVAL_SET: frozenset({"ml_lead"}),
    Action.TRAIN_MODEL: frozenset({"ml_lead"}),
    Action.EVALUATE_MODEL: frozenset({"ml_lead"}),
    Action.REGISTER_MODEL: frozenset({"ml_lead"}),
    Action.PROMOTE_MODEL: frozenset({"ml_lead"}),
    Action.SET_SERVING_POLICY: frozenset({"ml_lead"}),
    Action.DECIDE_POLICY: frozenset({"program_lead", "legal"}),
    Action.WRITE_BRIEF: frozenset({"program_lead", "ml_lead"}),
    Action.WRITE_RECIPE: frozenset({"ml_lead", "program_lead"}),
    Action.MANAGE_PEOPLE: frozenset({"program_lead", "engineering_manager"}),
    Action.VIEW_TRACES: frozenset({"ml_lead", "program_lead", "compliance", "domain_expert"}),
}


class PermissionDenied(PermissionError):
    pass


def allowed(actor: Actor, action: Action) -> bool:
    if actor.kind != "human":
        return False
    roles = ACTION_ROLES[action]
    return not roles or bool(actor.roles & roles)


def reason(action: Action) -> str:
    """Why a button is disabled, in the words shown next to it."""
    roles = ACTION_ROLES[action]
    if not roles:
        return "needs a signed-in person"
    return "needs " + " or ".join(sorted(roles))


def deny(audit: AuditLog, actor: Actor, action: str, subject: str, why: str) -> PermissionDenied:
    """Record a refusal and return the error for the caller to raise."""
    audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="console.denied",
        subject=subject or action,
        outcome=AuditOutcome.DENIED,
        details={"action": action, "reason": why, "authenticated": actor.authenticated},
    )
    return PermissionDenied(why)


def require(audit: AuditLog, actor: Actor, action: Action, subject: str = "") -> None:
    if not allowed(actor, action):
        raise deny(
            audit, actor, str(action), subject, f"{action.replace('_', ' ')} {reason(action)}"
        )
