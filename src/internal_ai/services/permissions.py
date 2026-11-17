"""Which human roles may take which action, and on which records.

The map is the specification. An action mapped to `ANYONE` is open to every
signed-in person; every other action names the roles that may take it. Gate
decisions are checked against the roles the gate itself requires, so they are
not in the map.

A role answers whether a person may take the action at all. When the caller
names a record, `services/attributes.py` answers whether they may take it on
that one, and `require` asks both: an attribute rule kept in a second call is a
rule a caller can forget to make.
"""

from __future__ import annotations

from enum import StrEnum

from internal_ai.audit import AuditLog
from internal_ai.domain import AuditOutcome
from internal_ai.services.actor import Actor
from internal_ai.services.attributes import ReadPolicy, Resource


class Action(StrEnum):
    ENGAGE_KILL_SWITCH = "engage_kill_switch"
    RELEASE_KILL_SWITCH = "release_kill_switch"
    ACKNOWLEDGE_ALERT = "acknowledge_alert"
    REQUEST_GATE = "request_gate"
    ASK = "ask"
    USE_WEB_GATE = "use_web_gate"
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
    SET_EXTRACTION_POLICY = "set_extraction_policy"
    DECIDE_POLICY = "decide_policy"
    WRITE_BRIEF = "write_brief"
    WRITE_RECIPE = "write_recipe"
    MANAGE_PEOPLE = "manage_people"
    VIEW_TRACES = "view_traces"
    ANCHOR_AUDIT = "anchor_audit"
    ADVISE_COMPUTE = "advise_compute"
    ATTACH_COMPUTE = "attach_compute"
    VERIFY_ARTIFACT = "verify_artifact"
    AUDIT_DATASET = "audit_dataset"
    CLOSE_BUDGET_PERIOD = "close_budget_period"
    EXPORT_SPEND = "export_spend"


ANYONE: frozenset[str] = frozenset()

ACTION_ROLES: dict[Action, frozenset[str]] = {
    Action.ENGAGE_KILL_SWITCH: ANYONE,
    Action.RELEASE_KILL_SWITCH: ANYONE,
    Action.ACKNOWLEDGE_ALERT: ANYONE,
    Action.REQUEST_GATE: ANYONE,
    Action.ASK: ANYONE,
    Action.USE_WEB_GATE: ANYONE,
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
    Action.SET_EXTRACTION_POLICY: frozenset({"ml_lead", "security"}),
    Action.DECIDE_POLICY: frozenset({"program_lead", "legal"}),
    Action.WRITE_BRIEF: frozenset({"program_lead", "ml_lead"}),
    Action.WRITE_RECIPE: frozenset({"ml_lead", "program_lead"}),
    # Anchoring is the ledger's own integrity operation, so it sits with the two
    # roles that answer for evidence rather than with whoever runs the platform.
    Action.ANCHOR_AUDIT: frozenset({"security", "compliance"}),
    # Reading what a corpus needs is open to anyone signed in; deciding which
    # machine the platform will use is the infra engineer's call, and it is
    # recorded as such.
    Action.ADVISE_COMPUTE: ANYONE,
    Action.ATTACH_COMPUTE: frozenset({"infra_engineer"}),
    Action.VERIFY_ARTIFACT: frozenset({"ml_lead", "security", "infra_engineer"}),
    Action.AUDIT_DATASET: frozenset({"data_owner", "compliance", "ml_lead"}),
    Action.CLOSE_BUDGET_PERIOD: frozenset({"program_lead", "compliance"}),
    Action.EXPORT_SPEND: frozenset({"program_lead", "compliance"}),
    Action.MANAGE_PEOPLE: frozenset({"program_lead", "engineering_manager"}),
    Action.VIEW_TRACES: frozenset({"ml_lead", "program_lead", "compliance", "domain_expert"}),
}


class PermissionDenied(PermissionError):
    pass


def allowed(
    actor: Actor,
    action: Action,
    resource: Resource | None = None,
    policy: ReadPolicy | None = None,
) -> bool:
    """May this actor take this action -- and, when one is named, on this record.

    The role and the attributes are one answer rather than two questions a
    caller has to remember to ask in order. A caller that names no record gets
    the role answer alone, which is what every button on a page needs.
    """
    if actor.kind != "human":
        return False
    roles = ACTION_ROLES[action]
    if roles and not (actor.roles & roles):
        return False
    if resource is None:
        return True
    grants = policy or ReadPolicy.strict()
    return grants.allows(actor.groups, resource.sensitivity, resource.acl)


def reason(action: Action) -> str:
    """Why a button is disabled, in the words shown next to it."""
    roles = ACTION_ROLES[action]
    if not roles:
        return "needs a signed-in person"
    return "needs " + " or ".join(sorted(roles))


def attribute_reason(resource: Resource, policy: ReadPolicy) -> str:
    """Why one record is out of reach, in the words shown next to it.

    The two refusals are not the same refusal: a class nobody granted is a policy
    decision, while an unreadable record of a readable class is the record's own
    ACL talking. A message that blurred them would send a person to the wrong
    administrator.
    """
    if resource.data_class not in policy.grants:
        return f"needs a group granted {resource.data_class} {resource.kind} material"
    groups = ", ".join(sorted(resource.acl)) or "none"
    return f"needs one of the groups this {resource.kind} belongs to ({groups})"


def deny(
    audit: AuditLog,
    actor: Actor,
    action: str,
    subject: str,
    why: str,
    attribute: dict[str, object] | None = None,
) -> PermissionDenied:
    """Record a refusal and return the error for the caller to raise."""
    details: dict[str, object] = {
        "action": action,
        "reason": why,
        "authenticated": actor.authenticated,
    }
    if attribute is not None:
        # Which half refused: the role or the record. Without it a reviewer cannot
        # tell a person who lacks the capability from one reading another group's data.
        details["attribute"] = attribute
    audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="console.denied",
        subject=subject or action,
        outcome=AuditOutcome.DENIED,
        details=details,
    )
    return PermissionDenied(why)


def require(
    audit: AuditLog,
    actor: Actor,
    action: Action,
    subject: str = "",
    resource: Resource | None = None,
    policy: ReadPolicy | None = None,
) -> None:
    """Refuse unless the role and, when a record is named, its attributes allow it.

    The two refusals read differently on purpose: the first names the roles the
    action needs, the second names the class of the record and the kind of thing
    it is, and never a document id or its content.
    """
    label = str(action).replace("_", " ")
    roles = ACTION_ROLES[action]
    if actor.kind != "human" or (roles and not (actor.roles & roles)):
        raise deny(audit, actor, str(action), subject, f"{label} {reason(action)}")
    if resource is None:
        return
    grants = policy or ReadPolicy.strict()
    if grants.allows(actor.groups, resource.sensitivity, resource.acl):
        return
    raise deny(
        audit,
        actor,
        str(action),
        subject,
        f"{label} {attribute_reason(resource, grants)}",
        attribute={
            "kind": resource.kind,
            "class": resource.data_class,
            "acl": sorted(resource.acl),
            "groups": sorted(actor.groups),
        },
    )
