"""Domain models for Internal-AI.

These types are the contract between the orchestrator, role agents, the gate
service and the audit log. They map directly onto the PRD: task states (FR-2),
agent roles (section 7), gates (section 14) and budgets (section 15).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TaskState(StrEnum):
    """Lifecycle states for a task (FR-2)."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    ESCALATED = "escalated"


TERMINAL_STATES = frozenset({TaskState.ACCEPTED, TaskState.REJECTED})
ACTIVE_STATES = frozenset(
    {
        TaskState.APPROVED,
        TaskState.IN_PROGRESS,
        TaskState.IN_REVIEW,
    }
)


class TaskKind(StrEnum):
    PLAN = "plan"
    DATA = "data"
    EVAL = "eval"
    RESEARCH = "research"
    TRAIN = "train"
    INFRA = "infra"
    SERVE = "serve"
    REVIEW = "review"
    GATE = "gate"
    OPS = "ops"


class GateType(StrEnum):
    """Human-in-the-loop gates from PRD section 14."""

    NEW_DATA_SOURCE = "new_data_source_access"
    DATASET_RELEASE = "dataset_release"
    EVAL_SET_FREEZE = "eval_set_freeze"
    TRAINING_RUN_COST = "training_run_above_threshold"
    MODEL_PROMOTION = "model_promotion"
    PRODUCTION_DEPLOYMENT = "production_deployment"
    BUDGET_INCREASE = "budget_increase"
    POLICY_DECISION = "policy_decision"


class GateStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    REQUESTED = "requested"


class BudgetScope(StrEnum):
    GLOBAL = "global"
    PROJECT = "project"
    AGENT = "agent"
    JOB = "job"
    WEEKLY = "weekly"


class Capability(StrEnum):
    """Least-privilege capabilities granted to agents (principle 4)."""

    TASK_READ = "task.read"
    TASK_WRITE = "task.write"
    TASK_BOARD_WRITE = "task.board.write"
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATASET_WRITE = "dataset.write"
    CATALOG_READ = "catalog.read"
    EVAL_READ = "eval.read"
    EVAL_WRITE = "eval.write"
    EVAL_FREEZE = "eval.freeze"
    MODEL_READ = "model.read"
    MODEL_WRITE = "model.write"
    MODEL_PROMOTE = "model.promote"
    TRAIN_RUN = "train.run"
    JOB_SUBMIT = "job.submit"
    BUDGET_READ = "budget.read"
    BUDGET_WRITE = "budget.write"
    DEPLOY = "deploy"
    AUDIT_READ = "audit.read"
    GATE_REQUEST = "gate.request"
    GATE_APPROVE = "gate.approve"
    NET_EGRESS = "net.egress"
    SECRET_READ = "secret.read"


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """Static definition of a role agent (PRD section 7)."""

    name: str
    mission: str
    sub_agents: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    capabilities: frozenset[Capability]
    gates: tuple[GateType, ...]
    kpis: tuple[str, ...]
    enabled_by_default: bool = False

    def can(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mission": self.mission,
            "sub_agents": list(self.sub_agents),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "capabilities": sorted(self.capabilities),
            "gates": list(self.gates),
            "kpis": list(self.kpis),
            "enabled_by_default": self.enabled_by_default,
        }


@dataclass(slots=True)
class Task:
    """A unit of work owned by a role agent."""

    id: str
    title: str
    description: str
    kind: TaskKind
    owner_role: str
    state: TaskState = TaskState.PROPOSED
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    cost_estimate_usd: float = 0.0
    cost_cap_usd: float = 0.0
    actual_cost_usd: float = 0.0
    attempts: int = 0
    review_attempts: int = 0
    artifacts: list[str] = field(default_factory=list)
    reviewer_role: str | None = None
    gate_id: str | None = None
    milestone: str | None = None
    blocked_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    last_state_change: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "kind": str(self.kind),
            "owner_role": self.owner_role,
            "state": str(self.state),
            "dependencies": list(self.dependencies),
            "acceptance_criteria": list(self.acceptance_criteria),
            "cost_estimate_usd": self.cost_estimate_usd,
            "cost_cap_usd": self.cost_cap_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "attempts": self.attempts,
            "review_attempts": self.review_attempts,
            "artifacts": list(self.artifacts),
            "reviewer_role": self.reviewer_role,
            "gate_id": self.gate_id,
            "milestone": self.milestone,
            "blocked_reason": self.blocked_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class TaskEvent:
    id: str
    task_id: str
    from_state: str | None
    to_state: str
    actor: str
    actor_kind: str
    note: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GateRequest:
    id: str
    gate_type: GateType
    title: str
    subject: str
    requested_by: str
    requested_by_kind: str
    required_approvers: tuple[str, ...]
    evidence: dict[str, Any] = field(default_factory=dict)
    status: GateStatus = GateStatus.PENDING
    created_at: str = ""
    decided_at: str | None = None
    decision_note: str = ""
    cost_usd: float = 0.0


@dataclass(slots=True)
class GateDecision:
    id: str
    gate_id: str
    approver: str
    approver_kind: str
    approved: bool
    note: str
    created_at: str


@dataclass(slots=True)
class AuditEvent:
    id: str
    seq: int
    actor: str
    actor_kind: str
    action: str
    subject: str
    outcome: AuditOutcome
    inputs: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    prev_hash: str = ""
    hash: str = ""
    created_at: str = ""
    tenant: str = "default"


@dataclass(slots=True)
class BudgetLimit:
    scope: BudgetScope
    key: str
    limit_usd: float
    period_start: str
    period_end: str | None = None


@dataclass(slots=True)
class SpendRecord:
    id: str
    scope: BudgetScope
    key: str
    amount_usd: float
    agent: str
    task_id: str | None
    job_id: str | None
    created_at: str
    notes: str = ""


@dataclass(slots=True)
class RunSummary:
    """Result of a single agent execution attempt."""

    task_id: str
    agent: str
    status: str
    output: str
    artifacts: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_s: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetVersion:
    id: str
    name: str
    version: int
    content_hash: str
    record_count: int
    token_count: int
    created_at: str
    lineage: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    released: bool = False
    path: str = ""
    parent_version: str | None = None


@dataclass(slots=True)
class ModelVersion:
    id: str
    name: str
    version: int
    content_hash: str
    created_at: str
    dataset_ids: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    eval_report_id: str | None = None
    status: str = "candidate"
    card: dict[str, Any] = field(default_factory=dict)
    path: str = ""


@dataclass(slots=True)
class EvalReport:
    id: str
    model_id: str
    eval_set_id: str
    created_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    notes: str = ""


@dataclass(slots=True)
class EvalSet:
    id: str
    name: str
    version: int
    frozen: bool
    created_at: str
    approved_by: str | None = None
    content_hash: str = ""
    path: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PolicyDecision:
    id: str
    question: str
    default_value: str
    current_value: str
    rationale: str
    decided_by: str | None = None
    decided_at: str | None = None
    status: str = "undecided"


@dataclass(slots=True)
class StatusReport:
    generated_at: str
    tasks_by_state: dict[str, int]
    blocked_tasks: list[dict[str, Any]]
    spend_by_agent: dict[str, float]
    spend_total_usd: float
    budget_usd: float
    open_gates: list[dict[str, Any]]
    risks: list[str] = field(default_factory=list)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)
