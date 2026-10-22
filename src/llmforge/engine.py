"""Orchestrator engine: dispatch, gates, budget enforcement, loop detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from llmforge.agent import AgentRuntime
from llmforge.audit import AuditLog
from llmforge.budget import BudgetManager
from llmforge.domain import (
    AuditOutcome,
    BudgetScope,
    GateStatus,
    GateType,
    StatusReport,
    Task,
    TaskState,
)
from llmforge.gates import GATE_CATALOG, GateService
from llmforge.ids import new_id, now_iso
from llmforge.planner import TaskDraft
from llmforge.roles import get_role, reviewer_for
from llmforge.store import Store


@dataclass(slots=True)
class EngineConfig:
    max_attempts: int = 3
    max_iterations: int = 200
    stale_minutes: int = 30
    respect_budget: bool = True
    request_gates: bool = True


@dataclass(slots=True)
class DispatchOutcome:
    task_id: str
    state: str
    reason: str
    cost_usd: float = 0.0
    artifacts: list[str] = field(default_factory=list)


class BudgetBlocked(RuntimeError):
    pass


class Orchestrator:
    def __init__(
        self,
        store: Store,
        audit: AuditLog,
        gates: GateService,
        budget: BudgetManager,
        runtime: AgentRuntime,
        config: EngineConfig | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.gates = gates
        self.budget = budget
        self.runtime = runtime
        self.config = config or EngineConfig()
        self.escalations: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ plan

    def submit_plan(
        self,
        drafts: list[TaskDraft],
        approved_by: str | None = None,
        approver_kind: str = "human",
    ) -> list[Task]:
        ids = {draft.key: f"tsk_{draft.key}" for draft in drafts}
        tasks: list[Task] = []
        for draft in drafts:
            task = Task(
                id=ids[draft.key],
                title=draft.title,
                description=draft.description,
                kind=draft.kind,
                owner_role=draft.owner_role,
                dependencies=[ids[dep] for dep in draft.dependencies if dep in ids],
                acceptance_criteria=list(draft.acceptance_criteria),
                cost_estimate_usd=draft.cost_estimate_usd,
                cost_cap_usd=draft.cost_estimate_usd * 1.5,
                milestone=draft.milestone,
                reviewer_role=reviewer_for(draft.owner_role),
                metadata={**draft.metadata, "goal_key": draft.key},
            )
            if draft.gate:
                task.metadata["gate_type"] = draft.gate
            if approved_by:
                task.state = TaskState.APPROVED
            self.store.add_task(task)
            tasks.append(task)
        self.audit.record(
            actor=approved_by or "orchestrator",
            actor_kind=approver_kind,
            action="plan.submitted",
            subject=f"plan:{new_id('plan')}",
            outcome=AuditOutcome.SUCCESS,
            details={
                "tasks": [task.id for task in tasks],
                "cost_estimate_usd": round(sum(task.cost_estimate_usd for task in tasks), 4),
            },
        )
        if approved_by:
            for task in tasks:
                self.store.record_task_event(
                    task,
                    str(TaskState.PROPOSED),
                    str(TaskState.APPROVED),
                    approved_by,
                    approver_kind,
                    "plan approved",
                )
        if self.config.request_gates:
            self._open_gates(tasks)
        return tasks

    def approve_plan(self, approver: str, approver_kind: str = "human") -> int:
        approved = 0
        for task in self.store.list_tasks(state=str(TaskState.PROPOSED)):
            self.store.set_task_state(
                task, TaskState.APPROVED, approver, approver_kind, "plan approved"
            )
            approved += 1
        self.audit.record(
            actor=approver,
            actor_kind=approver_kind,
            action="plan.approved",
            subject="plan",
            outcome=AuditOutcome.SUCCESS,
            details={"tasks_approved": approved},
        )
        return approved

    def _open_gates(self, tasks: list[Task]) -> None:
        for task in tasks:
            gate_type = task.metadata.get("gate_type")
            if not gate_type:
                continue
            try:
                resolved = GateType(str(gate_type))
            except ValueError:
                continue
            entry = GATE_CATALOG[resolved]
            evidence = {
                key: f"{task.title}: {'; '.join(task.acceptance_criteria)}"
                for key in entry.evidence
            }
            evidence["task_id"] = task.id
            gate = self.gates.request(
                resolved,
                title=f"{task.title} requires {resolved.replace('_', ' ')}",
                subject=task.id,
                requested_by="orchestrator",
                requested_by_kind="agent",
                evidence=evidence,
                cost_usd=task.cost_estimate_usd,
            )
            task.gate_id = gate.id
            self.store.update_task(task)
            self.audit.record(
                actor="orchestrator",
                action="gate.requested",
                subject=task.id,
                outcome=AuditOutcome.REQUESTED,
                details={"gate_id": gate.id, "gate_type": str(resolved)},
            )

    # ------------------------------------------------------------- dispatch

    def ready_tasks(self) -> list[Task]:
        ready: list[Task] = []
        for state in (TaskState.APPROVED, TaskState.IN_PROGRESS):
            for task in self.store.list_tasks(state=str(state)):
                if self._dependencies_met(task):
                    ready.append(task)
        return ready

    def release_unblocked(self) -> list[Task]:
        """Move tasks back into the queue once their gate has been approved."""
        released: list[Task] = []
        for task in self.store.list_tasks(state=str(TaskState.BLOCKED)):
            if not task.gate_id:
                continue
            gate = self.store.get_gate(task.gate_id)
            if gate is None or gate.status != GateStatus.APPROVED:
                continue
            self.store.set_task_state(
                task,
                TaskState.APPROVED,
                "orchestrator",
                "agent",
                f"gate {gate.id} approved",
            )
            released.append(task)
        return released

    def _dependencies_met(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            dep = self.store.get_task(dep_id)
            if dep is None or dep.state != TaskState.ACCEPTED:
                return False
        return True

    def _gate_ready(self, task: Task) -> tuple[bool, str]:
        if not task.gate_id:
            return True, ""
        gate = self.store.get_gate(task.gate_id)
        if gate is None:
            return False, f"gate {task.gate_id} missing"
        if gate.status == GateStatus.APPROVED:
            return True, ""
        if gate.status == GateStatus.REJECTED:
            return False, f"gate {gate.id} was rejected: {gate.decision_note}"
        return False, f"awaiting human approval on gate {gate.id} ({gate.gate_type})"

    def dispatch(self, task: Task) -> DispatchOutcome:
        if self.budget.kill_switch_engaged():
            return DispatchOutcome(task.id, str(task.state), "global kill switch engaged")
        gate_ok, gate_reason = self._gate_ready(task)
        if not gate_ok:
            if task.state != TaskState.BLOCKED:
                self.store.set_task_state(
                    task, TaskState.BLOCKED, "orchestrator", "agent", gate_reason
                )
            return DispatchOutcome(task.id, str(task.state), gate_reason)
        if self.config.respect_budget:
            decision = self.budget.check(task.owner_role, task_id=task.id)
            if not decision.allowed:
                self.store.set_task_state(
                    task, TaskState.BLOCKED, "orchestrator", "agent", decision.reason
                )
                self.audit.record(
                    actor="orchestrator",
                    action="budget.blocked",
                    subject=task.id,
                    outcome=AuditOutcome.DENIED,
                    details={"reason": decision.reason},
                )
                return DispatchOutcome(task.id, str(task.state), decision.reason)
        role = get_role(task.owner_role)
        self.store.set_task_state(task, TaskState.IN_PROGRESS, role.name, "agent", "agent started")
        task.attempts += 1
        upstream = self._upstream_artifacts(task)
        result = self.runtime.execute(role, task, upstream=upstream)
        task.actual_cost_usd += result.cost_usd
        task.artifacts = list(dict.fromkeys([*task.artifacts, *result.artifacts]))
        self.store.update_task(task)
        self.store.record_run(_run_summary(task.id, role.name, result))
        if result.status != "ok":
            self.store.set_task_state(
                task, TaskState.IN_PROGRESS, role.name, "agent", f"build failed: {result.summary}"
            )
            self._check_loops(task)
            return DispatchOutcome(task.id, str(task.state), result.summary, result.cost_usd)
        self.store.set_task_state(
            task, TaskState.IN_REVIEW, role.name, "agent", "submitted for review"
        )
        reviewer_role = get_role(task.reviewer_role or reviewer_for(task.owner_role))
        decision = self.runtime.review(reviewer_role, task, result, builder_role=role.name)
        self.store.set_task_state(
            task,
            TaskState.ACCEPTED if decision.approved else TaskState.IN_PROGRESS,
            reviewer_role.name,
            "agent",
            decision.reason,
        )
        if not decision.approved:
            task.review_attempts += 1
            self.store.update_task(task)
            self._check_loops(task)
        return DispatchOutcome(
            task.id,
            str(task.state),
            decision.reason,
            result.cost_usd,
            result.artifacts,
        )

    def _upstream_artifacts(self, task: Task) -> list[str]:
        artifacts: list[str] = []
        for dep_id in task.dependencies:
            dep = self.store.get_task(dep_id)
            if dep:
                artifacts.extend(dep.artifacts)
        return artifacts

    def _check_loops(self, task: Task) -> None:
        if task.attempts > self.config.max_attempts:
            self.escalate(task, f"attempt limit exceeded ({task.attempts} attempts)")
            return
        last = task.last_state_change
        if last:
            try:
                age = (datetime.now(UTC) - datetime.fromisoformat(last)).total_seconds() / 60
            except ValueError:
                return
            if age > self.config.stale_minutes and task.state == TaskState.IN_PROGRESS:
                self.escalate(task, f"no state change for {age:.0f} minutes")

    def escalate(self, task: Task, reason: str) -> None:
        self.store.set_task_state(task, TaskState.ESCALATED, "orchestrator", "agent", reason)
        self.audit.record(
            actor="orchestrator",
            action="task.escalated",
            subject=task.id,
            outcome=AuditOutcome.FAILURE,
            details={"reason": reason, "attempts": task.attempts},
        )
        self.escalations.append({"task_id": task.id, "reason": reason, "at": now_iso()})

    def run(self, max_iterations: int | None = None) -> dict[str, Any]:
        limit = max_iterations or self.config.max_iterations
        outcomes: list[DispatchOutcome] = []
        iterations = 0
        while iterations < limit:
            iterations += 1
            if self.budget.kill_switch_engaged():
                break
            self.release_unblocked()
            ready = self.ready_tasks()
            if not ready:
                break
            for task in ready:
                outcome = self.dispatch(task)
                outcomes.append(outcome)
                if self.budget.kill_switch_engaged():
                    break
            if all(
                outcome.state in (str(TaskState.IN_PROGRESS), str(TaskState.IN_REVIEW))
                for outcome in outcomes[-len(ready) :]
            ):
                continue
        summary = {
            "iterations": iterations,
            "dispatched": len(outcomes),
            "accepted": sum(1 for item in outcomes if item.state == str(TaskState.ACCEPTED)),
            "blocked": sum(1 for item in outcomes if item.state == str(TaskState.BLOCKED)),
            "escalated": len(self.escalations),
            "spend_usd": round(self.budget.store.spend_by(BudgetScope.GLOBAL, "global"), 4),
            "outcomes": [asdict(outcome) for outcome in outcomes],
        }
        self.audit.record(
            actor="orchestrator",
            action="orchestrator.run",
            subject="orchestrator",
            outcome=AuditOutcome.SUCCESS,
            details={key: value for key, value in summary.items() if key != "outcomes"},
        )
        return summary

    # --------------------------------------------------------------- report

    def status_report(self) -> StatusReport:
        tasks = self.store.list_tasks()
        by_state: dict[str, int] = {}
        for task in tasks:
            by_state[str(task.state)] = by_state.get(str(task.state), 0) + 1
        blocked = []
        for task in tasks:
            if task.state in (TaskState.BLOCKED, TaskState.ESCALATED):
                blocked.append(
                    {
                        "id": task.id,
                        "title": task.title,
                        "state": str(task.state),
                        "reason": task.blocked_reason,
                        "owner_role": task.owner_role,
                    }
                )
        spend_by_agent: dict[str, float] = {}
        for record in self.store.conn.execute(
            "SELECT agent, COALESCE(SUM(amount_usd), 0) AS total FROM spend_records "
            "WHERE scope = ? GROUP BY agent",
            (str(BudgetScope.AGENT),),
        ).fetchall():
            spend_by_agent[record["agent"]] = float(record["total"])
        total_spend = self.store.spend_by(BudgetScope.GLOBAL, "global")
        limits = {
            f"{limit.scope}:{limit.key}": limit.limit_usd
            for limit in self.store.list_budget_limits()
        }
        budget_usd = limits.get("global:global", 0.0)
        open_gates = [
            {
                "id": gate.id,
                "gate_type": str(gate.gate_type),
                "title": gate.title,
                "subject": gate.subject,
                "required_approvers": list(gate.required_approvers),
                "created_at": gate.created_at,
            }
            for gate in self.gates.pending()
        ]
        risks = [f"{item['task_id']}: {item['reason']}" for item in self.escalations]
        if not summary_under_budget(self.budget, total_spend, budget_usd):
            risks.append("spend exceeds the global budget cap")
        return StatusReport(
            generated_at=now_iso(),
            tasks_by_state=by_state,
            blocked_tasks=blocked,
            spend_by_agent=spend_by_agent,
            spend_total_usd=total_spend,
            budget_usd=budget_usd,
            open_gates=open_gates,
            risks=risks,
        )


def summary_under_budget(budget: BudgetManager, spent: float, cap: float) -> bool:
    if cap <= 0:
        return True
    return spent <= cap


def _run_summary(task_id: str, agent: str, result: Any) -> Any:
    from llmforge.domain import RunSummary

    return RunSummary(
        task_id=task_id,
        agent=agent,
        status=result.status,
        output=result.summary,
        artifacts=list(result.artifacts),
        cost_usd=result.cost_usd,
        duration_s=result.duration_s,
        started_at=now_iso(),
        finished_at=now_iso(),
        metadata=result.metadata,
    )
