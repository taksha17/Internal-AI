"""Control plane HTTP API.

Read and write access to the task board, gates, audit log, budgets, policy and
reports. Writes that carry governance weight (gate decisions, plan approval,
budget changes) require the caller's identity and are attributed in the audit
log.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from llmforge.context import Forge
from llmforge.domain import BudgetScope, TaskState
from llmforge.gates import GateError
from llmforge.planner import LLMPlanner, TemplatePlanner
from llmforge.store import TransitionError

API_KEY_HEADER = "X-Forge-Key"


class PlanRequest(BaseModel):
    goal: str
    milestone: str | None = None
    planner: str = "template"
    approve_by: str | None = None


class RunRequest(BaseModel):
    max_iterations: int = Field(default=50, ge=1, le=1000)
    approve_by: str | None = None


class GateDecisionRequest(BaseModel):
    approver: str
    approved: bool = True
    note: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class PolicyRequest(BaseModel):
    value: str
    requested_by: str
    rationale: str = ""


class BudgetLimitRequest(BaseModel):
    scope: str
    key: str
    limit_usd: float
    actor: str = "program_lead"


def create_app(
    root: Path | None = None, llm_kind: str = "auto", api_key: str | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        forge = Forge.create(root=root, llm_kind=llm_kind)
        app.state.forge = forge
        try:
            yield
        finally:
            forge.close()

    app = FastAPI(
        title="LLM-Forge control plane",
        version="0.1.0",
        description="Orchestration, gates, budgets, lineage and audit for the in-house LLM build.",
        lifespan=lifespan,
    )
    key = api_key or os.environ.get("LLMFORGE_API_KEY", "")

    def get_forge() -> Forge:
        forge: Forge | None = getattr(app.state, "forge", None)
        if forge is None:
            forge = Forge.create(root=root, llm_kind=llm_kind)
            app.state.forge = forge
        return forge

    def require_key(x_forge_key: str | None = Header(default=None)) -> None:
        if not key:
            return
        if x_forge_key != key:
            raise HTTPException(status_code=401, detail="invalid or missing X-Forge-Key")

    @app.get("/health")
    def health() -> dict[str, Any]:
        forge = get_forge()
        return {
            "status": "ok",
            "kill_switch": forge.budget.kill_switch_engaged(),
            "budget": [status.to_dict() for status in forge.budget.all_status()],
        }

    @app.get("/tasks")
    def list_tasks(
        state: str | None = None,
        owner_role: str | None = None,
        _: None = Depends(require_key),
    ) -> list[dict[str, Any]]:
        forge = get_forge()
        return [
            task.to_dict() for task in forge.store.list_tasks(state=state, owner_role=owner_role)
        ]

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str, _: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        task = forge.store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown task")
        events = [event.__dict__ for event in forge.store.task_events(task_id)]
        return {"task": task.to_dict(), "events": events}

    @app.post("/plan")
    def submit_plan(request: PlanRequest, _: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        planner = (
            TemplatePlanner()
            if request.planner == "template"
            else LLMPlanner(llm=forge.runtime.llm, fallback=TemplatePlanner())
        )
        drafts = planner.plan(request.goal, {"milestone": request.milestone})
        tasks = forge.orchestrator.submit_plan(drafts, approved_by=request.approve_by)
        return {"tasks": [task.to_dict() for task in tasks]}

    @app.post("/run")
    def run(request: RunRequest, _: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        if request.approve_by:
            forge.orchestrator.approve_plan(request.approve_by)
        summary = forge.orchestrator.run(max_iterations=request.max_iterations)
        summary.pop("outcomes", None)
        return summary

    @app.get("/gates")
    def list_gates(
        status: str | None = None, _: None = Depends(require_key)
    ) -> list[dict[str, Any]]:
        forge = get_forge()
        gates = forge.store.list_gates(status=status)
        return [
            {
                "id": gate.id,
                "gate_type": str(gate.gate_type),
                "title": gate.title,
                "subject": gate.subject,
                "status": str(gate.status),
                "requested_by": gate.requested_by,
                "required_approvers": list(gate.required_approvers),
                "evidence": gate.evidence,
                "created_at": gate.created_at,
                "decided_at": gate.decided_at,
            }
            for gate in gates
        ]

    @app.post("/gates/{gate_id}/decision")
    def decide_gate(
        gate_id: str, request: GateDecisionRequest, _: None = Depends(require_key)
    ) -> dict[str, Any]:
        forge = get_forge()
        try:
            gate = forge.gates.decide(
                gate_id,
                request.approver,
                approved=request.approved,
                approver_kind="human",
                note=request.note,
                evidence=request.evidence,
            )
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        forge.audit.record(
            actor=request.approver,
            actor_kind="human",
            action="gate.approved" if request.approved else "gate.rejected",
            subject=gate.id,
            details={"gate_type": str(gate.gate_type), "note": request.note},
        )
        return {"id": gate.id, "status": str(gate.status)}

    @app.get("/audit")
    def audit(
        limit: int = 100,
        action: str | None = None,
        actor: str | None = None,
        _: None = Depends(require_key),
    ) -> list[dict[str, Any]]:
        forge = get_forge()
        events = forge.audit.recent(limit=limit, action=action, actor=actor)
        return [
            {
                "seq": event.seq,
                "id": event.id,
                "actor": event.actor,
                "actor_kind": event.actor_kind,
                "action": event.action,
                "subject": event.subject,
                "outcome": str(event.outcome),
                "cost_usd": event.cost_usd,
                "created_at": event.created_at,
                "hash": event.hash,
            }
            for event in events
        ]

    @app.get("/audit/verify")
    def verify_audit(_: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        return forge.audit.verify().to_dict()

    @app.get("/budget")
    def budget(_: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        return {
            "status": [status.to_dict() for status in forge.budget.all_status()],
            "kill_switch": forge.budget.kill_switch_engaged(),
        }

    @app.post("/budget/limits")
    def set_limit(request: BudgetLimitRequest, _: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        try:
            scope = BudgetScope(request.scope)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown scope: {request.scope}") from exc
        forge.budget.set_limit(scope, request.key, request.limit_usd)
        forge.audit.record(
            actor=request.actor,
            actor_kind="human",
            action="budget.limit_set",
            subject=f"{scope}:{request.key}",
            details={"limit_usd": request.limit_usd},
        )
        return {"scope": str(scope), "key": request.key, "limit_usd": request.limit_usd}

    @app.get("/policy")
    def policy(_: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        return forge.policy.status_summary()

    @app.post("/policy/{policy_id}")
    def decide_policy(
        policy_id: str, request: PolicyRequest, _: None = Depends(require_key)
    ) -> dict[str, Any]:
        forge = get_forge()
        try:
            decision, gate_id = forge.policy.propose(
                policy_id,
                request.value,
                requested_by=request.requested_by,
                rationale=request.rationale,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        forge.audit.record(
            actor=request.requested_by,
            actor_kind="human",
            action="policy.proposed",
            subject=policy_id,
            details={"value": request.value, "gate_id": gate_id},
        )
        return {"id": decision.id, "status": decision.status, "gate_id": gate_id}

    @app.get("/report")
    def report(_: None = Depends(require_key)) -> dict[str, Any]:
        forge = get_forge()
        status = forge.orchestrator.status_report()
        return {
            "generated_at": status.generated_at,
            "tasks_by_state": status.tasks_by_state,
            "blocked_tasks": status.blocked_tasks,
            "spend_by_agent": status.spend_by_agent,
            "spend_total_usd": status.spend_total_usd,
            "budget_usd": status.budget_usd,
            "open_gates": status.open_gates,
            "risks": status.risks,
        }

    @app.get("/alerts")
    def alerts(
        all_alerts: bool = False,
        check: bool = False,
        _: None = Depends(require_key),
    ) -> list[dict[str, Any]]:
        from llmforge.alerts import AlertEngine

        forge = get_forge()
        engine = AlertEngine(forge.store, forge.audit, forge.budget)
        if check:
            engine.check_all()
        return [alert.to_dict() for alert in engine.list_alerts(unacknowledged_only=not all_alerts)]

    @app.post("/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(
        alert_id: str, actor: str = "program_lead", _: None = Depends(require_key)
    ) -> dict[str, Any]:
        from llmforge.alerts import AlertEngine

        forge = get_forge()
        engine = AlertEngine(forge.store, forge.audit, forge.budget)
        try:
            alert = engine.acknowledge(alert_id, actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return alert.to_dict()

    @app.get("/roles")
    def roles(_: None = Depends(require_key)) -> list[dict[str, Any]]:
        from llmforge.roles import all_roles

        return [role.to_dict() for role in all_roles()]

    @app.post("/tasks/{task_id}/transition")
    def transition(
        task_id: str,
        state: str,
        actor: str,
        _: None = Depends(require_key),
    ) -> dict[str, Any]:
        forge = get_forge()
        task = forge.store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown task")
        try:
            target = TaskState(state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown state: {state}") from exc
        try:
            forge.store.set_task_state(task, target, actor, "human", "manual transition")
        except TransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"id": task.id, "state": str(task.state)}

    return app
