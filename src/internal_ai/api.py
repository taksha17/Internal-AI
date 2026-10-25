"""Control plane HTTP API and the browser console.

Reads need the operator key. Writes that carry governance weight (plan
submission and approval, runs, gate requests and decisions, budget limits, the
kill switch, policy, task transitions, alert acknowledgement) also need a
personal token, and the actor is the person the token belongs to. No request
body or query parameter can name the actor.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from internal_ai.context import Platform
from internal_ai.engine import PlanError
from internal_ai.gates import GateError
from internal_ai.services import governance
from internal_ai.services.actor import Actor
from internal_ai.services.identity import AuthenticationError, IdentityService
from internal_ai.services.jobs import JobRunner
from internal_ai.services.permissions import PermissionDenied
from internal_ai.store import TransitionError

API_KEY_HEADER = "X-Internal-Key"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "form-action 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class PlanRequest(BaseModel):
    goal: str
    milestone: str | None = None
    planner: str = "template"
    approve: bool = False


class RunRequest(BaseModel):
    max_iterations: int = Field(default=50, ge=1, le=1000)
    approve: bool = False


class GateRequestBody(BaseModel):
    gate_type: str
    title: str
    subject: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class GateDecisionRequest(BaseModel):
    approved: bool = True
    note: str = ""


class PolicyRequest(BaseModel):
    value: str
    rationale: str = ""


class PolicyConfirmRequest(BaseModel):
    gate_id: str | None = None


class BudgetLimitRequest(BaseModel):
    scope: str
    key: str
    limit_usd: float
    gate_id: str | None = None


class KillSwitchRequest(BaseModel):
    engaged: bool
    reason: str = ""


def http_error(exc: Exception) -> HTTPException:
    """The status code a refusal deserves, so clients can tell them apart."""
    if isinstance(exc, AuthenticationError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, PermissionDenied | PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, GateError | TransitionError | PlanError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc.args[0] if exc.args else exc))
    return HTTPException(status_code=400, detail=str(exc))


REFUSALS = (
    AuthenticationError,
    PermissionError,
    GateError,
    TransitionError,
    PlanError,
    KeyError,
    ValueError,
)


def create_app(
    root: Path | None = None,
    llm_kind: str = "auto",
    api_key: str | None = None,
    secure_cookies: bool = False,
    sync_jobs: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        get_platform()
        try:
            yield
        finally:
            platform = getattr(app.state, "platform", None)
            if platform is not None:
                platform.close()
                app.state.platform = None

    app = FastAPI(
        title="Internal-AI control plane",
        version="0.2.0",
        description="Orchestration, gates, budgets, lineage and audit for the in-house LLM build.",
        lifespan=lifespan,
    )
    key = api_key or os.environ.get("INTERNAL_AI_API_KEY", "")

    def get_platform() -> Platform:
        platform: Platform | None = getattr(app.state, "platform", None)
        if platform is None:
            platform = Platform.create(root=root, llm_kind=llm_kind)
            app.state.platform = platform
            app.state.jobs = JobRunner(platform, synchronous=sync_jobs)
        return platform

    def get_jobs() -> JobRunner:
        get_platform()
        return app.state.jobs

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/console"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def require_key(x_internal_key: str | None = Header(default=None)) -> None:
        if not key:
            return
        if x_internal_key != key:
            raise HTTPException(status_code=401, detail="invalid or missing X-Internal-Key")

    def require_person(
        authorization: str | None = Header(default=None),
        _: None = Depends(require_key),
    ) -> Actor:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=401,
                detail="governance writes need a personal token: Authorization: Bearer <token>",
            )
        actor = IdentityService(get_platform()).token_actor(token.strip())
        if actor is None:
            raise HTTPException(status_code=401, detail="invalid or revoked personal token")
        return actor

    @app.get("/health")
    def health() -> dict[str, Any]:
        platform = get_platform()
        return {
            "status": "ok",
            "kill_switch": platform.budget.kill_switch_engaged(),
            "budget": [status.to_dict() for status in platform.budget.all_status()],
        }

    @app.get("/me")
    def me(actor: Actor = Depends(require_person)) -> dict[str, Any]:
        return actor.to_dict()

    @app.get("/tasks")
    def list_tasks(
        state: str | None = None,
        owner_role: str | None = None,
        _: None = Depends(require_key),
    ) -> list[dict[str, Any]]:
        platform = get_platform()
        return [
            task.to_dict() for task in platform.store.list_tasks(state=state, owner_role=owner_role)
        ]

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str, _: None = Depends(require_key)) -> dict[str, Any]:
        platform = get_platform()
        task = platform.store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown task")
        events = [event.__dict__ for event in platform.store.task_events(task_id)]
        return {"task": task.to_dict(), "events": events}

    @app.post("/plan")
    def submit_plan(request: PlanRequest, actor: Actor = Depends(require_person)) -> dict[str, Any]:
        try:
            tasks = governance.submit_plan(
                get_platform(),
                actor,
                request.goal,
                milestone=request.milestone,
                planner=request.planner,
                approve=request.approve,
            )
        except REFUSALS as exc:
            raise http_error(exc) from exc
        return {"tasks": [task.to_dict() for task in tasks]}

    @app.post("/run")
    def run(request: RunRequest, actor: Actor = Depends(require_person)) -> dict[str, Any]:
        try:
            return governance.run_plan(
                get_platform(),
                actor,
                max_iterations=request.max_iterations,
                approve=request.approve,
            )
        except REFUSALS as exc:
            raise http_error(exc) from exc

    @app.get("/gates")
    def list_gates(
        status: str | None = None, _: None = Depends(require_key)
    ) -> list[dict[str, Any]]:
        platform = get_platform()
        gates = platform.store.list_gates(status=status)
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

    @app.post("/gates")
    def request_gate(
        request: GateRequestBody, actor: Actor = Depends(require_person)
    ) -> dict[str, Any]:
        try:
            gate = governance.request_gate(
                get_platform(),
                actor,
                request.gate_type,
                title=request.title,
                subject=request.subject,
                evidence=request.evidence,
            )
        except REFUSALS as exc:
            raise http_error(exc) from exc
        return {"id": gate.id, "status": str(gate.status), "approvers": gate.required_approvers}

    @app.post("/gates/{gate_id}/decision")
    def decide_gate(
        gate_id: str, request: GateDecisionRequest, actor: Actor = Depends(require_person)
    ) -> dict[str, Any]:
        try:
            gate = governance.decide_gate(
                get_platform(), actor, gate_id, request.approved, note=request.note
            )
        except REFUSALS as exc:
            raise http_error(exc) from exc
        return {"id": gate.id, "status": str(gate.status)}

    @app.get("/audit")
    def audit(
        limit: int = 100,
        action: str | None = None,
        actor: str | None = None,
        _: None = Depends(require_key),
    ) -> list[dict[str, Any]]:
        platform = get_platform()
        events = platform.audit.recent(limit=limit, action=action, actor=actor)
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
        platform = get_platform()
        return platform.audit.verify().to_dict()

    @app.get("/budget")
    def budget(_: None = Depends(require_key)) -> dict[str, Any]:
        platform = get_platform()
        return {
            "status": [status.to_dict() for status in platform.budget.all_status()],
            "kill_switch": platform.budget.kill_switch_engaged(),
            "kill_switch_detail": governance.kill_switch_state(platform),
        }

    @app.post("/budget/limits")
    def set_limit(
        request: BudgetLimitRequest, actor: Actor = Depends(require_person)
    ) -> dict[str, Any]:
        try:
            return governance.set_budget_limit(
                get_platform(),
                actor,
                request.scope,
                request.key,
                request.limit_usd,
                gate_id=request.gate_id,
            )
        except REFUSALS as exc:
            raise http_error(exc) from exc

    @app.post("/kill-switch")
    def kill_switch(
        request: KillSwitchRequest, actor: Actor = Depends(require_person)
    ) -> dict[str, Any]:
        platform = get_platform()
        try:
            if request.engaged:
                return governance.engage_kill_switch(platform, actor, reason=request.reason)
            return governance.release_kill_switch(platform, actor, reason=request.reason)
        except REFUSALS as exc:
            raise http_error(exc) from exc

    @app.get("/policy")
    def policy(_: None = Depends(require_key)) -> dict[str, Any]:
        platform = get_platform()
        return platform.policy.status_summary()

    @app.post("/policy/{policy_id}")
    def decide_policy(
        policy_id: str, request: PolicyRequest, actor: Actor = Depends(require_person)
    ) -> dict[str, Any]:
        try:
            decision, gate_id = governance.propose_policy(
                get_platform(), actor, policy_id, request.value, rationale=request.rationale
            )
        except REFUSALS as exc:
            raise http_error(exc) from exc
        return {"id": decision.id, "status": decision.status, "gate_id": gate_id}

    @app.post("/policy/{policy_id}/confirm")
    def confirm_policy(
        policy_id: str, request: PolicyConfirmRequest, actor: Actor = Depends(require_person)
    ) -> dict[str, Any]:
        try:
            decision = governance.confirm_policy(
                get_platform(), actor, policy_id, gate_id=request.gate_id
            )
        except REFUSALS as exc:
            raise http_error(exc) from exc
        return {"id": decision.id, "status": decision.status, "value": decision.current_value}

    @app.get("/report")
    def report(_: None = Depends(require_key)) -> dict[str, Any]:
        platform = get_platform()
        status = platform.orchestrator.status_report()
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
        from internal_ai.alerts import AlertEngine

        platform = get_platform()
        engine = AlertEngine(platform.store, platform.audit, platform.budget)
        if check:
            engine.check_all()
        return [alert.to_dict() for alert in engine.list_alerts(unacknowledged_only=not all_alerts)]

    @app.post("/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(alert_id: str, actor: Actor = Depends(require_person)) -> dict[str, Any]:
        try:
            alert = governance.acknowledge_alert(get_platform(), actor, alert_id)
        except REFUSALS as exc:
            raise http_error(exc) from exc
        return alert.to_dict()

    @app.get("/roles")
    def roles(_: None = Depends(require_key)) -> list[dict[str, Any]]:
        from internal_ai.roles import all_roles

        return [role.to_dict() for role in all_roles()]

    @app.post("/tasks/{task_id}/transition")
    def transition(
        task_id: str,
        state: str,
        note: str = "manual transition",
        actor: Actor = Depends(require_person),
    ) -> dict[str, Any]:
        try:
            task = governance.transition_task(get_platform(), actor, task_id, state, note=note)
        except REFUSALS as exc:
            raise http_error(exc) from exc
        return {"id": task.id, "state": str(task.state)}

    from internal_ai.console.app import mount_console

    mount_console(app, get_platform, get_jobs, secure_cookies=secure_cookies)
    return app
