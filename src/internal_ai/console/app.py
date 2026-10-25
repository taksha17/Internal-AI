"""Routes for the browser console.

Every page needs a signed-in person, every form post needs the session's CSRF
token, and every action goes through the same service functions as the
command line, so the console can never do what a person's roles and the gates
do not allow.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from internal_ai.context import Platform
from internal_ai.domain import GateStatus, GateType, TaskState
from internal_ai.engine import PlanError
from internal_ai.gates import GATE_CATALOG, GateError
from internal_ai.people import PeopleRegistry
from internal_ai.roles import all_roles
from internal_ai.services import build, governance
from internal_ai.services.actor import Actor
from internal_ai.services.identity import (
    MIN_PASSWORD_LENGTH,
    AuthenticationError,
    IdentityService,
    Session,
)
from internal_ai.services.jobs import JobRunner
from internal_ai.services.permissions import Action, allowed, deny, reason
from internal_ai.store import ALLOWED_TRANSITIONS, TransitionError

HERE = Path(__file__).resolve().parent
SESSION_COOKIE = "iai_session"
SIGNIN_COOKIE = "iai_signin"
SIGNIN_ATTEMPTS = 10
SIGNIN_WINDOW_SECONDS = 300

REFUSALS = (
    AuthenticationError,
    PermissionError,
    GateError,
    TransitionError,
    PlanError,
    KeyError,
    ValueError,
)

NAV = [
    ("Home", "/console/home"),
    ("Approvals", "/console/approvals"),
    ("Data", "/console/sources"),
    ("Datasets", "/console/datasets"),
    ("Eval sets", "/console/evals"),
    ("Models", "/console/models"),
    ("Jobs", "/console/jobs"),
    ("Ask", "/console/ask"),
    ("Plan", "/console/plan"),
    ("Audit", "/console/audit"),
    ("Admin", "/console/admin/people"),
]


class SignInRequired(Exception):
    pass


class PasswordChangeRequired(Exception):
    pass


@dataclass(slots=True)
class Ctx:
    request: Request
    platform: Platform
    identity: IdentityService
    actor: Actor
    session: Session


class SignInLimiter:
    """Slows down password guessing from one client before the account lockout."""

    def __init__(
        self, attempts: int = SIGNIN_ATTEMPTS, window: float = SIGNIN_WINDOW_SECONDS
    ) -> None:
        self.attempts = attempts
        self.window = window
        self._seen: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            recent = [stamp for stamp in self._seen.get(client, []) if now - stamp < self.window]
            if len(recent) >= self.attempts:
                self._seen[client] = recent
                return False
            recent.append(now)
            self._seen[client] = recent
            return True


def message(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def gate_view(platform: Platform, actor: Actor, gate: Any) -> dict[str, Any]:
    ok, why = governance.can_decide(platform, actor, gate)
    return {
        "id": gate.id,
        "gate_type": str(gate.gate_type),
        "title": gate.title,
        "subject": gate.subject,
        "status": str(gate.status),
        "requested_by": gate.requested_by,
        "required_approvers": list(gate.required_approvers),
        "evidence": [(key, _pretty(value)) for key, value in sorted((gate.evidence or {}).items())],
        "created_at": gate.created_at,
        "decided_at": gate.decided_at,
        "decisions": platform.store.gate_decisions(gate.id),
        "can_decide": ok and gate.status == GateStatus.PENDING,
        "why_not": why,
    }


def _pretty(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, dict | list):
            return json.dumps(parsed, indent=2, sort_keys=True)
        return value
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _number(raw: str, kind: type, default: Any) -> Any:
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        return kind(raw)
    except ValueError as exc:
        raise ValueError(f"{raw!r} is not a number") from exc


def _groups(raw: str) -> list[str]:
    return sorted({item.strip() for item in (raw or "").split(",") if item.strip()})


def mount_console(
    app: FastAPI,
    get_platform: Callable[[], Platform],
    get_jobs: Callable[[], JobRunner],
    secure_cookies: bool = False,
) -> None:
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["nav"] = NAV
    templates.env.globals["min_password_length"] = MIN_PASSWORD_LENGTH
    app.mount("/console/static", StaticFiles(directory=str(HERE / "static")), name="console-static")
    limiter = SignInLimiter()
    flashes: dict[str, list[tuple[str, str]]] = {}
    flash_lock = threading.Lock()

    @app.exception_handler(SignInRequired)
    async def _to_signin(request: Request, exc: SignInRequired) -> Response:
        return RedirectResponse("/console/signin", status_code=303)

    @app.exception_handler(PasswordChangeRequired)
    async def _to_password(request: Request, exc: PasswordChangeRequired) -> Response:
        return RedirectResponse("/console/password", status_code=303)

    # ------------------------------------------------------------ plumbing

    def context(request: Request, allow_password_change: bool = False) -> Ctx:
        platform = get_platform()
        identity = IdentityService(platform)
        resumed = identity.resume_session(request.cookies.get(SESSION_COOKIE, ""))
        if resumed is None:
            raise SignInRequired()
        actor, session = resumed
        if session.must_change_password and not allow_password_change:
            raise PasswordChangeRequired()
        return Ctx(request, platform, identity, actor, session)

    def flash(ctx: Ctx, kind: str, text: str) -> None:
        with flash_lock:
            flashes.setdefault(ctx.session.token, []).append((kind, text))

    def pop_flashes(ctx: Ctx) -> list[tuple[str, str]]:
        with flash_lock:
            return flashes.pop(ctx.session.token, [])

    def same_origin(request: Request) -> bool:
        origin = request.headers.get("origin")
        if not origin or origin == "null":
            return origin is None
        return urlsplit(origin).netloc == request.url.netloc

    async def posted(ctx: Ctx) -> dict[str, str]:
        form = await ctx.request.form()
        sent = str(form.get("csrf_token", ""))
        if not same_origin(ctx.request) or not hmac.compare_digest(
            sent.encode(), ctx.session.csrf_token.encode()
        ):
            deny(
                ctx.platform.audit,
                ctx.actor,
                "console.post",
                ctx.request.url.path,
                "missing or wrong CSRF token, or a cross-site request",
            )
            raise HTTPException(
                status_code=403,
                detail="this form is stale or came from another site; reload the page and retry",
            )
        return {key: str(value) for key, value in form.items() if key != "csrf_token"}

    def render(ctx: Ctx, name: str, status_code: int = 200, **data: Any) -> HTMLResponse:
        platform = ctx.platform
        pending = [
            gate
            for gate in platform.store.list_gates(status=str(GateStatus.PENDING))
            if governance.can_decide(platform, ctx.actor, gate)[0]
        ]
        values = {
            "actor": ctx.actor,
            "csrf": ctx.session.csrf_token,
            "kill": governance.kill_switch_state(platform),
            "flashes": pop_flashes(ctx),
            "path": ctx.request.url.path,
            "inbox": len(pending),
            "can": lambda action: allowed(ctx.actor, Action(action)),
            "why": lambda action: reason(Action(action)),
        }
        values.update(data)
        return templates.TemplateResponse(ctx.request, name, values, status_code=status_code)

    def redirect(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    async def act(
        ctx: Ctx,
        back: str,
        work: Callable[[], Any],
        done: str | Callable[[Any], str],
        to: str | Callable[[Any], str] | None = None,
    ) -> RedirectResponse:
        try:
            result = await run_in_threadpool(work)
        except REFUSALS as exc:
            flash(ctx, "error", message(exc))
            return redirect(back)
        flash(ctx, "ok", done(result) if callable(done) else done)
        target = to(result) if callable(to) else to
        return redirect(target or back)

    def job_link(result: dict[str, Any]) -> str:
        return f"/console/jobs/{result['id']}"

    # ------------------------------------------------------------ sign-in

    def signin_page(request: Request, error: str = "", status_code: int = 200) -> HTMLResponse:
        token = secrets.token_urlsafe(24)
        response = templates.TemplateResponse(
            request,
            "signin.html",
            {"signin_csrf": token, "error": error, "actor": None, "flashes": []},
            status_code=status_code,
        )
        response.set_cookie(
            SIGNIN_COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=secure_cookies,
            path="/console",
            max_age=900,
        )
        return response

    @app.get("/console", include_in_schema=False)
    def console_root() -> Response:
        return redirect("/console/home")

    @app.get("/console/signin", include_in_schema=False)
    def signin_form(request: Request) -> Response:
        return signin_page(request)

    @app.post("/console/signin", include_in_schema=False)
    async def signin(request: Request) -> Response:
        form = await request.form()
        expected = request.cookies.get(SIGNIN_COOKIE, "")
        sent = str(form.get("csrf_token", ""))
        if (
            not expected
            or not same_origin(request)
            or not hmac.compare_digest(sent.encode(), expected.encode())
        ):
            return signin_page(request, "the sign-in form expired; try again", 403)
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            return signin_page(request, "too many sign-in attempts; wait a few minutes", 429)
        platform = get_platform()
        identity = IdentityService(platform)
        try:
            actor = await run_in_threadpool(
                identity.authenticate,
                str(form.get("person", "")),
                str(form.get("password", "")),
                client,
            )
        except AuthenticationError as exc:
            return signin_page(request, str(exc), 401)
        session = identity.start_session(actor)
        response = redirect(
            "/console/password" if session.must_change_password else "/console/home"
        )
        response.set_cookie(
            SESSION_COOKIE,
            session.token,
            httponly=True,
            samesite="strict",
            secure=secure_cookies,
            path="/console",
        )
        response.delete_cookie(SIGNIN_COOKIE, path="/console")
        return response

    @app.post("/console/signout", include_in_schema=False)
    async def signout(request: Request) -> Response:
        ctx = context(request, allow_password_change=True)
        await posted(ctx)
        ctx.identity.end_session(ctx.session.token)
        ctx.platform.audit.record(
            actor=ctx.actor.id,
            actor_kind="human",
            action="account.signed_out",
            subject=ctx.actor.id,
        )
        response = redirect("/console/signin")
        response.delete_cookie(SESSION_COOKIE, path="/console")
        return response

    @app.get("/console/password", include_in_schema=False)
    def password_form(request: Request) -> Response:
        ctx = context(request, allow_password_change=True)
        return render(ctx, "password.html", forced=ctx.session.must_change_password)

    @app.post("/console/password", include_in_schema=False)
    async def password_change(request: Request) -> Response:
        ctx = context(request, allow_password_change=True)
        form = await posted(ctx)
        if form.get("new_password", "") != form.get("confirm_password", ""):
            flash(ctx, "error", "the two new passwords do not match")
            return redirect("/console/password")
        try:
            await run_in_threadpool(
                ctx.identity.change_password,
                ctx.actor.id,
                form.get("current_password", ""),
                form.get("new_password", ""),
            )
        except REFUSALS as exc:
            flash(ctx, "error", message(exc))
            return redirect("/console/password")
        ctx.identity.end_all_sessions(ctx.actor.id)
        fresh = ctx.identity.start_session(ctx.actor)
        response = redirect("/console/home")
        response.set_cookie(
            SESSION_COOKIE,
            fresh.token,
            httponly=True,
            samesite="strict",
            secure=secure_cookies,
            path="/console",
        )
        with flash_lock:
            flashes.setdefault(fresh.token, []).append(("ok", "password changed"))
        return response

    # ---------------------------------------------------------------- home

    @app.get("/console/home", include_in_schema=False)
    def home(request: Request) -> Response:
        ctx = context(request)
        platform = ctx.platform
        sources = build.list_sources(platform)
        datasets = platform.store.list_datasets()
        eval_sets = platform.store.list_eval_sets()
        models = platform.store.list_models()
        steps = [
            (
                "Register a data source and get it approved",
                "/console/sources",
                any(item["gate_status"] == "approved" for item in sources),
            ),
            (
                "Build, split and release a dataset",
                "/console/datasets",
                any(item.released for item in datasets),
            ),
            (
                "Train a tokenizer",
                "/console/datasets",
                any(build.tokenizer_path(platform, item.id) for item in datasets),
            ),
            (
                "Create and freeze an eval set",
                "/console/evals",
                any(item.frozen for item in eval_sets),
            ),
            ("Train and register a model", "/console/models", bool(models)),
            (
                "Evaluate the model against the frozen set",
                "/console/models",
                bool(platform.store.list_eval_reports()),
            ),
            (
                "Promote the model through its gate",
                "/console/models",
                any(item.status == "promoted" for item in models),
            ),
        ]
        from internal_ai.alerts import AlertEngine

        alerts = AlertEngine(platform.store, platform.audit, platform.budget).list_alerts(
            unacknowledged_only=True
        )
        current = next((label for label, _href, done in steps if not done), "every step is done")
        return render(
            ctx,
            "home.html",
            steps=steps,
            current=current,
            budget=platform.budget.all_status(),
            jobs=get_jobs().list(limit=8),
            alerts=alerts,
        )

    @app.post("/console/kill-switch", include_in_schema=False)
    async def kill_switch(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        back = form.get("back", "/console/home")
        if not back.startswith("/console/"):
            back = "/console/home"
        reason = form.get("reason", "")
        if form.get("engage") == "1":
            return await act(
                ctx,
                back,
                lambda: governance.engage_kill_switch(ctx.platform, ctx.actor, reason),
                "kill switch engaged: no new jobs or spend until a second person releases it",
            )
        return await act(
            ctx,
            back,
            lambda: governance.release_kill_switch(ctx.platform, ctx.actor, reason),
            "kill switch released",
        )

    @app.post("/console/alerts/{alert_id}/acknowledge", include_in_schema=False)
    async def acknowledge(request: Request, alert_id: str) -> Response:
        ctx = context(request)
        await posted(ctx)
        return await act(
            ctx,
            "/console/home",
            lambda: governance.acknowledge_alert(ctx.platform, ctx.actor, alert_id),
            "alert acknowledged",
        )

    # ----------------------------------------------------------- approvals

    @app.get("/console/approvals", include_in_schema=False)
    def approvals(request: Request, status: str = "pending") -> Response:
        ctx = context(request)
        chosen = status if status in ("pending", "approved", "rejected", "all") else "pending"
        gates = ctx.platform.store.list_gates(status=None if chosen == "all" else chosen)
        views = [gate_view(ctx.platform, ctx.actor, gate) for gate in gates]
        views.sort(key=lambda item: (not item["can_decide"], item["created_at"]), reverse=False)
        return render(ctx, "approvals.html", gates=views, status=chosen)

    @app.post("/console/approvals/{gate_id}", include_in_schema=False)
    async def decide(request: Request, gate_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        approved = form.get("decision") == "approve"
        return await act(
            ctx,
            "/console/approvals",
            lambda: governance.decide_gate(
                ctx.platform, ctx.actor, gate_id, approved, note=form.get("note", "")
            ),
            lambda gate: (
                f"{gate.title}: you {'approved' if approved else 'rejected'} it; "
                f"the gate is now {gate.status}"
            ),
        )

    @app.get("/console/gates/new", include_in_schema=False)
    def gate_form(request: Request, gate_type: str = "", subject: str = "") -> Response:
        ctx = context(request)
        entry = None
        if gate_type:
            try:
                entry = GATE_CATALOG[GateType(gate_type)]
            except ValueError:
                entry = None
        return render(
            ctx,
            "gate_new.html",
            catalog=list(GATE_CATALOG.values()),
            entry=entry,
            subject=subject,
        )

    @app.post("/console/gates", include_in_schema=False)
    async def gate_request(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        gate_type = form.get("gate_type", "")
        evidence = {
            key.removeprefix("evidence_"): value
            for key, value in form.items()
            if key.startswith("evidence_")
        }
        return await act(
            ctx,
            f"/console/gates/new?gate_type={gate_type}&subject={form.get('subject', '')}",
            lambda: governance.request_gate(
                ctx.platform,
                ctx.actor,
                gate_type,
                title=form.get("title", "") or gate_type.replace("_", " "),
                subject=form.get("subject", ""),
                evidence=evidence,
            ),
            lambda gate: (
                f"gate {gate.id} requested; it needs " + " or ".join(gate.required_approvers)
            ),
            "/console/approvals",
        )

    # ---------------------------------------------------------------- audit

    @app.get("/console/audit", include_in_schema=False)
    def audit_page(
        request: Request, actor: str = "", action: str = "", limit: int = 100
    ) -> Response:
        ctx = context(request)
        events = ctx.platform.audit.recent(
            limit=max(1, min(limit, 500)), action=action or None, actor=actor or None
        )
        return render(
            ctx,
            "audit.html",
            events=events,
            verify=ctx.platform.audit.verify().to_dict(),
            actor_filter=actor,
            action_filter=action,
        )

    # ----------------------------------------------------------------- plan

    @app.get("/console/plan", include_in_schema=False)
    def plan_page(request: Request) -> Response:
        ctx = context(request)
        tasks = ctx.platform.store.list_tasks()
        moves = {
            task.id: sorted(str(state) for state in ALLOWED_TRANSITIONS[TaskState(str(task.state))])
            for task in tasks
        }
        history = {task.id: ctx.platform.store.task_events(task.id) for task in tasks}
        return render(
            ctx,
            "plan.html",
            tasks=tasks,
            moves=moves,
            history=history,
            report=ctx.platform.orchestrator.status_report(),
        )

    @app.post("/console/plan", include_in_schema=False)
    async def plan_submit(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/plan",
            lambda: governance.submit_plan(
                ctx.platform,
                ctx.actor,
                form.get("goal", ""),
                milestone=form.get("milestone") or None,
            ),
            lambda tasks: f"plan submitted with {len(tasks)} tasks; approve it to start work",
        )

    @app.post("/console/plan/approve", include_in_schema=False)
    async def plan_approve(request: Request) -> Response:
        ctx = context(request)
        await posted(ctx)
        return await act(
            ctx,
            "/console/plan",
            lambda: governance.approve_plan(ctx.platform, ctx.actor),
            lambda count: f"{count} tasks approved",
        )

    @app.post("/console/plan/run", include_in_schema=False)
    async def plan_run(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        iterations = _number(form.get("max_iterations", ""), int, 20)
        return await act(
            ctx,
            "/console/plan",
            lambda: governance.run_plan(ctx.platform, ctx.actor, max_iterations=iterations),
            lambda summary: (
                "orchestrator ran: "
                + ", ".join(f"{key} {value}" for key, value in sorted(summary.items()))
            ),
        )

    @app.post("/console/tasks/{task_id}/transition", include_in_schema=False)
    async def task_transition(request: Request, task_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/plan",
            lambda: governance.transition_task(
                ctx.platform,
                ctx.actor,
                task_id,
                form.get("state", ""),
                note=form.get("note", "") or "console transition",
            ),
            lambda task: f"{task.title} is now {task.state}",
        )

    # -------------------------------------------------------------- sources

    @app.get("/console/sources", include_in_schema=False)
    def sources_page(request: Request, inspect: str = "") -> Response:
        ctx = context(request)
        inspection = build.inspect_source(ctx.platform, inspect) if inspect.strip() else None
        return render(
            ctx,
            "sources.html",
            sources=build.list_sources(ctx.platform),
            roots=build.data_roots(ctx.platform),
            inspection=inspection,
            inspect_path=inspect,
        )

    @app.post("/console/sources/sample", include_in_schema=False)
    async def source_sample(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        records = _number(form.get("records", ""), int, 120)
        return await act(
            ctx,
            "/console/sources",
            lambda: build.generate_sample(ctx.platform, ctx.actor, max(20, min(records, 5000))),
            lambda path: f"synthetic corpus written to {path}; inspect it below",
            lambda path: f"/console/sources?inspect={path}",
        )

    @app.post("/console/sources", include_in_schema=False)
    async def source_register(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/sources",
            lambda: build.register_source(
                ctx.platform,
                ctx.actor,
                form.get("path", ""),
                form.get("name", ""),
                form.get("purpose", ""),
                _groups(form.get("acl", "")),
            ),
            lambda source: (
                f"source {source['name']} registered; a data owner must approve "
                f"gate {source['gate_id']} before it can be used"
            ),
        )

    @app.post("/console/sources/{source_id}/build", include_in_schema=False)
    async def source_build(request: Request, source_id: str) -> Response:
        ctx = context(request)
        await posted(ctx)
        return await act(
            ctx,
            "/console/sources",
            lambda: build.build_dataset(ctx.platform, ctx.actor, get_jobs(), source_id),
            "dataset build started",
            job_link,
        )

    @app.post("/console/sources/{source_id}/withdraw", include_in_schema=False)
    async def source_withdraw(request: Request, source_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/sources",
            lambda: build.withdraw_source(
                ctx.platform, ctx.actor, source_id, form.get("reason", "")
            ),
            "source withdrawn; new dataset versions will exclude it",
        )

    @app.post("/console/sources/{source_id}/purge", include_in_schema=False)
    async def source_purge(request: Request, source_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        if form.get("confirm") == "1":
            return await act(
                ctx,
                "/console/sources",
                lambda: build.confirm_purge(ctx.platform, ctx.actor, source_id),
                lambda removed: f"purge confirmed; {removed} records erased",
            )
        return await act(
            ctx,
            "/console/sources",
            lambda: build.request_purge(ctx.platform, ctx.actor, source_id, form.get("reason", "")),
            "purge requested; a second person must confirm it",
        )

    # ------------------------------------------------------------- datasets

    @app.get("/console/datasets", include_in_schema=False)
    def datasets_page(request: Request) -> Response:
        ctx = context(request)
        datasets = ctx.platform.store.list_datasets()
        rows = [
            {
                "dataset": item,
                "tokenizer": build.tokenizer_path(ctx.platform, item.id) is not None,
                "has_splits": "splits" in item.stats,
            }
            for item in datasets
        ]
        return render(ctx, "datasets.html", rows=rows)

    @app.get("/console/datasets/{dataset_id}", include_in_schema=False)
    def dataset_page(request: Request, dataset_id: str) -> Response:
        ctx = context(request)
        try:
            view = build.dataset_view(ctx.platform, ctx.actor, dataset_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown dataset") from None
        release_gate = view["release_gate"]
        return render(
            ctx,
            "dataset.html",
            view=view,
            release_gate=gate_view(ctx.platform, ctx.actor, release_gate) if release_gate else None,
            tokenizer=build.tokenizer_path(ctx.platform, dataset_id),
            jobs=[
                job
                for job in get_jobs().list(limit=50)
                if job["params"].get("dataset_id") == dataset_id
            ],
        )

    def dataset_action(
        name: str,
        work: Callable[[Ctx, str, dict[str, str]], Any],
        done: str | Callable[[Any], str],
        to: Callable[[Any], str] | None = None,
    ) -> None:
        async def handler(request: Request, dataset_id: str) -> Response:
            ctx = context(request)
            form = await posted(ctx)
            return await act(
                ctx,
                f"/console/datasets/{dataset_id}",
                lambda: work(ctx, dataset_id, form),
                done,
                to,
            )

        app.add_api_route(
            f"/console/datasets/{{dataset_id}}/{name}",
            handler,
            methods=["POST"],
            include_in_schema=False,
            name=f"dataset_{name}",
        )

    dataset_action(
        "split",
        lambda ctx, dataset_id, form: build.split_dataset(ctx.platform, ctx.actor, dataset_id),
        lambda stats: (
            "split into "
            + ", ".join(f"{key} {value}" for key, value in sorted(stats.get("counts", {}).items()))
        ),
    )
    dataset_action(
        "request-release",
        lambda ctx, dataset_id, form: build.request_release(ctx.platform, ctx.actor, dataset_id),
        lambda gate: (
            f"release gate {gate.id} requested; it needs " + " or ".join(gate.required_approvers)
        ),
    )
    dataset_action(
        "release",
        lambda ctx, dataset_id, form: build.release_dataset(ctx.platform, ctx.actor, dataset_id),
        "dataset released; it is now immutable",
    )
    dataset_action(
        "tokenizer",
        lambda ctx, dataset_id, form: build.train_tokenizer(
            ctx.platform,
            ctx.actor,
            get_jobs(),
            dataset_id,
            vocab_size=_number(form.get("vocab_size", ""), int, 4000),
        ),
        "tokenizer training started",
        job_link,
    )
    dataset_action(
        "eval-set",
        lambda ctx, dataset_id, form: build.create_eval_set(
            ctx.platform,
            ctx.actor,
            dataset_id,
            form.get("name", "") or "support-eval",
            mode=form.get("mode", "paraphrase"),
            count=_number(form.get("count", ""), int, 24),
        ),
        lambda eval_set: (
            f"eval set {eval_set.name} v{eval_set.version} created with "
            f"{len(eval_set.items)} questions; check it and ask for it to be frozen"
        ),
        lambda eval_set: "/console/evals",
    )

    # ------------------------------------------------------------ eval sets

    @app.get("/console/evals", include_in_schema=False)
    def evals_page(request: Request) -> Response:
        ctx = context(request)
        rows = []
        for eval_set in ctx.platform.store.list_eval_sets():
            gate = build.gate_for(ctx.platform, GateType.EVAL_SET_FREEZE, eval_set.id)
            rows.append(
                {
                    "set": eval_set,
                    "leakage": build.eval_set_leakage(ctx.platform, eval_set.id),
                    "gate": gate_view(ctx.platform, ctx.actor, gate) if gate else None,
                }
            )
        return render(ctx, "evals.html", rows=rows)

    @app.post("/console/evals/{eval_set_id}/{step}", include_in_schema=False)
    async def eval_action(request: Request, eval_set_id: str, step: str) -> Response:
        ctx = context(request)
        await posted(ctx)
        actions: dict[str, tuple[Callable[[], Any], str | Callable[[Any], str]]] = {
            "request-freeze": (
                lambda: build.request_freeze(ctx.platform, ctx.actor, eval_set_id),
                lambda gate: (
                    f"freeze gate {gate.id} requested; it needs "
                    + " or ".join(gate.required_approvers)
                ),
            ),
            "freeze": (
                lambda: build.freeze_eval_set(ctx.platform, ctx.actor, eval_set_id),
                "eval set frozen; no one can change it now",
            ),
            "edit": (
                lambda: build.edit_eval_set(ctx.platform, ctx.actor, eval_set_id),
                "",
            ),
        }
        if step not in actions:
            raise HTTPException(status_code=404, detail="unknown step")
        work, done = actions[step]
        return await act(ctx, "/console/evals", work, done)

    # --------------------------------------------------------------- models

    @app.get("/console/models", include_in_schema=False)
    def models_page(request: Request) -> Response:
        ctx = context(request)
        platform = ctx.platform
        registered_runs = {model.card.get("run_id") for model in platform.store.list_models()}
        runs = [
            {**job, "registered": job["result"].get("run_id") in registered_runs}
            for job in get_jobs().list(limit=50, kind="train_model")
        ]
        models = []
        for model in platform.store.list_models():
            gate = build.gate_for(platform, GateType.MODEL_PROMOTION, model.id)
            models.append(
                {
                    "model": model,
                    "report": build.latest_report(platform, model.id),
                    "gate": gate_view(platform, ctx.actor, gate) if gate else None,
                }
            )
        datasets = [item for item in platform.store.list_datasets() if item.released]
        return render(
            ctx,
            "models.html",
            datasets=[
                {"dataset": item, "tokenizer": build.tokenizer_path(platform, item.id) is not None}
                for item in datasets
            ],
            runs=runs,
            models=models,
            frozen_sets=[item for item in platform.store.list_eval_sets() if item.frozen],
            cost_gate_usd=build.TRAINING_COST_GATE_USD,
        )

    @app.post("/console/models/train", include_in_schema=False)
    async def model_train(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.train_model(
                ctx.platform,
                ctx.actor,
                get_jobs(),
                form.get("dataset_id", ""),
                name=form.get("name", "") or "embedding-v1",
                steps=_number(form.get("steps", ""), int, 60),
                cost_cap_usd=_number(form.get("cost_cap_usd", ""), float, 2.0),
                pair_mode=form.get("pair_mode", "paraphrase"),
            ),
            "training started",
            job_link,
        )

    @app.post("/console/runs/{job_id}/register", include_in_schema=False)
    async def model_register(request: Request, job_id: str) -> Response:
        ctx = context(request)
        await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.register_model(ctx.platform, ctx.actor, get_jobs(), job_id),
            lambda model: f"registered {model.name} v{model.version} as {model.id}",
        )

    @app.post("/console/models/{model_id}/evaluate", include_in_schema=False)
    async def model_evaluate(request: Request, model_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.evaluate_model(
                ctx.platform,
                ctx.actor,
                get_jobs(),
                model_id,
                form.get("eval_set_id", ""),
                margin=_number(form.get("margin", ""), float, 0.0),
            ),
            "evaluation started",
            job_link,
        )

    @app.post("/console/models/{model_id}/request-promotion", include_in_schema=False)
    async def model_request_promotion(request: Request, model_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.request_promotion(
                ctx.platform, ctx.actor, model_id, form.get("red_team_results", "")
            ),
            lambda gate: (
                f"promotion gate {gate.id} requested; it needs "
                + " or ".join(gate.required_approvers)
            ),
        )

    @app.post("/console/models/{model_id}/promote", include_in_schema=False)
    async def model_promote(request: Request, model_id: str) -> Response:
        ctx = context(request)
        await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.promote(ctx.platform, ctx.actor, model_id),
            "model promoted",
        )

    # ----------------------------------------------------------------- jobs

    @app.get("/console/jobs", include_in_schema=False)
    def jobs_page(request: Request) -> Response:
        ctx = context(request)
        return render(ctx, "jobs.html", jobs=get_jobs().list(limit=100))

    @app.get("/console/jobs/{job_id}", include_in_schema=False)
    def job_page(request: Request, job_id: str) -> Response:
        ctx = context(request)
        job = get_jobs().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return render(ctx, "job.html", job=job, next_step=_next_step(job))

    @app.get("/console/jobs/{job_id}/status", include_in_schema=False)
    def job_status(request: Request, job_id: str) -> Response:
        ctx = context(request)
        job = get_jobs().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return render(ctx, "_job_status.html", job=job, next_step=_next_step(job))

    # ------------------------------------------------------------------ ask

    @app.get("/console/ask", include_in_schema=False)
    def ask_page(request: Request) -> Response:
        ctx = context(request)
        return render(
            ctx,
            "ask.html",
            datasets=[item for item in ctx.platform.store.list_datasets() if item.released],
            result=None,
            question="",
            chosen="",
        )

    @app.post("/console/ask", include_in_schema=False)
    async def ask_submit(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        question = form.get("question", "")
        dataset_id = form.get("dataset_id", "")
        result = None
        try:
            result = await run_in_threadpool(
                build.ask, ctx.platform, ctx.actor, dataset_id, question
            )
        except REFUSALS as exc:
            flash(ctx, "error", message(exc))
        return render(
            ctx,
            "ask.html",
            datasets=[item for item in ctx.platform.store.list_datasets() if item.released],
            result=result,
            question=question,
            chosen=dataset_id,
        )

    # ---------------------------------------------------------------- admin

    @app.get("/console/admin/people", include_in_schema=False)
    def people_page(request: Request) -> Response:
        ctx = context(request)
        accounts = {item["person_id"]: item for item in ctx.identity.list_accounts()}
        people = PeopleRegistry(ctx.platform.store).list_people()
        return render(
            ctx,
            "people.html",
            people=people,
            accounts=accounts,
            roles=[role.name for role in all_roles()],
            one_time=None,
        )

    def one_time_page(ctx: Ctx, person_id: str, password: str) -> HTMLResponse:
        accounts = {item["person_id"]: item for item in ctx.identity.list_accounts()}
        people = PeopleRegistry(ctx.platform.store).list_people()
        return render(
            ctx,
            "people.html",
            people=people,
            accounts=accounts,
            roles=[role.name for role in all_roles()],
            one_time={"person": person_id, "password": password},
        )

    @app.post("/console/admin/people", include_in_schema=False)
    async def people_add(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/admin/people",
            lambda: governance.add_person(
                ctx.platform,
                ctx.actor,
                form.get("name", "").strip(),
                form.get("title", "").strip(),
                _groups(form.get("roles", "")),
                _groups(form.get("groups", "")),
            ),
            lambda person: f"{person['name']} added",
        )

    @app.post("/console/admin/people/{person_id}/replace", include_in_schema=False)
    async def people_replace(request: Request, person_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/admin/people",
            lambda: governance.replace_person(
                ctx.platform,
                ctx.actor,
                person_id,
                form.get("name", "").strip(),
                form.get("title", "").strip(),
                _groups(form.get("roles", "")),
                _groups(form.get("groups", "")) or None,
            ),
            lambda person: f"{person_id} is now {person['name']}",
        )

    @app.post("/console/admin/people/{person_id}/account", include_in_schema=False)
    async def people_account(request: Request, person_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        step = form.get("step", "")
        try:
            if step == "create":
                password = await run_in_threadpool(
                    ctx.identity.create_account, person_id, ctx.actor
                )
                return one_time_page(ctx, person_id, password)
            if step == "reset":
                password = await run_in_threadpool(
                    ctx.identity.reset_password, person_id, ctx.actor
                )
                return one_time_page(ctx, person_id, password)
            if step in ("disable", "enable"):
                await run_in_threadpool(
                    ctx.identity.set_disabled, person_id, step == "disable", ctx.actor
                )
                flash(ctx, "ok", f"account {step}d")
                return redirect("/console/admin/people")
        except REFUSALS as exc:
            flash(ctx, "error", message(exc))
            return redirect("/console/admin/people")
        raise HTTPException(status_code=400, detail="unknown account step")

    @app.get("/console/account", include_in_schema=False)
    def account_page(request: Request) -> Response:
        ctx = context(request)
        return render(
            ctx, "account.html", tokens=ctx.identity.list_tokens(ctx.actor.id), new_token=None
        )

    @app.post("/console/account/tokens", include_in_schema=False)
    async def token_create(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        try:
            token = ctx.identity.create_token(ctx.actor, form.get("label", ""))
        except REFUSALS as exc:
            flash(ctx, "error", message(exc))
            return redirect("/console/account")
        return render(
            ctx, "account.html", tokens=ctx.identity.list_tokens(ctx.actor.id), new_token=token
        )

    @app.post("/console/account/tokens/{token_id}/revoke", include_in_schema=False)
    async def token_revoke(request: Request, token_id: str) -> Response:
        ctx = context(request)
        await posted(ctx)
        return await act(
            ctx,
            "/console/account",
            lambda: ctx.identity.revoke_token(token_id, ctx.actor),
            "token revoked",
        )

    @app.get("/console/admin/policy", include_in_schema=False)
    def policy_page(request: Request) -> Response:
        ctx = context(request)
        decisions = []
        for decision in ctx.platform.policy.list():
            gate_id = ctx.platform.store.kv_get(f"policy_gate:{decision.id}", "") or ""
            gate = ctx.platform.store.get_gate(gate_id) if gate_id else None
            decisions.append(
                {
                    "decision": decision,
                    "gate": gate_view(ctx.platform, ctx.actor, gate) if gate else None,
                }
            )
        return render(ctx, "policy.html", decisions=decisions)

    @app.post("/console/admin/policy/{policy_id}", include_in_schema=False)
    async def policy_post(request: Request, policy_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        if form.get("confirm") == "1":
            return await act(
                ctx,
                "/console/admin/policy",
                lambda: governance.confirm_policy(
                    ctx.platform,
                    ctx.actor,
                    policy_id,
                    gate_id=ctx.platform.store.kv_get(f"policy_gate:{policy_id}", "") or None,
                ),
                lambda decision: f"{policy_id} is now {decision.current_value}",
            )
        return await act(
            ctx,
            "/console/admin/policy",
            lambda: governance.propose_policy(
                ctx.platform,
                ctx.actor,
                policy_id,
                form.get("value", ""),
                rationale=form.get("rationale", ""),
            ),
            lambda pair: f"proposal recorded; gate {pair[1]} needs a second person",
        )

    @app.get("/console/admin/budget", include_in_schema=False)
    def budget_page(request: Request) -> Response:
        ctx = context(request)
        gates = [
            gate_view(ctx.platform, ctx.actor, gate)
            for gate in ctx.platform.store.list_gates()
            if str(gate.gate_type) == str(GateType.BUDGET_INCREASE)
        ]
        return render(ctx, "budget.html", budget=ctx.platform.budget.all_status(), gates=gates)

    @app.post("/console/admin/budget", include_in_schema=False)
    async def budget_set(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/admin/budget",
            lambda: governance.set_budget_limit(
                ctx.platform,
                ctx.actor,
                form.get("scope", ""),
                form.get("key", ""),
                _number(form.get("limit_usd", ""), float, -1.0),
                gate_id=form.get("gate_id") or None,
            ),
            lambda result: (
                f"{result['scope']}:{result['key']} limit set to ${result['limit_usd']:.2f}"
            ),
        )


def _next_step(job: dict[str, Any]) -> tuple[str, str] | None:
    if job["state"] != "succeeded":
        return None
    kind = job["kind"]
    result = job["result"]
    if kind == "build_dataset" and result.get("dataset_id"):
        return (
            "Open the dataset to split and release it",
            f"/console/datasets/{result['dataset_id']}",
        )
    if kind == "train_tokenizer":
        return ("Back to the dataset", f"/console/datasets/{job['params'].get('dataset_id', '')}")
    if kind in ("train_model", "evaluate_model"):
        return ("Go to models", "/console/models")
    return None
