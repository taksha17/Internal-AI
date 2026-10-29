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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from internal_ai import brief
from internal_ai.console import find as finder
from internal_ai.console import inbox, lists
from internal_ai.context import Platform
from internal_ai.data.suggest import suggest_model_types
from internal_ai.domain import GateStatus, GateType, TaskState
from internal_ai.engine import PlanError
from internal_ai.evals import advice
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
from internal_ai.serving.service import out_of_service
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
    ("Brief", "/console/brief"),
    ("Inbox", "/console/inbox"),
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


@dataclass(slots=True)
class InspectionRun:
    token: str
    actor_id: str
    path: str
    percent: int = 0
    detail: str = "starting"
    done: bool = False
    result: build.SourceInspection | None = None
    error: str = ""


INSPECTIONS_KEPT = 32


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


def brief_from_form(form: dict[str, str]) -> brief.ProjectBrief:
    metrics = []
    for index in (1, 2, 3):
        name = (form.get(f"metric_name_{index}") or "").strip()
        threshold = (form.get(f"metric_threshold_{index}") or "").strip()
        if name or threshold:
            metrics.append(
                brief.Metric(
                    name=name,
                    threshold=threshold,
                    direction=form.get(f"metric_direction_{index}", "at_least"),
                )
            )
    return brief.ProjectBrief(
        target=(form.get("target") or "").strip(),
        languages=(form.get("languages") or "").strip(),
        context_length=_number(form.get("context_length", ""), int, 0),
        quality_bar=(form.get("quality_bar") or "").strip(),
        metrics=metrics,
        budget_usd=_number(form.get("budget_usd", ""), float, 0.0),
        timeline=(form.get("timeline") or "").strip(),
        deployment=(form.get("deployment") or "").strip(),
        data_restrictions=(form.get("data_restrictions") or "").strip(),
        latency_ms=_number(form.get("latency_ms", ""), int, 0),
        owners={role: (form.get(f"owner_{role}") or "").strip() for role in brief.OWNER_ROLES},
        non_goals=(form.get("non_goals") or "").strip(),
    )


def when(value: Any) -> Markup:
    """A time read at a glance, citable exactly.

    Rows carry ISO-8601 from the audit and job stores. A person scanning a list
    wants "4 min ago"; an auditor wants the exact instant. The element keeps
    both, and anything unparseable falls back to the value it was given.
    """
    if value is None or value == "":
        return Markup("")
    text = str(value)
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return Markup.escape(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    seconds = (datetime.now(UTC) - stamp).total_seconds()
    if seconds < 0:
        label = stamp.strftime("%d %b %Y")
    elif seconds < 60:
        label = "just now"
    elif seconds < 3600:
        label = f"{int(seconds // 60)} min ago"
    elif seconds < 86_400:
        label = f"{int(seconds // 3600)} h ago"
    elif seconds < 7 * 86_400:
        label = f"{int(seconds // 86_400)} d ago"
    else:
        label = stamp.strftime("%d %b %Y")
    return Markup('<time datetime="{}" title="{}">{}</time>').format(
        text, stamp.strftime("%d %b %Y, %H:%M UTC"), label
    )


def loss_points(path: str) -> Markup:
    """Two polylines, training solid and validation dashed, from a metrics file."""
    file = Path(path)
    if not path or not file.is_file():
        return Markup("")
    rows = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("loss"), int | float):
            rows.append(row)
    rows = rows[-24:]
    if len(rows) < 2:
        return Markup("")
    numbers = [
        float(row[key])
        for row in rows
        for key in ("loss", "validation_loss")
        if isinstance(row.get(key), int | float)
    ]
    if len(numbers) < 2:
        return Markup("")
    low, high = min(numbers), max(numbers)
    span = max(high - low, 1e-6)

    def series(key: str) -> str:
        coords = []
        for index, row in enumerate(rows):
            value = row.get(key)
            if not isinstance(value, int | float):
                continue
            x = 4 + (232 * index / max(len(rows) - 1, 1))
            y = 44 - (40 * (float(value) - low) / span)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    train = series("loss")
    held_out = series("validation_loss")
    markup = f'<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="{train}" />'
    if held_out:
        markup += (
            '<polyline fill="none" stroke="currentColor" stroke-width="1.5" '
            f'stroke-dasharray="3 2" points="{held_out}" />'
        )
    return Markup(markup)


def mount_console(
    app: FastAPI,
    get_platform: Callable[[], Platform],
    get_jobs: Callable[[], JobRunner],
    secure_cookies: bool = False,
) -> None:
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["nav"] = NAV
    templates.env.globals["min_password_length"] = MIN_PASSWORD_LENGTH
    templates.env.filters["when"] = when
    app.mount("/console/static", StaticFiles(directory=str(HERE / "static")), name="console-static")
    limiter = SignInLimiter()
    flashes: dict[str, list[tuple[str, str]]] = {}
    flash_lock = threading.Lock()
    inspections: dict[str, InspectionRun] = {}
    inspections_lock = threading.Lock()

    @app.exception_handler(SignInRequired)
    async def _to_signin(request: Request, exc: SignInRequired) -> Response:
        return RedirectResponse("/console/signin", status_code=303)

    @app.exception_handler(PasswordChangeRequired)
    async def _to_password(request: Request, exc: PasswordChangeRequired) -> Response:
        return RedirectResponse("/console/password", status_code=303)

    @app.exception_handler(StarletteHTTPException)
    async def _console_http_error(request: Request, exc: StarletteHTTPException) -> Response:
        """A wrong id in the console is a page, not a JSON body.

        A polled fragment keeps the plain error: htmx swaps whatever comes back
        into a small target, so a full page there would corrupt the page.
        """
        path = request.url.path
        if exc.status_code not in (403, 404) or not path.startswith("/console"):
            return await http_exception_handler(request, exc)
        if path.startswith("/console/static"):
            return await http_exception_handler(request, exc)
        if request.headers.get("hx-request"):
            return await http_exception_handler(request, exc)
        try:
            ctx = context(request)
        except (SignInRequired, PasswordChangeRequired):
            return RedirectResponse("/console/signin", status_code=303)
        return render(
            ctx,
            "error.html",
            status_code=exc.status_code,
            code=exc.status_code,
            detail=str(exc.detail),
        )

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
        """Whether this form post came from this host.

        A browser can send Origin: null for a same-site form. The console
        cookies are SameSite=Strict, so another site cannot attach one, and a
        matching cookie is what proves the post. A named foreign origin is
        still refused.
        """
        if request.headers.get("sec-fetch-site") == "cross-site":
            return False
        origin = request.headers.get("origin")
        if not origin or origin == "null":
            return True
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
        waiting = inbox.collect(platform, ctx.actor, get_jobs())
        values = {
            "actor": ctx.actor,
            "csrf": ctx.session.csrf_token,
            "kill": governance.kill_switch_state(platform),
            "flashes": pop_flashes(ctx),
            "path": ctx.request.url.path,
            "brief": brief.load(platform.store),
            "programme_gaps": brief.programme_gaps(platform, release=True),
            "waiting": waiting,
            "inbox": len(waiting.yours),
            "attention": waiting.count,
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
        jobs = get_jobs().list(limit=50)
        return render(
            ctx,
            "home.html",
            steps=steps,
            current=current,
            steps_done=sum(1 for _label, _href, done in steps if done),
            step_total=len(steps),
            active_jobs=sum(1 for job in jobs if job["state"] in {"running", "queued"}),
            budget=platform.budget.all_status(),
            jobs=jobs,
            alerts=alerts,
        )

    @app.get("/console/brief", include_in_schema=False)
    def brief_page(request: Request) -> Response:
        ctx = context(request)
        return render(
            ctx,
            "brief.html",
            form=brief.load(ctx.platform.store),
            roster=PeopleRegistry(ctx.platform.store).list_people(),
            roles=brief.OWNER_ROLES,
        )

    @app.post("/console/brief", include_in_schema=False)
    async def brief_save(request: Request) -> Response:
        ctx = context(request)
        payload = await posted(ctx)
        try:
            parsed = brief_from_form(payload)
        except ValueError as exc:
            flash(ctx, "error", message(exc))
            return redirect("/console/brief")
        return await act(
            ctx,
            "/console/brief",
            lambda: brief.save(ctx.platform.store, ctx.platform.audit, ctx.actor, parsed),
            lambda saved: (
                "brief saved"
                if saved.complete
                else "brief saved; still missing " + ", ".join(saved.missing)
            ),
        )

    @app.get("/console/inbox", include_in_schema=False)
    def inbox_page(request: Request) -> Response:
        ctx = context(request)
        return render(ctx, "inbox.html")

    @app.get("/console/search", include_in_schema=False)
    def search_page(request: Request, q: str = "") -> Response:
        ctx = context(request)
        return render(
            ctx,
            "search.html",
            q=q,
            term=q.strip(),
            groups=finder.search(ctx.platform, get_jobs(), q),
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
        form = await posted(ctx)
        back = form.get("back", "/console/home")
        if not back.startswith("/console/"):
            back = "/console/home"
        return await act(
            ctx,
            back,
            lambda: governance.acknowledge_alert(ctx.platform, ctx.actor, alert_id),
            "alert acknowledged",
        )

    # ----------------------------------------------------------- approvals

    @app.get("/console/approvals", include_in_schema=False)
    def approvals(request: Request, status: str = "pending", q: str = "") -> Response:
        ctx = context(request)
        chosen = status if status in ("pending", "approved", "rejected", "all") else "pending"
        gates = ctx.platform.store.list_gates(status=None if chosen == "all" else chosen)
        views = [gate_view(ctx.platform, ctx.actor, gate) for gate in gates]
        if q:
            views = [
                view
                for view in views
                if lists.matches(
                    q,
                    view["title"],
                    view["subject"],
                    view["id"],
                    view["gate_type"],
                    view["requested_by"],
                )
            ]
        views.sort(key=lambda item: (not item["can_decide"], item["created_at"]), reverse=False)
        return render(ctx, "approvals.html", gates=views, status=chosen, q=q)

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
        request: Request,
        actor: str = "",
        action: str = "",
        window: str = "",
        dir: str = "",
        page: int = 1,
    ) -> Response:
        ctx = context(request)
        per = lists.AUDIT_PER_PAGE
        direction = "asc" if dir == "asc" else "desc"
        cut = lists.since(window)
        events = ctx.platform.audit.recent(
            limit=per,
            offset=(max(page, 1) - 1) * per,
            action=action or None,
            actor=actor or None,
            since=cut,
            descending=direction == "desc",
        )
        table = lists.window(
            events,
            total=ctx.platform.store.count_audit(
                actor=actor or None, action=action or None, since=cut
            ),
            path="/console/audit",
            sort="seq",
            direction=direction,
            page=page,
            per_page=per,
            filters={"actor": actor, "action": action, "window": window},
        )
        return render(
            ctx,
            "audit.html",
            table=table,
            events=table.rows,
            verify=ctx.platform.audit.verify().to_dict(),
            actor_filter=actor,
            action_filter=action,
            window=window,
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

    def start_inspection(ctx: Ctx, raw_path: str) -> InspectionRun:
        """Scan in a worker so the page can show a real percentage of the walk."""
        path = raw_path.strip()
        with inspections_lock:
            for run in inspections.values():
                if run.actor_id == ctx.actor.id and run.path == path and not run.done:
                    return run
            run = InspectionRun(secrets.token_urlsafe(12), ctx.actor.id, path)
            inspections[run.token] = run
            finished = [key for key, value in inspections.items() if value.done]
            for key in finished[: max(0, len(inspections) - INSPECTIONS_KEPT)]:
                inspections.pop(key, None)

        def progress(percent: int, detail: str) -> None:
            with inspections_lock:
                run.percent, run.detail = percent, detail

        def work() -> None:
            try:
                result = build.inspect_source(ctx.platform, path, progress=progress)
            except Exception as exc:
                with inspections_lock:
                    run.error, run.percent, run.detail, run.done = str(exc), 100, "failed", True
                return
            with inspections_lock:
                run.result, run.percent, run.detail, run.done = result, 100, "done", True

        threading.Thread(target=work, name=f"inspect-{run.token}", daemon=True).start()
        return run

    def suggestions_for(inspection: Any) -> list[dict[str, Any]]:
        """Which model to build from a scan, and nothing when it was refused.

        A source that failed inspection has a refusal to read first, and a
        recommendation on top of it would bury the reason it cannot be used.
        """
        if inspection is None or not inspection.acceptable:
            return []
        return suggest_model_types(inspection.to_dict())

    def guided_suggestions(platform: Platform, source_id: str) -> list[dict[str, Any]]:
        """The same ranking on a later request, from what the source row stored.

        The in-memory scan does not survive a restart, so this reads the copy
        saved on the source and adds the dataset's record count once built.
        """
        source = build.get_source(platform, source_id)
        if source is None:
            return []
        summary = dict(source.get("scan") or {})
        if source.get("dataset_id"):
            dataset = platform.store.get_dataset(source["dataset_id"])
            if dataset is not None:
                summary["records"] = dataset.record_count
        return suggest_model_types(summary)

    def inspection_panel(ctx: Ctx, run: InspectionRun, clear_result: bool = False) -> HTMLResponse:
        return templates.TemplateResponse(
            ctx.request,
            "_inspect_live.html",
            {
                "csrf": ctx.session.csrf_token,
                "inspect_token": run.token,
                "inspect_running": not run.done,
                "inspect_percent": run.percent,
                "inspect_detail": run.detail,
                "inspection": run.result,
                "suggestions": suggestions_for(run.result),
                "inspect_error": run.error,
                "inspect_clear_result": clear_result and not run.done,
                "can": lambda action: allowed(ctx.actor, Action(action)),
            },
        )

    def intent_tail(
        platform: Platform, intent: str, dataset: Any, released: bool
    ) -> list[dict[str, Any]]:
        """Where each intent goes after release, read from rows that already exist.

        Search ends at a question anyone may ask. Decide ends at a scored
        classifier and names the encoder it sits on, because a person told only
        "train a classifier" would think the tool had broken when it asked for
        one. Generate ends at promotion behind the gate it already has.
        """
        if intent == "search":
            return [
                {
                    "label": "Askable",
                    "done": released,
                    "href": "/console/ask" if released else "",
                    "action": Action.ASK,
                }
            ]
        dataset_id = dataset.id if dataset is not None else ""
        tokenised = bool(dataset and build.tokenizer_path(platform, dataset_id) is not None)
        steps = [
            {
                "label": "Tokenizer trained",
                "done": tokenised,
                "href": f"/console/datasets/{dataset_id}" if dataset else "",
                "action": Action.BUILD_DATASET,
            }
        ]
        if intent == "decide":
            runs = [
                job
                for job in get_jobs().list(kind="train_classifier", limit=200)
                if dataset_id
                and job["state"] == "succeeded"
                and job["result"].get("dataset_id") == dataset_id
            ]
            run_ids = {job["result"].get("run_id") for job in runs}
            scored = [
                job
                for job in get_jobs().list(kind="evaluate_classifier", limit=200)
                if run_ids
                and job["state"] == "succeeded"
                and job["result"].get("run_id") in run_ids
            ]
            encoders = [
                model
                for model in build.classifier_encoders(platform)
                if not dataset_id or dataset_id in model.dataset_ids
            ]
            steps.extend(
                [
                    {
                        "label": "Encoder trained on this data",
                        "done": bool(encoders),
                        "href": "/console/models",
                        "action": Action.TRAIN_MODEL,
                    },
                    {
                        "label": "Classifier trained",
                        "done": bool(runs),
                        "href": "/console/models",
                        "action": Action.TRAIN_MODEL,
                    },
                    {
                        "label": "Scored on a held-out split",
                        "done": bool(scored),
                        "href": "/console/models",
                        "action": Action.EVALUATE_MODEL,
                    },
                ]
            )
            return steps
        models = [
            model
            for model in platform.store.list_models()
            if not dataset_id or dataset_id in model.dataset_ids
        ]
        report = build.latest_report(platform, models[-1].id) if models else None
        steps.extend(
            [
                {
                    "label": "Model trained and registered",
                    "done": bool(models),
                    "href": "/console/models",
                    "action": Action.TRAIN_MODEL,
                },
                {
                    "label": "Evaluated on a frozen set",
                    "done": report is not None,
                    "href": "/console/evals",
                    "action": Action.EVALUATE_MODEL,
                },
                {
                    "label": "Promoted behind a gate",
                    "done": any(model.status == "promoted" for model in models),
                    "href": "/console/models",
                    "action": Action.PROMOTE_MODEL,
                },
            ]
        )
        return steps

    def guided_steps(
        platform: Platform, actor: Actor, source_id: str, intent: str = ""
    ) -> dict[str, Any] | None:
        """The chain this intent needs, read from gates, jobs and models.

        Steps are a view over rows that already exist, never a new machine. The
        intent is not stored either: it picks which chain is shown, and the one
        thing written down is the purpose at registration, so an approver reads
        what is really being built. Each step carries the action it takes, so a
        person without that role sees who has it instead of a dead end.
        """
        source = build.get_source(platform, source_id)
        if source is None:
            return None
        ranking = guided_suggestions(platform, source_id)
        recommended = ranking[0]["kind"] if ranking else "search"
        chosen = build.guided_intent(intent or recommended)
        dataset = None
        if source.get("dataset_id"):
            dataset = platform.store.get_dataset(source["dataset_id"])
        released = bool(dataset and dataset.released)
        steps = [
            {"label": "Inspected", "done": True},
            {
                "label": "Source approved",
                "done": source["gate_status"] == "approved",
                "href": "/console/approvals?status=all",
            },
            {
                "label": "Dataset built",
                "done": dataset is not None,
                "action": Action.BUILD_DATASET,
            },
            {
                "label": "Split",
                "done": bool(dataset and "splits" in dataset.stats),
                "action": Action.BUILD_DATASET,
            },
            {
                "label": "Released",
                "done": released,
                "action": Action.RELEASE_DATASET,
            },
        ]
        steps.extend(intent_tail(platform, chosen, dataset, released))
        for step in steps:
            action = step.get("action")
            step["needs"] = reason(action) if action else ""
            step["you_can"] = allowed(actor, action) if action else True
        labels = dict(build.GUIDED_INTENTS)
        return {
            "intent": chosen,
            "intent_label": labels.get(chosen, chosen),
            "recommended": recommended,
            "recommended_label": labels.get(recommended, recommended),
            "alternatives": [
                {"kind": kind, "label": label}
                for kind, label in build.GUIDED_INTENTS
                if kind != chosen
            ],
            "options": ranking,
            "steps": steps,
            "done": sum(1 for step in steps if step["done"]),
        }

    @app.get("/console/sources", include_in_schema=False)
    def sources_page(
        request: Request,
        inspect: str = "",
        guided: str = "",
        intent: str = "",
        q: str = "",
        state: str = "",
        sort: str = "",
        dir: str = "",
        page: int = 1,
    ) -> Response:
        ctx = context(request)
        run = start_inspection(ctx, inspect) if inspect.strip() else None
        plan = guided_steps(ctx.platform, ctx.actor, guided, intent) if guided.strip() else None
        found = [
            item
            for item in build.list_sources(ctx.platform)
            if (
                not q
                or lists.matches(
                    q,
                    item["name"],
                    item["id"],
                    item["path"],
                    item["purpose"],
                    *item["acl"],
                )
            )
            and (not state or ("withdrawn" if item["withdrawn"] else item["gate_status"]) == state)
        ]
        table = lists.build(
            found,
            path="/console/sources",
            sorters={
                "name": lambda item: item["name"].casefold(),
                "state": lambda item: (
                    "withdrawn" if item["withdrawn"] else item["gate_status"],
                    item["name"].casefold(),
                ),
            },
            default_sort="name",
            sort=sort,
            direction=dir,
            page=page,
            filters={"q": q, "state": state},
        )
        return render(
            ctx,
            "sources.html",
            sources=table.rows,
            table=table,
            q=q,
            state=state,
            roots=build.data_roots(ctx.platform),
            inspection=run.result if run else None,
            suggestions=suggestions_for(run.result) if run else [],
            guided_suggestions=plan["options"] if plan else [],
            inspect_path=inspect,
            inspect_token=run.token if run else "",
            inspect_running=bool(run and not run.done),
            inspect_percent=run.percent if run else 0,
            inspect_detail=run.detail if run else "",
            inspect_error=run.error if run else "",
            guided_source=guided.strip() or "",
            guided_steps=plan["steps"] if plan else None,
            guided_intent=plan["intent"] if plan else "",
            guided_intent_label=plan["intent_label"] if plan else "",
            guided_recommended=plan["recommended"] if plan else "",
            guided_recommended_label=plan["recommended_label"] if plan else "",
            guided_alternatives=plan["alternatives"] if plan else [],
            guided_done=plan["done"] if plan else 0,
            guided_kinds=build.GUIDED_INTENTS,
        )

    @app.get("/console/sources/inspect", include_in_schema=False)
    def source_inspect(request: Request, inspect: str = "") -> Response:
        ctx = context(request)
        if not inspect.strip():
            return HTMLResponse("")
        return inspection_panel(ctx, start_inspection(ctx, inspect), clear_result=True)

    @app.get("/console/sources/inspect/{token}", include_in_schema=False)
    def source_inspect_status(request: Request, token: str) -> Response:
        ctx = context(request)
        with inspections_lock:
            run = inspections.get(token)
        if run is None or run.actor_id != ctx.actor.id:
            raise HTTPException(status_code=404, detail="unknown inspection")
        return inspection_panel(ctx, run)

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

    @app.post("/console/guided", include_in_schema=False)
    async def guided_register(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        path = form.get("path", "")
        intent = build.guided_intent(form.get("intent", ""))
        return await act(
            ctx,
            "/console/sources",
            lambda: build.guided_register(ctx.platform, ctx.actor, path, intent),
            lambda source: (
                f"guided run started for {source['name']}; "
                f"gate {source['gate_id']} needs approval before the build"
            ),
            lambda source: (
                f"/console/sources?guided={source['id']}&inspect={source['path']}&intent={intent}"
            ),
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
                legal_status=form.get("legal_status", ""),
                license_name=form.get("license_name", ""),
                license_url=form.get("license_url", ""),
                license_note=form.get("license_note", ""),
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
    def datasets_page(
        request: Request,
        q: str = "",
        state: str = "",
        sort: str = "",
        dir: str = "",
        page: int = 1,
    ) -> Response:
        ctx = context(request)
        rows = [
            {
                "dataset": item,
                "tokenizer": build.tokenizer_path(ctx.platform, item.id) is not None,
                "has_splits": "splits" in item.stats,
            }
            for item in ctx.platform.store.list_datasets()
            if (not q or lists.matches(q, item.name, item.id))
            and (state != "released" or item.released)
            and (state != "draft" or not item.released)
        ]
        table = lists.build(
            rows,
            path="/console/datasets",
            sorters={
                "name": lambda row: (row["dataset"].name.casefold(), row["dataset"].version),
                "records": lambda row: row["dataset"].record_count,
                "released": lambda row: (row["dataset"].released, row["dataset"].name.casefold()),
            },
            default_sort="name",
            sort=sort,
            direction=dir,
            page=page,
            filters={"q": q, "state": state},
            first_dirs={"records": "desc"},
        )
        return render(ctx, "datasets.html", rows=table.rows, table=table, q=q, state=state)

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
        "dedup",
        lambda ctx, dataset_id, form: build.drop_duplicate_copies(
            ctx.platform, ctx.actor, dataset_id
        ),
        lambda created: (
            f"version {created.version} keeps {created.record_count} records; "
            "the previous version was not changed and still needs its own release"
        ),
        lambda created: f"/console/datasets/{created.id}",
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
    def evals_page(
        request: Request,
        q: str = "",
        state: str = "",
        sort: str = "",
        dir: str = "",
        page: int = 1,
    ) -> Response:
        ctx = context(request)
        rows = []
        for eval_set in ctx.platform.store.list_eval_sets():
            if q and not lists.matches(q, eval_set.name, eval_set.id):
                continue
            if state == "frozen" and not eval_set.frozen:
                continue
            if state == "open" and eval_set.frozen:
                continue
            gate = build.gate_for(ctx.platform, GateType.EVAL_SET_FREEZE, eval_set.id)
            rows.append(
                {
                    "set": eval_set,
                    "leakage": build.eval_set_leakage(ctx.platform, eval_set.id),
                    "gate": gate_view(ctx.platform, ctx.actor, gate) if gate else None,
                }
            )
        table = lists.build(
            rows,
            path="/console/evals",
            sorters={
                "name": lambda row: (row["set"].name.casefold(), row["set"].version),
                "frozen": lambda row: (row["set"].frozen, row["set"].name.casefold()),
            },
            default_sort="name",
            sort=sort,
            direction=dir,
            page=page,
            filters={"q": q, "state": state},
        )
        return render(ctx, "evals.html", rows=table.rows, table=table, q=q, state=state)

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
    def models_page(
        request: Request,
        q: str = "",
        state: str = "",
        sort: str = "",
        dir: str = "",
        page: int = 1,
    ) -> Response:
        ctx = context(request)
        platform = ctx.platform
        registered_runs = {model.card.get("run_id") for model in platform.store.list_models()}
        runs = [
            {
                **job,
                "registered": job["result"].get("run_id") in registered_runs,
                "losses": loss_points(str(job["result"].get("metrics_path") or "")),
            }
            for job in get_jobs().list(limit=50, kind="train_model")
        ]
        models = []
        for model in platform.store.list_models():
            if q and not lists.matches(q, model.name, model.id, model.status):
                continue
            if state and str(model.status) != state:
                continue
            gate = build.gate_for(platform, GateType.MODEL_PROMOTION, model.id)
            report = build.latest_report(platform, model.id)
            first_dataset = model.dataset_ids[0] if model.dataset_ids else ""
            dataset = platform.store.get_dataset(first_dataset) if first_dataset else None
            models.append(
                {
                    "model": model,
                    "report": report,
                    "advice": advice.retrieval_advice(
                        report,
                        dataset.record_count if dataset else 0,
                        dataset.token_count if dataset else 0,
                    ),
                    "service": out_of_service(platform.store, model),
                    "regressed": build.recall_drop(platform, model),
                    "legacy_permission": bool(
                        report
                        and (report.checks.get("permission_leak") or {}).get("note")
                        == "permission enforcement is verified by the retrieval service tests"
                    ),
                    "builder": str(model.card.get("registered_by") or ""),
                    "judge": report.judged_by if report else "",
                    "self_judging": bool(report and report.judged_by)
                    and report.judged_by == str(model.card.get("registered_by") or ""),
                    "progress": brief.progress_rows(
                        brief.load(platform.store),
                        platform.store.list_eval_reports(model.id),
                    ),
                    "gate": gate_view(platform, ctx.actor, gate) if gate else None,
                }
            )
        datasets = [item for item in platform.store.list_datasets() if item.released]
        evaluated = {
            job["params"].get("run_id"): job["result"]
            for job in get_jobs().list(limit=50, kind="evaluate_classifier")
            if job["state"] == "succeeded" and job["params"].get("run_id")
        }
        classifiers = []
        for job in get_jobs().list(limit=50, kind="train_classifier"):
            report = evaluated.get(job["result"].get("run_id"))
            classifiers.append(
                {"run": job, "report": report, "advice": advice.classifier_advice(report)}
            )
        table = lists.build(
            models,
            path="/console/models",
            sorters={
                "name": lambda row: (row["model"].name.casefold(), row["model"].version),
                "status": lambda row: (str(row["model"].status), row["model"].name.casefold()),
            },
            default_sort="name",
            sort=sort,
            direction=dir,
            page=page,
            filters={"q": q, "state": state},
        )
        return render(
            ctx,
            "models.html",
            datasets=[
                {
                    "dataset": item,
                    "tokenizer": build.tokenizer_path(platform, item.id) is not None,
                    "labels": (item.lineage or {}).get("labels") or {},
                }
                for item in datasets
            ],
            classifiers=classifiers,
            encoders=build.classifier_encoders(platform),
            runs=runs,
            models=table.rows,
            models_table=table,
            q=q,
            state=state,
            frozen_sets=[item for item in platform.store.list_eval_sets() if item.frozen],
            cost_gate_usd=build.TRAINING_COST_GATE_USD,
        )

    @app.post("/console/models/overfit", include_in_schema=False)
    async def overfit_probe(request: Request) -> Response:
        from internal_ai.training.overfit import overfit_check

        ctx = context(request)
        await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: overfit_check(ctx.platform, ctx.actor),
            lambda report: (
                f"overfit check {'passed' if report['passed'] else 'failed'}: "
                f"loss {report['first_loss']} to {report['final_loss']} in "
                f"{report['steps']} steps on {report['corpus']} lines"
            ),
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
                vocab_rationale=form.get("vocab_rationale", ""),
            ),
            "training started",
            job_link,
        )

    @app.post("/console/models/classifier", include_in_schema=False)
    async def classifier_train(request: Request) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.train_classifier(
                ctx.platform,
                ctx.actor,
                get_jobs(),
                form.get("dataset_id", ""),
                form.get("encoder_model_id", ""),
                name=form.get("name", "") or "router-v1",
                epochs=_number(form.get("epochs", ""), int, 40),
                cost_cap_usd=_number(form.get("cost_cap_usd", ""), float, 2.0),
            ),
            "classifier training started",
            job_link,
        )

    @app.post("/console/models/classifier/{job_id}/evaluate", include_in_schema=False)
    async def classifier_evaluate(request: Request, job_id: str) -> Response:
        ctx = context(request)
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.evaluate_classifier(
                ctx.platform,
                ctx.actor,
                get_jobs(),
                job_id,
                split=form.get("split", "") or "test",
            ),
            "classifier evaluation started",
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
        form = await posted(ctx)
        return await act(
            ctx,
            "/console/models",
            lambda: build.promote(ctx.platform, ctx.actor, model_id, form.get("note", "")),
            "model promoted",
        )

    # ----------------------------------------------------------------- jobs

    @app.get("/console/jobs", include_in_schema=False)
    def jobs_page(
        request: Request,
        state: str = "",
        q: str = "",
        dir: str = "",
        page: int = 1,
    ) -> Response:
        ctx = context(request)
        choices = ("queued", "running", "succeeded", "failed")
        chosen = state if state in choices else ""
        per = lists.PER_PAGE
        direction = "asc" if dir == "asc" else "desc"
        runner = get_jobs()
        rows = runner.list(
            limit=per,
            state=chosen or None,
            text=q or None,
            offset=(max(page, 1) - 1) * per,
            descending=direction == "desc",
        )
        table = lists.window(
            rows,
            total=runner.count(state=chosen or None, text=q or None),
            path="/console/jobs",
            sort="created_at",
            direction=direction,
            page=page,
            per_page=per,
            filters={"state": chosen, "q": q},
        )
        return render(
            ctx,
            "jobs.html",
            jobs=table.rows,
            table=table,
            states=list(choices),
            state=chosen,
            q=q,
        )

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

    def people_rows(
        ctx: Ctx,
        q: str = "",
        role: str = "",
        sort: str = "",
        direction: str = "",
        page: int = 1,
    ) -> lists.Table:
        people = [
            person
            for person in PeopleRegistry(ctx.platform.store).list_people()
            if (
                not q
                or lists.matches(
                    q,
                    person["name"],
                    person["id"],
                    person["title"],
                    *person["roles"],
                    *person["groups"],
                )
            )
            and (not role or role in person["roles"])
        ]
        return lists.build(
            people,
            path="/console/admin/people",
            sorters={"name": lambda person: person["name"].casefold()},
            default_sort="name",
            sort=sort,
            direction=direction,
            page=page,
            filters={"q": q, "role": role},
        )

    @app.get("/console/admin/people", include_in_schema=False)
    def people_page(
        request: Request,
        q: str = "",
        role: str = "",
        sort: str = "",
        dir: str = "",
        page: int = 1,
    ) -> Response:
        ctx = context(request)
        table = people_rows(ctx, q, role, sort, dir, page)
        accounts = {item["person_id"]: item for item in ctx.identity.list_accounts()}
        return render(
            ctx,
            "people.html",
            people=table.rows,
            accounts=accounts,
            roles=[item.name for item in all_roles()],
            one_time=None,
            table=table,
            q=q,
            role=role,
        )

    def one_time_page(ctx: Ctx, person_id: str, password: str) -> HTMLResponse:
        table = people_rows(ctx)
        accounts = {item["person_id"]: item for item in ctx.identity.list_accounts()}
        return render(
            ctx,
            "people.html",
            people=table.rows,
            accounts=accounts,
            roles=[item.name for item in all_roles()],
            one_time={"person": person_id, "password": password},
            table=table,
            q="",
            role="",
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
    if kind in ("train_model", "evaluate_model", "train_classifier", "evaluate_classifier"):
        return ("Go to models", "/console/models")
    return None
