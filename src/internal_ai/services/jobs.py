"""Long-running console steps as compute-plane jobs.

Dataset builds, tokenizer training, model training and evaluation can take
minutes, so the console submits them as jobs and the page polls their status.
Each job goes through `JobScheduler.submit`, so it gets the scheduler's kill
switch and budget check and its audit trail. The work itself runs in-process
on a background thread: the subprocess sandbox limits address space to 4 GB,
which is too small to load torch, and these steps run platform code rather
than untrusted agent code.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from internal_ai.budget import week_start
from internal_ai.compute.scheduler import Job, JobScheduler, JobSpec, JobState
from internal_ai.context import Platform
from internal_ai.domain import BudgetScope
from internal_ai.ids import new_id, now_iso
from internal_ai.sandbox import Sandbox
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import deny

Work = Callable[[], dict[str, Any]]


class InProcessBackend:
    """Runs a Python callable as the job body and keeps its result."""

    name = "in-process"

    def __init__(self, work: Work) -> None:
        self.work = work
        self.result: dict[str, Any] = {}

    def submit(self, job: Job, sandbox: Sandbox) -> dict[str, Any]:
        try:
            self.result = self.work()
        except Exception as exc:
            return {"exit_code": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
        return {"exit_code": 0, "stdout": "", "stderr": ""}


def check_spend(platform: Platform, actor: Actor, agent: str, cost_cap_usd: float) -> None:
    """Refuse a job whose cost cap does not fit in what is left of any budget."""
    decision = platform.budget.check(agent)
    if not decision.allowed:
        raise deny(platform.audit, actor, "start_job", agent, decision.reason)
    scopes = [
        (BudgetScope.GLOBAL, "global"),
        (BudgetScope.PROJECT, platform.budget.project),
        (BudgetScope.AGENT, agent),
        (BudgetScope.WEEKLY, week_start().isoformat()),
    ]
    for scope, key in scopes:
        status = platform.budget.status(scope, key)
        if status is not None and status.remaining_usd < cost_cap_usd:
            raise deny(
                platform.audit,
                actor,
                "start_job",
                agent,
                f"this run may spend up to {cost_cap_usd:.2f} USD but only "
                f"{max(status.remaining_usd, 0):.2f} USD is left in the {scope}:{key} budget",
            )


class JobRunner:
    def __init__(self, platform: Platform, synchronous: bool = False) -> None:
        self.platform = platform
        self.synchronous = synchronous
        self._threads: dict[str, threading.Thread] = {}

    def submit(
        self,
        actor: Actor,
        kind: str,
        title: str,
        params: dict[str, Any],
        work: Work,
        agent: str,
        cost_cap_usd: float = 0.0,
    ) -> dict[str, Any]:
        check_spend(self.platform, actor, agent, cost_cap_usd)
        backend = InProcessBackend(work)
        scheduler = JobScheduler(
            self.platform.audit,
            self.platform.budget,
            Sandbox(self.platform.settings.sandbox_dir),
            backend=backend,
        )
        spec = JobSpec(
            name=kind,
            command=["internal-ai", "console-job", kind],
            agent=agent,
            cost_cap_usd=max(cost_cap_usd, 0.0),
        )
        job = scheduler.submit(spec, actor=actor.id)
        if job.state == JobState.REJECTED:
            raise deny(self.platform.audit, actor, "start_job", kind, job.rejection_reason)
        job_id = new_id("cjob")
        self.platform.store.conn.execute(
            "INSERT INTO console_jobs (id, kind, title, requested_by, state, params, cost_cap_usd,"
            " scheduler_job_id, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                kind,
                title,
                actor.id,
                "queued",
                json.dumps(params, sort_keys=True, default=str),
                cost_cap_usd,
                job.id,
                now_iso(),
            ),
        )

        def execute() -> None:
            self._update(job_id, state="running", started_at=now_iso())
            finished = scheduler.run(job, actor=actor.id)
            if finished.state == JobState.SUCCEEDED:
                self._update(
                    job_id,
                    state="succeeded",
                    result=json.dumps(backend.result, sort_keys=True, default=str),
                    finished_at=now_iso(),
                )
            else:
                self._update(
                    job_id, state="failed", error=finished.stderr[-2000:], finished_at=now_iso()
                )

        if self.synchronous:
            execute()
        else:
            thread = threading.Thread(target=execute, name=f"console-{job_id}", daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return self.get(job_id) or {}

    def _update(self, job_id: str, **fields: Any) -> None:
        columns = ", ".join(f"{name} = ?" for name in fields)
        self.platform.store.conn.execute(
            f"UPDATE console_jobs SET {columns} WHERE id = ?", (*fields.values(), job_id)
        )

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.platform.store.conn.execute(
            "SELECT * FROM console_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return _job_dict(row) if row else None

    def list(
        self,
        limit: int = 50,
        kind: str | None = None,
        state: str | None = None,
        text: str | None = None,
        offset: int = 0,
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        clauses, values = _job_filters(kind, state, text)
        query = "SELECT * FROM console_jobs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        order = "DESC" if descending else "ASC"
        query += f" ORDER BY created_at {order}, id {order} LIMIT ? OFFSET ?"
        rows = self.platform.store.conn.execute(query, (*values, limit, max(offset, 0))).fetchall()
        return [_job_dict(row) for row in rows]

    def count(
        self, kind: str | None = None, state: str | None = None, text: str | None = None
    ) -> int:
        clauses, values = _job_filters(kind, state, text)
        query = "SELECT COUNT(*) AS total FROM console_jobs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        row = self.platform.store.conn.execute(query, values).fetchone()
        return int(row["total"])


def _job_filters(
    kind: str | None, state: str | None, text: str | None
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    values: list[Any] = []
    if kind:
        clauses.append("kind = ?")
        values.append(kind)
    if state:
        clauses.append("state = ?")
        values.append(state)
    if text:
        clauses.append("(title LIKE ? OR id LIKE ? OR requested_by LIKE ?)")
        like = f"%{text}%"
        values.extend([like, like, like])
    return clauses, values


def _job_dict(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["params"] = json.loads(item.get("params") or "{}")
    item["result"] = json.loads(item.get("result") or "{}")
    item["finished"] = item["state"] in ("succeeded", "failed")
    return item
