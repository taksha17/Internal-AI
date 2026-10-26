"""Repositories over SQLite.

Task state transitions are enforced here so that the FR-2 lifecycle cannot be
skipped by any agent, human or CLI path.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from internal_ai.db import connect, init_db, transaction
from internal_ai.domain import (
    AuditEvent,
    BudgetLimit,
    BudgetScope,
    DatasetVersion,
    EvalReport,
    EvalSet,
    GateDecision,
    GateRequest,
    ModelVersion,
    PolicyDecision,
    RunSummary,
    SpendRecord,
    Task,
    TaskEvent,
    TaskState,
)
from internal_ai.ids import content_hash, new_id, now_iso

ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PROPOSED: frozenset(
        {TaskState.APPROVED, TaskState.REJECTED, TaskState.BLOCKED, TaskState.ESCALATED}
    ),
    TaskState.APPROVED: frozenset(
        {TaskState.IN_PROGRESS, TaskState.BLOCKED, TaskState.ESCALATED, TaskState.REJECTED}
    ),
    TaskState.IN_PROGRESS: frozenset(
        {TaskState.IN_REVIEW, TaskState.BLOCKED, TaskState.ESCALATED, TaskState.REJECTED}
    ),
    TaskState.IN_REVIEW: frozenset(
        {TaskState.ACCEPTED, TaskState.REJECTED, TaskState.IN_PROGRESS, TaskState.ESCALATED}
    ),
    TaskState.BLOCKED: frozenset(
        {TaskState.APPROVED, TaskState.IN_PROGRESS, TaskState.ESCALATED, TaskState.REJECTED}
    ),
    TaskState.ESCALATED: frozenset(
        {TaskState.APPROVED, TaskState.REJECTED, TaskState.BLOCKED, TaskState.IN_PROGRESS}
    ),
    TaskState.ACCEPTED: frozenset(),
    TaskState.REJECTED: frozenset(),
}


class TransitionError(RuntimeError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class Store:
    """Single entry point for all persistent state.

    Connections are thread-local so the store can be shared by the API's
    request threads, the orchestrator and background jobs.
    """

    def __init__(self, db_path: Any) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._connection()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            return existing
        conn = connect(self.db_path)
        init_db(conn)
        self._local.conn = conn
        with self._lock:
            self._connections.append(conn)
        return conn

    def backup_to(self, dest: Path) -> None:
        """Copy the live database with SQLite's online backup.

        A second connection reads a consistent snapshot while this process
        keeps the file open.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(dest)
        try:
            self.conn.backup(target)
        finally:
            target.close()

    def close(self) -> None:
        with self._lock:
            for conn in self._connections:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._connections.clear()
        self._local = threading.local()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- tasks

    def add_task(self, task: Task) -> Task:
        stamp = now_iso()
        task.created_at = task.created_at or stamp
        task.updated_at = stamp
        task.last_state_change = task.last_state_change or stamp
        self.conn.execute(
            """
            INSERT INTO tasks (
                id, title, description, kind, owner_role, state, dependencies,
                acceptance_criteria, cost_estimate_usd, cost_cap_usd, actual_cost_usd,
                attempts, review_attempts, artifacts, reviewer_role, gate_id, milestone,
                blocked_reason, created_at, updated_at, started_at, completed_at,
                last_state_change, metadata
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                task.id,
                task.title,
                task.description,
                str(task.kind),
                task.owner_role,
                str(task.state),
                _json(task.dependencies),
                _json(task.acceptance_criteria),
                task.cost_estimate_usd,
                task.cost_cap_usd,
                task.actual_cost_usd,
                task.attempts,
                task.review_attempts,
                _json(task.artifacts),
                task.reviewer_role,
                task.gate_id,
                task.milestone,
                task.blocked_reason,
                task.created_at,
                task.updated_at,
                task.started_at,
                task.completed_at,
                task.last_state_change,
                _json(task.metadata),
            ),
        )
        self.record_task_event(task, None, task.state, "system", "human", "task created")
        return task

    def get_task(self, task_id: str) -> Task | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(
        self,
        state: str | None = None,
        owner_role: str | None = None,
        kind: str | None = None,
    ) -> list[Task]:
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []
        if state:
            query += " AND state = ?"
            params.append(str(state))
        if owner_role:
            query += " AND owner_role = ?"
            params.append(owner_role)
        if kind:
            query += " AND kind = ?"
            params.append(str(kind))
        query += " ORDER BY created_at ASC, id ASC"
        return [self._row_to_task(row) for row in self.conn.execute(query, params).fetchall()]

    def update_task(self, task: Task) -> Task:
        task.updated_at = now_iso()
        self.conn.execute(
            """
            UPDATE tasks SET
                title=?, description=?, kind=?, owner_role=?, state=?, dependencies=?,
                acceptance_criteria=?, cost_estimate_usd=?, cost_cap_usd=?, actual_cost_usd=?,
                attempts=?, review_attempts=?, artifacts=?, reviewer_role=?, gate_id=?, milestone=?,
                blocked_reason=?, updated_at=?, started_at=?, completed_at=?, last_state_change=?,
                metadata=?
            WHERE id=?
            """,
            (
                task.title,
                task.description,
                str(task.kind),
                task.owner_role,
                str(task.state),
                _json(task.dependencies),
                _json(task.acceptance_criteria),
                task.cost_estimate_usd,
                task.cost_cap_usd,
                task.actual_cost_usd,
                task.attempts,
                task.review_attempts,
                _json(task.artifacts),
                task.reviewer_role,
                task.gate_id,
                task.milestone,
                task.blocked_reason,
                task.updated_at,
                task.started_at,
                task.completed_at,
                task.last_state_change,
                _json(task.metadata),
                task.id,
            ),
        )
        return task

    def set_task_state(
        self,
        task: Task,
        new_state: TaskState,
        actor: str,
        actor_kind: str = "agent",
        note: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        target = TaskState(new_state)
        current = TaskState(task.state)
        if target != current and target not in ALLOWED_TRANSITIONS[current]:
            raise TransitionError(f"illegal transition {current} -> {target} for task {task.id}")
        stamp = now_iso()
        previous = str(current)
        task.state = target
        task.updated_at = stamp
        task.last_state_change = stamp
        if target == TaskState.IN_PROGRESS and task.started_at is None:
            task.started_at = stamp
        if target in (TaskState.ACCEPTED, TaskState.REJECTED):
            task.completed_at = stamp
        if target in (TaskState.BLOCKED, TaskState.ESCALATED) and note:
            task.blocked_reason = note
        self.update_task(task)
        self.record_task_event(task, previous, str(target), actor, actor_kind, note, metadata)
        return task

    def record_task_event(
        self,
        task: Task,
        from_state: str | None,
        to_state: str,
        actor: str,
        actor_kind: str,
        note: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskEvent:
        event = TaskEvent(
            id=new_id("evt"),
            task_id=task.id,
            from_state=from_state,
            to_state=to_state,
            actor=actor,
            actor_kind=actor_kind,
            note=note,
            created_at=now_iso(),
            metadata=metadata or {},
        )
        self.conn.execute(
            "INSERT INTO task_events (id, task_id, from_state, to_state, actor, actor_kind,"
            " note, created_at, metadata) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                event.id,
                event.task_id,
                event.from_state,
                event.to_state,
                event.actor,
                event.actor_kind,
                event.note,
                event.created_at,
                _json(event.metadata),
            ),
        )
        return event

    def task_events(self, task_id: str) -> list[TaskEvent]:
        rows = self.conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY created_at ASC, rowid ASC",
            (task_id,),
        ).fetchall()
        return [
            TaskEvent(
                id=row["id"],
                task_id=row["task_id"],
                from_state=row["from_state"],
                to_state=row["to_state"],
                actor=row["actor"],
                actor_kind=row["actor_kind"],
                note=row["note"],
                created_at=row["created_at"],
                metadata=_loads(row["metadata"], {}),
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            kind=row["kind"],
            owner_role=row["owner_role"],
            state=row["state"],
            dependencies=_loads(row["dependencies"], []),
            acceptance_criteria=_loads(row["acceptance_criteria"], []),
            cost_estimate_usd=row["cost_estimate_usd"],
            cost_cap_usd=row["cost_cap_usd"],
            actual_cost_usd=row["actual_cost_usd"],
            attempts=row["attempts"],
            review_attempts=row["review_attempts"],
            artifacts=_loads(row["artifacts"], []),
            reviewer_role=row["reviewer_role"],
            gate_id=row["gate_id"],
            milestone=row["milestone"],
            blocked_reason=row["blocked_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            last_state_change=row["last_state_change"],
            metadata=_loads(row["metadata"], {}),
        )

    # ----------------------------------------------------------------- gates

    def create_gate(self, gate: GateRequest) -> GateRequest:
        gate.created_at = gate.created_at or now_iso()
        self.conn.execute(
            """
            INSERT INTO gates (id, gate_type, title, subject, requested_by, requested_by_kind,
                               required_approvers, evidence, status, created_at, decided_at,
                               decision_note, cost_usd)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                gate.id,
                str(gate.gate_type),
                gate.title,
                gate.subject,
                gate.requested_by,
                gate.requested_by_kind,
                _json(list(gate.required_approvers)),
                _json(gate.evidence),
                str(gate.status),
                gate.created_at,
                gate.decided_at,
                gate.decision_note,
                gate.cost_usd,
            ),
        )
        return gate

    def get_gate(self, gate_id: str) -> GateRequest | None:
        row = self.conn.execute("SELECT * FROM gates WHERE id = ?", (gate_id,)).fetchone()
        return self._row_to_gate(row) if row else None

    def list_gates(self, status: str | None = None) -> list[GateRequest]:
        query = "SELECT * FROM gates"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(str(status))
        query += " ORDER BY created_at ASC, id ASC"
        return [self._row_to_gate(row) for row in self.conn.execute(query, params).fetchall()]

    def update_gate(self, gate: GateRequest) -> GateRequest:
        self.conn.execute(
            """
            UPDATE gates SET status=?, decided_at=?, decision_note=?, evidence=?
            WHERE id=?
            """,
            (
                str(gate.status),
                gate.decided_at,
                gate.decision_note,
                _json(gate.evidence),
                gate.id,
            ),
        )
        return gate

    def add_gate_decision(self, decision: GateDecision) -> GateDecision:
        self.conn.execute(
            "INSERT INTO gate_decisions (id, gate_id, approver, approver_kind, approved, note,"
            " created_at) VALUES (?,?,?,?,?,?,?)",
            (
                decision.id,
                decision.gate_id,
                decision.approver,
                decision.approver_kind,
                int(decision.approved),
                decision.note,
                decision.created_at,
            ),
        )
        return decision

    def gate_decisions(self, gate_id: str) -> list[GateDecision]:
        rows = self.conn.execute(
            "SELECT * FROM gate_decisions WHERE gate_id = ? ORDER BY created_at ASC",
            (gate_id,),
        ).fetchall()
        return [
            GateDecision(
                id=row["id"],
                gate_id=row["gate_id"],
                approver=row["approver"],
                approver_kind=row["approver_kind"],
                approved=bool(row["approved"]),
                note=row["note"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_gate(row: sqlite3.Row) -> GateRequest:
        return GateRequest(
            id=row["id"],
            gate_type=row["gate_type"],
            title=row["title"],
            subject=row["subject"],
            requested_by=row["requested_by"],
            requested_by_kind=row["requested_by_kind"],
            required_approvers=tuple(_loads(row["required_approvers"], [])),
            evidence=_loads(row["evidence"], {}),
            status=row["status"],
            created_at=row["created_at"],
            decided_at=row["decided_at"],
            decision_note=row["decision_note"],
            cost_usd=row["cost_usd"],
        )

    # ----------------------------------------------------------------- audit

    def append_audit(self, event: AuditEvent) -> AuditEvent:
        with transaction(self.conn):
            last = self.conn.execute(
                "SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = last["hash"] if last else ""
            event.prev_hash = prev_hash
            event.created_at = event.created_at or now_iso()
            event.hash = content_hash(
                {
                    "id": event.id,
                    "actor": event.actor,
                    "actor_kind": event.actor_kind,
                    "action": event.action,
                    "subject": event.subject,
                    "outcome": str(event.outcome),
                    "inputs": event.inputs,
                    "details": event.details,
                    "cost_usd": event.cost_usd,
                    "prev_hash": prev_hash,
                    "created_at": event.created_at,
                    "tenant": event.tenant,
                }
            )
            self.conn.execute(
                """
                INSERT INTO audit_events (id, actor, actor_kind, action, subject, outcome, inputs,
                                         details, cost_usd, prev_hash, hash, created_at, tenant)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.id,
                    event.actor,
                    event.actor_kind,
                    event.action,
                    event.subject,
                    str(event.outcome),
                    _json(event.inputs),
                    _json(event.details),
                    event.cost_usd,
                    event.prev_hash,
                    event.hash,
                    event.created_at,
                    event.tenant,
                ),
            )
        return event

    def list_audit(
        self,
        actor: str | None = None,
        action: str | None = None,
        subject: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        query = "SELECT * FROM audit_events WHERE 1=1"
        params: list[Any] = []
        if actor:
            query += " AND actor = ?"
            params.append(actor)
        if action:
            query += " AND action = ?"
            params.append(action)
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        query += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        return [
            AuditEvent(
                id=row["id"],
                seq=row["seq"],
                actor=row["actor"],
                actor_kind=row["actor_kind"],
                action=row["action"],
                subject=row["subject"],
                outcome=row["outcome"],
                inputs=_loads(row["inputs"], {}),
                details=_loads(row["details"], {}),
                cost_usd=row["cost_usd"],
                prev_hash=row["prev_hash"],
                hash=row["hash"],
                created_at=row["created_at"],
                tenant=row["tenant"],
            )
            for row in self.conn.execute(query, params).fetchall()
        ]

    def audit_chain(self) -> list[AuditEvent]:
        return [
            AuditEvent(
                id=row["id"],
                seq=row["seq"],
                actor=row["actor"],
                actor_kind=row["actor_kind"],
                action=row["action"],
                subject=row["subject"],
                outcome=row["outcome"],
                inputs=_loads(row["inputs"], {}),
                details=_loads(row["details"], {}),
                cost_usd=row["cost_usd"],
                prev_hash=row["prev_hash"],
                hash=row["hash"],
                created_at=row["created_at"],
                tenant=row["tenant"],
            )
            for row in self.conn.execute("SELECT * FROM audit_events ORDER BY seq ASC").fetchall()
        ]

    # --------------------------------------------------------------- budgets

    def set_budget_limit(self, limit: BudgetLimit) -> BudgetLimit:
        self.conn.execute(
            """
            INSERT INTO budget_limits (scope, key, limit_usd, period_start, period_end)
            VALUES (?,?,?,?,?)
            ON CONFLICT(scope, key) DO UPDATE SET
                limit_usd=excluded.limit_usd,
                period_start=excluded.period_start,
                period_end=excluded.period_end
            """,
            (str(limit.scope), limit.key, limit.limit_usd, limit.period_start, limit.period_end),
        )
        return limit

    def list_budget_limits(self) -> list[BudgetLimit]:
        return [
            BudgetLimit(
                scope=row["scope"],
                key=row["key"],
                limit_usd=row["limit_usd"],
                period_start=row["period_start"],
                period_end=row["period_end"],
            )
            for row in self.conn.execute("SELECT * FROM budget_limits").fetchall()
        ]

    def record_spend(self, spend: SpendRecord) -> SpendRecord:
        self.conn.execute(
            """
            INSERT INTO spend_records (id, scope, key, amount_usd, agent, task_id, job_id,
            created_at, notes) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                spend.id,
                str(spend.scope),
                spend.key,
                spend.amount_usd,
                spend.agent,
                spend.task_id,
                spend.job_id,
                spend.created_at,
                spend.notes,
            ),
        )
        return spend

    def spend_by(self, scope: BudgetScope, key: str) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(amount_usd), 0) AS total FROM spend_records "
            "WHERE scope = ? AND key = ?",
            (str(scope), key),
        ).fetchone()
        return float(row["total"])

    def spend_totals(self) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT scope, key, COALESCE(SUM(amount_usd), 0) AS total FROM spend_records "
            "GROUP BY scope, key"
        ).fetchall()
        return {f"{row['scope']}:{row['key']}": float(row["total"]) for row in rows}

    # ----------------------------------------------------------------- runs

    def record_run(self, run: RunSummary) -> RunSummary:
        self.conn.execute(
            """
            INSERT INTO runs (id, task_id, agent, status, output, artifacts, cost_usd, duration_s,
                              started_at, finished_at, error, metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                new_id("run"),
                run.task_id,
                run.agent,
                run.status,
                run.output,
                _json(run.artifacts),
                run.cost_usd,
                run.duration_s,
                run.started_at,
                run.finished_at,
                run.error,
                _json(run.metadata),
            ),
        )
        return run

    def list_runs(self, task_id: str | None = None) -> list[RunSummary]:
        query = "SELECT * FROM runs"
        params: list[Any] = []
        if task_id:
            query += " WHERE task_id = ?"
            params.append(task_id)
        query += " ORDER BY started_at ASC"
        return [
            RunSummary(
                task_id=row["task_id"],
                agent=row["agent"],
                status=row["status"],
                output=row["output"],
                artifacts=_loads(row["artifacts"], []),
                cost_usd=row["cost_usd"],
                duration_s=row["duration_s"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                error=row["error"],
                metadata=_loads(row["metadata"], {}),
            )
            for row in self.conn.execute(query, params).fetchall()
        ]

    # ------------------------------------------------------------- datasets

    def add_dataset(self, dataset: DatasetVersion) -> DatasetVersion:
        self.conn.execute(
            """
            INSERT INTO datasets (id, name, version, content_hash, record_count, token_count,
                                  created_at, lineage, stats, released, path, parent_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                dataset.id,
                dataset.name,
                dataset.version,
                dataset.content_hash,
                dataset.record_count,
                dataset.token_count,
                dataset.created_at,
                _json(dataset.lineage),
                _json(dataset.stats),
                int(dataset.released),
                dataset.path,
                dataset.parent_version,
            ),
        )
        return dataset

    def get_dataset(self, dataset_id: str) -> DatasetVersion | None:
        row = self.conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return self._row_to_dataset(row) if row else None

    def list_datasets(self, name: str | None = None) -> list[DatasetVersion]:
        query = "SELECT * FROM datasets"
        params: list[Any] = []
        if name:
            query += " WHERE name = ?"
            params.append(name)
        query += " ORDER BY created_at ASC"
        return [self._row_to_dataset(row) for row in self.conn.execute(query, params).fetchall()]

    def release_dataset(self, dataset_id: str) -> None:
        self.conn.execute("UPDATE datasets SET released = 1 WHERE id = ?", (dataset_id,))

    def add_dataset_records(self, dataset_id: str, records: Iterable[dict[str, Any]]) -> int:
        count = 0
        for record in records:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO dataset_records
                    (id, dataset_id, record_key, payload, source_id, sensitivity, acl, tokens,
                    split, meta) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.get("id") or new_id("rec"),
                    dataset_id,
                    record["record_key"],
                    record["payload"],
                    record.get("source_id", "unknown"),
                    record.get("sensitivity", "internal"),
                    _json(record.get("acl", [])),
                    int(record.get("tokens", 0)),
                    record.get("split", ""),
                    _json(record.get("meta", {})),
                ),
            )
            count += 1
        return count

    def set_record_splits(self, dataset_id: str, splits: dict[str, str]) -> int:
        for record_key, split in splits.items():
            self.conn.execute(
                "UPDATE dataset_records SET split = ? WHERE dataset_id = ? AND record_key = ?",
                (split, dataset_id, record_key),
            )
        return len(splits)

    def dataset_records_by_split(self, dataset_id: str, split: str) -> list[dict[str, Any]]:
        return [row for row in self.dataset_records(dataset_id) if row["split"] == split]

    def dataset_records(self, dataset_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM dataset_records WHERE dataset_id = ? ORDER BY record_key", (dataset_id,)
        ).fetchall()
        return [
            {
                "id": row["id"],
                "record_key": row["record_key"],
                "payload": row["payload"],
                "source_id": row["source_id"],
                "sensitivity": row["sensitivity"],
                "acl": _loads(row["acl"], []),
                "tokens": row["tokens"],
                "split": row["split"],
                "meta": _loads(row["meta"], {}),
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_dataset(row: sqlite3.Row) -> DatasetVersion:
        return DatasetVersion(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            content_hash=row["content_hash"],
            record_count=row["record_count"],
            token_count=row["token_count"],
            created_at=row["created_at"],
            lineage=_loads(row["lineage"], {}),
            stats=_loads(row["stats"], {}),
            released=bool(row["released"]),
            path=row["path"],
            parent_version=row["parent_version"],
        )

    # ---------------------------------------------------------------- models

    def add_model(self, model: ModelVersion) -> ModelVersion:
        self.conn.execute(
            """
            INSERT INTO models (id, name, version, content_hash, created_at, dataset_ids, config,
                                eval_report_id, status, card, path)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                model.id,
                model.name,
                model.version,
                model.content_hash,
                model.created_at,
                _json(model.dataset_ids),
                _json(model.config),
                model.eval_report_id,
                model.status,
                _json(model.card),
                model.path,
            ),
        )
        return model

    def get_model(self, model_id: str) -> ModelVersion | None:
        row = self.conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return self._row_to_model(row) if row else None

    def list_models(self, name: str | None = None) -> list[ModelVersion]:
        query = "SELECT * FROM models"
        params: list[Any] = []
        if name:
            query += " WHERE name = ?"
            params.append(name)
        query += " ORDER BY created_at ASC"
        return [self._row_to_model(row) for row in self.conn.execute(query, params).fetchall()]

    def set_model_status(
        self, model_id: str, status: str, eval_report_id: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE models SET status = ?, eval_report_id = COALESCE(?, eval_report_id) "
            "WHERE id = ?",
            (status, eval_report_id, model_id),
        )

    def update_model_card(self, model_id: str, card: dict[str, Any]) -> None:
        self.conn.execute("UPDATE models SET card = ? WHERE id = ?", (_json(card), model_id))

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            dataset_ids=_loads(row["dataset_ids"], []),
            config=_loads(row["config"], {}),
            eval_report_id=row["eval_report_id"],
            status=row["status"],
            card=_loads(row["card"], {}),
            path=row["path"],
        )

    # ------------------------------------------------------------ eval sets

    def add_eval_set(self, eval_set: EvalSet) -> EvalSet:
        self.conn.execute(
            """
            INSERT INTO eval_sets (id, name, version, frozen, created_at, approved_by,
                                   content_hash, path, items)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                eval_set.id,
                eval_set.name,
                eval_set.version,
                int(eval_set.frozen),
                eval_set.created_at,
                eval_set.approved_by,
                eval_set.content_hash,
                eval_set.path,
                _json(eval_set.items),
            ),
        )
        return eval_set

    def get_eval_set(self, eval_set_id: str) -> EvalSet | None:
        row = self.conn.execute("SELECT * FROM eval_sets WHERE id = ?", (eval_set_id,)).fetchone()
        return self._row_to_eval_set(row) if row else None

    def list_eval_sets(self) -> list[EvalSet]:
        return [
            self._row_to_eval_set(row)
            for row in self.conn.execute(
                "SELECT * FROM eval_sets ORDER BY created_at ASC"
            ).fetchall()
        ]

    def freeze_eval_set(self, eval_set_id: str, approved_by: str) -> None:
        self.conn.execute(
            "UPDATE eval_sets SET frozen = 1, approved_by = ? WHERE id = ?",
            (approved_by, eval_set_id),
        )

    @staticmethod
    def _row_to_eval_set(row: sqlite3.Row) -> EvalSet:
        return EvalSet(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            frozen=bool(row["frozen"]),
            created_at=row["created_at"],
            approved_by=row["approved_by"],
            content_hash=row["content_hash"],
            path=row["path"],
            items=_loads(row["items"], []),
        )

    # ---------------------------------------------------------- eval reports

    def add_eval_report(self, report: EvalReport) -> EvalReport:
        self.conn.execute(
            """
            INSERT INTO eval_reports (id, model_id, eval_set_id, created_at, metrics,
                                      baseline_metrics, checks, passed, notes)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                report.id,
                report.model_id,
                report.eval_set_id,
                report.created_at,
                _json(report.metrics),
                _json(report.baseline_metrics),
                _json(report.checks),
                int(report.passed),
                report.notes,
            ),
        )
        return report

    def get_eval_report(self, report_id: str) -> EvalReport | None:
        row = self.conn.execute("SELECT * FROM eval_reports WHERE id = ?", (report_id,)).fetchone()
        return self._row_to_report(row) if row else None

    def list_eval_reports(self, model_id: str | None = None) -> list[EvalReport]:
        query = "SELECT * FROM eval_reports"
        params: list[Any] = []
        if model_id:
            query += " WHERE model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at ASC"
        return [self._row_to_report(row) for row in self.conn.execute(query, params).fetchall()]

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> EvalReport:
        return EvalReport(
            id=row["id"],
            model_id=row["model_id"],
            eval_set_id=row["eval_set_id"],
            created_at=row["created_at"],
            metrics=_loads(row["metrics"], {}),
            baseline_metrics=_loads(row["baseline_metrics"], {}),
            checks=_loads(row["checks"], {}),
            passed=bool(row["passed"]),
            notes=row["notes"],
        )

    # ---------------------------------------------------------------- policy

    def upsert_policy(self, decision: PolicyDecision) -> PolicyDecision:
        self.conn.execute(
            """
            INSERT INTO policy_decisions (id, question, default_value, current_value, rationale,
                                           decided_by, decided_at, status)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                current_value=excluded.current_value,
                rationale=excluded.rationale,
                decided_by=excluded.decided_by,
                decided_at=excluded.decided_at,
                status=excluded.status
            """,
            (
                decision.id,
                decision.question,
                decision.default_value,
                decision.current_value,
                decision.rationale,
                decision.decided_by,
                decision.decided_at,
                decision.status,
            ),
        )
        return decision

    def list_policy(self) -> list[PolicyDecision]:
        return [
            PolicyDecision(
                id=row["id"],
                question=row["question"],
                default_value=row["default_value"],
                current_value=row["current_value"],
                rationale=row["rationale"],
                decided_by=row["decided_by"],
                decided_at=row["decided_at"],
                status=row["status"],
            )
            for row in self.conn.execute(
                "SELECT * FROM policy_decisions ORDER BY id ASC"
            ).fetchall()
        ]

    # ------------------------------------------------------------------- kv

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
