"""SQLite storage: connection handling and schema.

The audit table is protected by triggers so that side-effecting history cannot
be edited or removed by an agent (NFR: Auditability, FR-20).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    owner_role TEXT NOT NULL,
    state TEXT NOT NULL,
    dependencies TEXT NOT NULL DEFAULT '[]',
    acceptance_criteria TEXT NOT NULL DEFAULT '[]',
    cost_estimate_usd REAL NOT NULL DEFAULT 0,
    cost_cap_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    review_attempts INTEGER NOT NULL DEFAULT 0,
    artifacts TEXT NOT NULL DEFAULT '[]',
    reviewer_role TEXT,
    gate_id TEXT,
    milestone TEXT,
    blocked_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    last_state_change TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_role);

CREATE TABLE IF NOT EXISTS task_events (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    from_state TEXT,
    to_state TEXT NOT NULL,
    actor TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id);

CREATE TABLE IF NOT EXISTS gates (
    id TEXT PRIMARY KEY,
    gate_type TEXT NOT NULL,
    title TEXT NOT NULL,
    subject TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    requested_by_kind TEXT NOT NULL,
    required_approvers TEXT NOT NULL DEFAULT '[]',
    evidence TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    decision_note TEXT NOT NULL DEFAULT '',
    cost_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_gates_status ON gates(status);

CREATE TABLE IF NOT EXISTS gate_decisions (
    id TEXT PRIMARY KEY,
    gate_id TEXT NOT NULL REFERENCES gates(id),
    approver TEXT NOT NULL,
    approver_kind TEXT NOT NULL,
    approved INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gate_decisions_gate ON gate_decisions(gate_id);

CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    actor TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    action TEXT NOT NULL,
    subject TEXT NOT NULL,
    outcome TEXT NOT NULL,
    inputs TEXT NOT NULL DEFAULT '{}',
    details TEXT NOT NULL DEFAULT '{}',
    cost_usd REAL NOT NULL DEFAULT 0,
    prev_hash TEXT NOT NULL DEFAULT '',
    hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tenant TEXT NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_events(action);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_events(actor);

CREATE TRIGGER IF NOT EXISTS audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;

CREATE TABLE IF NOT EXISTS budget_limits (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    limit_usd REAL NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT,
    PRIMARY KEY (scope, key)
);

CREATE TABLE IF NOT EXISTS spend_records (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    amount_usd REAL NOT NULL,
    agent TEXT NOT NULL,
    task_id TEXT,
    job_id TEXT,
    created_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_spend_scope ON spend_records(scope, key);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    output TEXT NOT NULL DEFAULT '',
    artifacts TEXT NOT NULL DEFAULT '[]',
    cost_usd REAL NOT NULL DEFAULT 0,
    duration_s REAL NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    error TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    lineage TEXT NOT NULL DEFAULT '{}',
    stats TEXT NOT NULL DEFAULT '{}',
    released INTEGER NOT NULL DEFAULT 0,
    path TEXT NOT NULL DEFAULT '',
    parent_version TEXT
);

CREATE TABLE IF NOT EXISTS dataset_records (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id),
    record_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    source_id TEXT NOT NULL,
    sensitivity TEXT NOT NULL DEFAULT 'internal',
    acl TEXT NOT NULL DEFAULT '[]',
    tokens INTEGER NOT NULL DEFAULT 0,
    split TEXT NOT NULL DEFAULT '',
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_dataset_records_dataset ON dataset_records(dataset_id);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dataset_ids TEXT NOT NULL DEFAULT '[]',
    config TEXT NOT NULL DEFAULT '{}',
    eval_report_id TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    card TEXT NOT NULL DEFAULT '{}',
    path TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);

CREATE TABLE IF NOT EXISTS eval_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    frozen INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    approved_by TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    items TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS eval_reports (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    eval_set_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics TEXT NOT NULL DEFAULT '{}',
    baseline_metrics TEXT NOT NULL DEFAULT '{}',
    checks TEXT NOT NULL DEFAULT '{}',
    passed INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS policy_decisions (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    default_value TEXT NOT NULL,
    current_value TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    decided_by TEXT,
    decided_at TEXT,
    status TEXT NOT NULL DEFAULT 'undecided'
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    subject TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);

CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT NOT NULL,
    roles TEXT NOT NULL DEFAULT '',
    groups TEXT NOT NULL DEFAULT '',
    placeholder INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_tombstones (
    source_id TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(dataset_records)").fetchall()}
    if "split" not in columns:
        conn.execute("ALTER TABLE dataset_records ADD COLUMN split TEXT NOT NULL DEFAULT ''")
    conn.execute(
        "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
