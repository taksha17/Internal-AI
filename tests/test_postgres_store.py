"""The platform's own writes, against a real PostgreSQL server.

Skipped unless ``INTERNAL_AI_TEST_PG_URL`` points at a reachable server, because CI
has none. Run it by hand against the tunnel:

    INTERNAL_AI_TEST_PG_URL="$(cat .secrets/pg_url)" \\
        .venv/bin/python -m pytest tests/test_postgres_store.py

What it proves is the thing the schema check cannot: that ``Store`` -- the only
door the application has to a database -- creates its schema, writes, reads back
and chains an audit event on a server, with the dictionary rows and the ``?``
placeholders the sqlite path relies on.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from internal_ai.domain import AuditEvent, AuditOutcome, Task, TaskKind, TaskState
from internal_ai.store import Store

URL = os.environ.get("INTERNAL_AI_TEST_PG_URL", "").strip()
pytestmark = pytest.mark.skipif(not URL, reason="no INTERNAL_AI_TEST_PG_URL to test against")


@pytest.fixture()
def store() -> Iterator[Store]:
    opened = Store(URL)
    # A server keeps its rows between runs, so each test starts from an empty
    # audit chain -- otherwise it would assert against the previous run's hash.
    opened.conn.execute("TRUNCATE tasks, task_events, audit_events RESTART IDENTITY CASCADE")
    try:
        yield opened
    finally:
        opened.close()


def test_it_opens_a_server_connection(store: Store) -> None:
    assert store.conn.dialect == "postgres"
    assert store.conn.is_sqlite is False
    version = store.conn.execute("SELECT value FROM schema_meta WHERE key = ?", ("version",))
    assert version.fetchone()["value"] == "1"


def test_the_schema_is_the_one_the_port_check_expects(store: Store) -> None:
    counted = store.conn.execute(
        "SELECT count(*) AS n FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchone()

    assert counted["n"] == 34


def test_a_task_survives_a_round_trip(store: Store) -> None:
    task = Task(
        id="pg-task-1",
        title="Write through a server",
        description="Does a question mark in prose? survive the shim?",
        kind=TaskKind.INFRA,
        owner_role="platform_engineer",
        state=TaskState.PROPOSED,
        dependencies=["pg-task-0"],
        metadata={"why": "a ? inside a literal is data"},
    )

    store.add_task(task)
    found = store.get_task("pg-task-1")

    assert found is not None
    assert found.title == "Write through a server"
    assert "survive the shim?" in found.description
    assert found.dependencies == ["pg-task-0"]
    assert found.metadata == {"why": "a ? inside a literal is data"}
    assert str(found.kind) == "infra"


def test_a_state_change_is_recorded_once(store: Store) -> None:
    task = store.add_task(
        Task(id="pg-task-2", title="t", description="", kind=TaskKind.DATA, owner_role="x")
    )

    store.set_task_state(task, TaskState.APPROVED, actor="owner", actor_kind="human")

    found = store.get_task("pg-task-2")
    assert found is not None
    assert str(found.state) == "approved"
    assert [str(event.to_state) for event in store.task_events("pg-task-2")] == [
        "proposed",
        "approved",
    ]


def test_the_audit_chain_links_on_a_server(store: Store) -> None:
    first = store.append_audit(
        AuditEvent(
            id="pg-audit-1",
            seq=0,
            actor="platform_engineer",
            actor_kind="agent",
            action="test.first",
            subject="postgres",
            outcome=AuditOutcome.SUCCESS,
        )
    )
    second = store.append_audit(
        AuditEvent(
            id="pg-audit-2",
            seq=0,
            actor="platform_engineer",
            actor_kind="agent",
            action="test.second",
            subject="postgres",
            outcome=AuditOutcome.SUCCESS,
        )
    )

    assert second.prev_hash == first.hash
    chain = store.audit_chain()
    assert [event.id for event in chain] == ["pg-audit-1", "pg-audit-2"]
    # The server assigns the sequence; the chain is what proves it arrived in order.
    assert [event.seq for event in chain] == [1, 2]
    assert chain[1].prev_hash == chain[0].hash


def test_the_server_refuses_to_rewrite_an_audit_event(store: Store) -> None:
    import psycopg

    store.append_audit(
        AuditEvent(
            id="pg-audit-3",
            seq=0,
            actor="platform_engineer",
            actor_kind="agent",
            action="test.append_only",
            subject="postgres",
            outcome=AuditOutcome.SUCCESS,
        )
    )

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        store.conn.execute("UPDATE audit_events SET actor = 'tamper' WHERE id = 'pg-audit-3'")
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        store.conn.execute("DELETE FROM audit_events WHERE id = 'pg-audit-3'")
