"""Moving an existing sqlite install onto a server.

The order half runs anywhere; the copy half is skipped unless a server is
reachable, because the point of the copy is that a real server accepted it.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from internal_ai.domain import AuditEvent, AuditOutcome, Task, TaskKind
from internal_ai.migrate import MigrationRefused, copy_sqlite_to_server, table_order
from internal_ai.store import Store

URL = os.environ.get("INTERNAL_AI_TEST_PG_URL", "").strip()

needs_server = pytest.mark.skipif(not URL, reason="no INTERNAL_AI_TEST_PG_URL to migrate into")


def test_a_referenced_table_comes_before_the_one_pointing_at_it(tmp_path: Path) -> None:
    store = Store(tmp_path / "legacy.db")
    try:
        order = table_order(store.conn)
    finally:
        store.close()

    assert order.index("people") < order.index("accounts")
    assert order.index("people") < order.index("sessions")
    assert order.index("datasets") < order.index("dataset_records")
    assert order.index("tasks") < order.index("task_events")


def test_the_order_is_the_same_every_run(tmp_path: Path) -> None:
    store = Store(tmp_path / "legacy.db")
    try:
        assert table_order(store.conn) == table_order(store.conn)
    finally:
        store.close()


def test_a_reference_cycle_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    local = sqlite3.connect(tmp_path / "cycle.db")
    local.executescript(
        """
        CREATE TABLE a (id TEXT PRIMARY KEY, b_id TEXT REFERENCES b(id));
        CREATE TABLE b (id TEXT PRIMARY KEY, a_id TEXT REFERENCES a(id));
        """
    )

    with pytest.raises(MigrationRefused, match="cycle"):
        table_order(local)
    local.close()


def test_copying_into_sqlite_is_refused(tmp_path: Path) -> None:
    target = Store(tmp_path / "target.db")
    try:
        with pytest.raises(MigrationRefused, match="server"):
            copy_sqlite_to_server(tmp_path / "target.db", target.conn)
    finally:
        target.close()


def test_a_missing_source_is_refused(tmp_path: Path) -> None:
    class Server:
        is_sqlite = False

    with pytest.raises(MigrationRefused, match="does not exist"):
        copy_sqlite_to_server(tmp_path / "nothing.db", Server())


@pytest.mark.skipif(not URL, reason="no INTERNAL_AI_TEST_PG_URL to migrate into")
def test_a_sqlite_install_moves_to_the_server_with_its_chain_intact(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.db"
    source = Store(legacy)
    try:
        source.add_task(
            Task(
                id="legacy-task",
                title="Raised before the move",
                description="still here afterwards",
                kind=TaskKind.DATA,
                owner_role="data_engineer",
            )
        )
        for index in (1, 2):
            source.append_audit(
                AuditEvent(
                    id=f"legacy-audit-{index}",
                    seq=0,
                    actor="data_engineer",
                    actor_kind="agent",
                    action=f"legacy.step.{index}",
                    subject="file",
                    outcome=AuditOutcome.SUCCESS,
                )
            )
    finally:
        source.close()

    server = Store(URL)
    try:
        server.conn.execute("TRUNCATE tasks, task_events, audit_events RESTART IDENTITY CASCADE")

        report = copy_sqlite_to_server(legacy, server.conn)

        assert report["tasks"] == 1
        assert report["audit_events"] == 2
        moved = server.get_task("legacy-task")
        assert moved is not None
        assert moved.description == "still here afterwards"

        chain = server.audit_chain()
        assert [event.id for event in chain] == ["legacy-audit-1", "legacy-audit-2"]
        assert chain[1].prev_hash == chain[0].hash

        # The identity has to continue past the copied rows, or the next event
        # reuses a sequence and the chain has two of the same number.
        server.append_audit(
            AuditEvent(
                id="after-the-move",
                seq=0,
                actor="data_engineer",
                actor_kind="agent",
                action="moved.on",
                subject="server",
                outcome=AuditOutcome.SUCCESS,
            )
        )
        assert server.audit_chain()[-1].seq == 3

        with pytest.raises(MigrationRefused, match="already holds rows"):
            copy_sqlite_to_server(legacy, server.conn)
    finally:
        server.close()
