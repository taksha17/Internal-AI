"""Carry an existing sqlite install onto a server, in one transaction.

An install that has been running for a year has a ``internal_ai.db`` with tasks,
gates, datasets and an audit chain. "Point at a cluster" cannot mean "start
empty", so this module copies the rows across in an order the foreign keys allow
and then proves the counts match. Two things it refuses to do quietly: write into
a server that already has rows (that would double every record), and lose the
audit sequence (the chain is ordered by it).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Written by whoever opens the database, not by the install's history: the target
# already carries the right version, and copying it would collide on the key.
BOOKKEEPING = frozenset({"schema_meta"})


class MigrationRefused(RuntimeError):
    """The copy was refused rather than half-done."""


def sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]


def table_order(conn: sqlite3.Connection) -> list[str]:
    """Tables ordered so a referenced table is written before the one pointing at it.

    Readings come from the source's own foreign keys rather than a hand-kept list,
    which would go stale the first time a table gains a reference. Ties are broken
    alphabetically so two runs produce the same order.
    """
    names = sqlite_tables(conn)
    dependencies: dict[str, set[str]] = {name: set() for name in names}
    for name in names:
        for row in conn.execute(f"PRAGMA foreign_key_list({name})").fetchall():
            parent = row[2]
            if parent in dependencies and parent != name:
                dependencies[name].add(parent)

    ordered: list[str] = []
    remaining = dict(dependencies)
    while remaining:
        ready = sorted(name for name, needs in remaining.items() if not needs - set(ordered))
        if not ready:
            raise MigrationRefused(
                "the source schema has a reference cycle between "
                f"{', '.join(sorted(remaining))}; a copy order cannot be derived"
            )
        for name in ready:
            ordered.append(name)
            del remaining[name]
    return ordered


def _target_counts(conn: Any, tables: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in tables:
        row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
        counts[table] = row["n"]
    return counts


def copy_sqlite_to_server(source: str | Path, target: Any) -> dict[str, int]:
    """Copy every row, in order, then verify. Returns the row count per table.

    The target must already have the schema (``Store`` creates it on open) and must
    be empty, because a second run would insert the corpus twice and the audit
    chain would fork.
    """
    if getattr(target, "is_sqlite", True):
        raise MigrationRefused("this copies into a server; for sqlite the file is the database")
    path = Path(source)
    if not path.exists():
        raise MigrationRefused(f"{path} does not exist")

    local = sqlite3.connect(path)
    local.row_factory = sqlite3.Row
    try:
        order = [table for table in table_order(local) if table not in BOOKKEEPING]
        existing = _target_counts(target, order)
        already = {table: count for table, count in existing.items() if count}
        if already:
            raise MigrationRefused(
                f"the target already holds rows ({already}); refusing to copy a second time"
            )

        copied: dict[str, int] = {}
        for table in order:
            rows = local.execute(f"SELECT * FROM {table}").fetchall()
            copied[table] = len(rows)
            if not rows:
                continue
            columns = list(rows[0].keys())
            placeholders = ", ".join("?" for _ in columns)
            statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            target.executemany(
                statement,
                [tuple(row[column] for column in columns) for row in rows],
            )
    finally:
        local.close()

    arrived = _target_counts(target, order)
    short = {
        table: (copied[table], arrived[table]) for table in order if copied[table] != arrived[table]
    }
    if short:
        raise MigrationRefused(f"row counts disagree after the copy (source, target): {short}")

    _continue_audit_sequence(target)
    return copied


def _continue_audit_sequence(target: Any) -> None:
    """Move the identity past the copied rows, or the next event reuses a sequence.

    ``seq`` is copied explicitly to keep the chain in order, which leaves the
    identity counter at its start; without this the first new event collides.
    """
    row = target.execute("SELECT COALESCE(max(seq), 0) AS last FROM audit_events").fetchone()
    if not row["last"]:
        return
    target.execute(
        "SELECT setval(pg_get_serial_sequence('audit_events', 'seq'), ?, true)",
        (row["last"],),
    )
