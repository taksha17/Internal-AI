#!/usr/bin/env python
"""Compare the sqlite schema with the one a PostgreSQL server actually holds.

The port is only real if the server agrees with the file. This is dev tooling, not
a test: it needs a reachable PostgreSQL, which CI does not have, so it is run by
hand and recorded in docs/changes when the schema changes.

    INTERNAL_AI_TEST_PG_URL=postgresql://user:pw@host:port/db \
        .venv/bin/python scripts/pg_port_check.py

A scratch sqlite database is built from the same SCHEMA the application uses, the
PostgreSQL DDL is generated from it, applied with psql, and the two inventories are
compared. Any table or column present on one side and missing on the other is a
failure, because that is how a release reads a column that never existed.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from internal_ai.db import SCHEMA
from internal_ai.dialects import postgres_schema


def sqlite_inventory(conn: sqlite3.Connection) -> dict[str, set[str]]:
    conn.executescript(SCHEMA)
    return _inventory(conn)


def _inventory(conn: sqlite3.Connection) -> dict[str, set[str]]:
    conn.row_factory = sqlite3.Row
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    inventory: dict[str, set[str]] = {}
    for table in tables:
        columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
        inventory[table] = {row["name"] for row in columns}
    return inventory


def psql(url: str, sql: str) -> str:
    result = subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-tA", "-F", "|", "-d", url, "-c", sql],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"psql failed: {result.stderr.strip()}")
    return result.stdout


def postgres_inventory(url: str) -> dict[str, set[str]]:
    rows = psql(
        url,
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position",
    )
    inventory: dict[str, set[str]] = {}
    for line in rows.splitlines():
        if not line.strip():
            continue
        table, column = line.split("|", 1)
        inventory.setdefault(table, set()).add(column)
    return inventory


def main() -> int:
    url = os.environ.get("INTERNAL_AI_TEST_PG_URL", "").strip()
    if not url:
        print("INTERNAL_AI_TEST_PG_URL is not set; nothing to compare against")
        return 2

    ddl = postgres_schema(SCHEMA)
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as handle:
        handle.write(ddl)
        ddl_path = handle.name
    applied = subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-q", "-d", url, "-f", ddl_path],
        capture_output=True,
        text=True,
    )
    if applied.returncode != 0:
        print("applying the PostgreSQL schema failed:\n" + applied.stderr.strip())
        return 1

    with sqlite3.connect(":memory:") as local:
        expected = sqlite_inventory(local)
    actual = postgres_inventory(url)

    missing_tables = sorted(set(expected) - set(actual))
    extra_tables = sorted(set(actual) - set(expected))
    column_drift = {
        table: (sorted(expected[table] - actual[table]), sorted(actual[table] - expected[table]))
        for table in sorted(set(expected) & set(actual))
        if expected[table] != actual[table]
    }

    print(f"tables: sqlite {len(expected)}, postgres {len(actual)}")
    if missing_tables:
        print("missing on postgres:", ", ".join(missing_tables))
    if extra_tables:
        print("only on postgres:", ", ".join(extra_tables))
    for table, (missing, extra) in column_drift.items():
        print(f"column drift in {table}: missing {missing}, extra {extra}")

    if missing_tables or extra_tables or column_drift:
        return 1
    print("schemas agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
