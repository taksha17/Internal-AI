#!/usr/bin/env python
"""Copy this install's sqlite database into a PostgreSQL server.

    INTERNAL_AI_DATABASE_URL=postgresql://user:pw@host:5432/internal_ai \
        .venv/bin/python scripts/migrate_to_postgres.py var/internal_ai.db

The target is created if it is empty and refused if it is not: a second copy would
insert every record twice and fork the audit chain. The source is opened read-only
and is not modified, so the file stays as the rollback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from internal_ai.dialects import ENV_DATABASE_URL
from internal_ai.migrate import MigrationRefused, copy_sqlite_to_server
from internal_ai.store import Store


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: migrate_to_postgres.py <path to internal_ai.db>")
        return 2
    url = os.environ.get(ENV_DATABASE_URL, "").strip()
    if not url.startswith(("postgresql://", "postgres://")):
        print(f"{ENV_DATABASE_URL} must name the server this install is moving to")
        return 2

    source = Path(argv[1])
    try:
        store = Store(url)
    except Exception as failure:
        print(f"could not open the target: {failure}")
        return 1
    try:
        report = copy_sqlite_to_server(source, store.conn)
    except MigrationRefused as refusal:
        print(f"refused: {refusal}")
        return 1
    finally:
        store.close()

    for table, count in sorted(report.items()):
        if count:
            print(f"{table:24} {count}")
    print(f"{sum(report.values())} rows copied from {source}")
    print("the sqlite file is untouched; keep it until the server has served a day")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
