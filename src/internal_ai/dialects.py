"""Which database this install talks to, named in one place.

Every module reaches the database through ``Store``, and ``Store`` reached a
file path. An enterprise install names a cluster instead, and the difference must
not be discovered halfway through a migration -- so the target is parsed here,
the sqlite case keeps working exactly as before, and a PostgreSQL URL is refused
with the reason and the next step rather than opening something that looks alive.

The refusal is deliberate. The schema in ``db.py`` is sqlite through and through:
PRAGMA pragmas, ``AUTOINCREMENT``, ``strftime``, triggers, and ``INSERT OR
REPLACE``. Pointing a PostgreSQL connection at it would fail on the first
statement with a syntax error an operator has to decode; naming the gap is more
useful than a traceback, and it keeps a half-ported database from looking
supported.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ENV_DATABASE_URL = "INTERNAL_AI_DATABASE_URL"
SQLITE_SCHEMES = frozenset({"sqlite", "sqlite3", "file"})
POSTGRES_SCHEMES = frozenset({"postgres", "postgresql"})

POSTGRES_NOT_PORTED = (
    "PostgreSQL is not ported yet: the schema still uses PRAGMA, AUTOINCREMENT, "
    "strftime, triggers and INSERT OR REPLACE, none of which exist there. Set "
    f"{ENV_DATABASE_URL} to a sqlite:/// path until the migration lands; a "
    "connection that opened against the sqlite schema would fail on its first "
    "statement."
)


class DatabaseDenied(RuntimeError):
    """The database target was refused instead of opened with a guess."""


@dataclass(frozen=True, slots=True)
class Target:
    """A parsed database destination: the scheme, and where it points."""

    scheme: str
    location: str

    @property
    def is_sqlite(self) -> bool:
        return self.scheme in SQLITE_SCHEMES

    @property
    def is_memory(self) -> bool:
        return self.is_sqlite and self.location in {":memory:", ""}

    @property
    def label(self) -> str:
        if self.is_memory:
            return "sqlite://:memory:"
        if self.is_sqlite:
            return f"sqlite://{self.location}"
        return f"{self.scheme}://{_redact(self.location)}"

    @property
    def path(self) -> Path:
        if not self.is_sqlite:
            raise DatabaseDenied(f"{self.label} is not a file path")
        if self.is_memory:
            raise DatabaseDenied("an in-memory database has no path")
        return Path(self.location)


def _redact(location: str) -> str:
    """A target is logged; the password in it is not."""
    parsed = urlparse(location if "://" in location else f"//{location}")
    if parsed.password:
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return f"{parsed.scheme}://{netloc}{parsed.path}" if parsed.scheme else netloc
    return location


def parse(source: str | Path) -> Target:
    """A path or a URL, resolved to one target; anything else is refused.

    A bare path is sqlite, which is what every existing call site passes and what
    every existing install has on disk. A URL has to name a scheme this module
    knows, because guessing is how an install ends up writing to a file nobody
    looked for.
    """
    if isinstance(source, Path):
        return Target("sqlite", str(source))
    text = str(source).strip()
    if not text:
        raise DatabaseDenied("a database target is a path or a URL, and this one is empty")
    if "://" not in text:
        return Target("sqlite", text)
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme in SQLITE_SCHEMES:
        location = f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        if location in {"", "/"}:
            raise DatabaseDenied(f"{text} names no database file")
        if location == "/:memory:":
            return Target("sqlite", ":memory:")
        return Target(scheme, location)
    if scheme in POSTGRES_SCHEMES:
        raise DatabaseDenied(POSTGRES_NOT_PORTED)
    raise DatabaseDenied(
        f"unknown database scheme {scheme!r}; this install speaks sqlite and is "
        "being ported to postgresql"
    )


def open_target(target: Target) -> sqlite3.Connection:
    """A connection to this target, or a refusal naming why not.

    Only sqlite has a schema here. Opening a PostgreSQL connection would produce a
    handle that cannot run a single statement of ``db.py``, so the refusal happens
    before the driver is asked for anything.
    """
    if not target.is_sqlite:
        raise DatabaseDenied(POSTGRES_NOT_PORTED)
    if target.is_memory:
        conn = sqlite3.connect(":memory:", isolation_level=None, timeout=30.0)
    else:
        path = target.path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def describe(target: Target) -> dict[str, Any]:
    """What an operator sees when the install reports where it is writing."""
    return {"scheme": target.scheme, "target": target.label, "sqlite": target.is_sqlite}
