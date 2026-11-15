"""Which database this install talks to, named in one place.

Every module reaches the database through ``Store``, and ``Store`` reached a
file path. An enterprise install names a cluster instead, and the difference must
not be discovered halfway through a migration -- so the target is parsed here,
the sqlite case keeps working exactly as before, and a PostgreSQL URL is refused
with the reason and the next step rather than opening something that looks alive.

The schema a server gets is generated from the sqlite text by ``postgres_schema``,
so the two cannot drift; ``init_db`` applies it statement by statement when the
connection speaks PostgreSQL. What is still missing is the driver: without
``psycopg`` nothing can connect, and saying so is better than a handle that looks
alive until its first statement fails.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ENV_DATABASE_URL = "INTERNAL_AI_DATABASE_URL"
SQLITE_SCHEMES = frozenset({"sqlite", "sqlite3", "file"})
POSTGRES_SCHEMES = frozenset({"postgres", "postgresql"})

POSTGRES_DRIVER_MISSING = (
    "PostgreSQL needs its optional driver and it is not installed: "
    "pip install 'internal-ai[postgres]'. The schema is generated and verified "
    "against a server, but no statement can be sent without psycopg."
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
        if not parsed.netloc:
            raise DatabaseDenied(f"{text} names no host")
        return Target(scheme, text)
    raise DatabaseDenied(
        f"unknown database scheme {scheme!r}; this install speaks sqlite and is "
        "being ported to postgresql"
    )


POSTGRES_TRIGGERS = """
/* The audit table is append-only here for the same reason as in sqlite: an agent
   must not be able to rewrite history, and a trigger cannot be forgotten by a
   caller the way a convention can. */
CREATE OR REPLACE FUNCTION audit_events_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_events_no_update ON audit_events;
CREATE TRIGGER audit_events_no_update
    BEFORE UPDATE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION audit_events_append_only();

DROP TRIGGER IF EXISTS audit_events_no_delete ON audit_events;
CREATE TRIGGER audit_events_no_delete
    BEFORE DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION audit_events_append_only();
"""

_TRIGGER_BLOCK = re.compile(r"CREATE\s+TRIGGER\b.*?END\s*;", re.IGNORECASE | re.DOTALL)


def _dollar_tag(sql: str, index: int) -> str:
    """The opening tag of a dollar-quoted body, or empty if this is not one."""
    match = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$").match(sql, index)
    return match.group(0) if match else ""


def split_statements(sql: str) -> list[str]:
    """Statements in order, keeping dollar-quoted bodies and literals whole.

    A PL/pgSQL function body contains semicolons, so splitting on ``;`` would
    hand the server half a function and fail the migration somewhere unrelated.
    """
    statements: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(sql):
        char = sql[index]
        pair = sql[index : index + 2]
        if char in {"'", '"'}:
            end = _skip_quoted(sql, index, char)
            current.append(sql[index:end])
            index = end
            continue
        if char == "$":
            tag = _dollar_tag(sql, index)
            if tag:
                end = sql.find(tag, index + len(tag))
                end = len(sql) if end == -1 else end + len(tag)
                current.append(sql[index:end])
                index = end
                continue
        if pair == "--":
            end = sql.find("\n", index)
            end = len(sql) if end == -1 else end
            current.append(sql[index:end])
            index = end
            continue
        if pair == "/*":
            end = sql.find("*/", index)
            end = len(sql) if end == -1 else end + 2
            current.append(sql[index:end])
            index = end
            continue
        if char == ";":
            text = "".join(current).strip()
            if text:
                statements.append(text)
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def postgres_schema(schema: str) -> str:
    """The sqlite schema, in the dialect a server will accept.

    Generated from the same text rather than kept as a second file: two hand-
    maintained 400-line schemas drift, and the drift is invisible until a release
    reads a column the other side never had. Only the parts sqlite spells
    differently are rewritten; column names, keys and indexes are inherited, so a
    statement that runs on one runs on the other.
    """
    without_triggers = _TRIGGER_BLOCK.sub("", schema)
    statements = [part.strip() for part in without_triggers.split(";") if part.strip()]
    converted: list[str] = []
    for statement in statements:
        text = re.sub(
            r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
            "BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY",
            statement,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\bREAL\b", "DOUBLE PRECISION", text, flags=re.IGNORECASE)
        converted.append(f"{text};")
    return "\n\n".join(converted) + "\n\n" + POSTGRES_TRIGGERS


PLACEHOLDERS: dict[str, str] = {
    "sqlite": "?",
    "sqlite3": "?",
    "file": "?",
    "postgres": "%s",
    "postgresql": "%s",
}


def translate(sql: str, placeholder: str) -> str:
    """Rewrite ``?`` placeholders, leaving quoted text and comments untouched.

    A literal that contains a question mark is data (a JSON blob, a question in a
    prompt); rewriting it would corrupt the row. The scanner is what makes the
    shim safe enough to sit on the write path of every statement.
    """
    if placeholder == "?" or "?" not in sql:
        return sql
    out: list[str] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        pair = sql[index : index + 2]
        if char == "'":
            end = _skip_quoted(sql, index, "'")
            out.append(sql[index:end])
            index = end
            continue
        if char == '"':
            end = _skip_quoted(sql, index, '"')
            out.append(sql[index:end])
            index = end
            continue
        if pair == "--":
            end = sql.find("\n", index)
            end = length if end == -1 else end
            out.append(sql[index:end])
            index = end
            continue
        if pair == "/*":
            end = sql.find("*/", index)
            end = length if end == -1 else end + 2
            out.append(sql[index:end])
            index = end
            continue
        out.append(placeholder if char == "?" else char)
        index += 1
    return "".join(out)


def _skip_quoted(sql: str, start: int, quote: str) -> int:
    """Index just past a quoted run, honouring the doubled-quote escape."""
    index = start + 1
    while index < len(sql):
        if sql[index] == quote:
            if sql[index + 1 : index + 2] == quote:
                index += 2
                continue
            return index + 1
        index += 1
    return len(sql)


class Connection:
    """A connection that speaks the schema's sqlite dialect and no other yet.

    Wrapping the connection rather than rewriting every call site means the
    placeholders and row access live in one place, and this wrapper runs on every
    statement the suite executes -- in pass-through for sqlite -- so the shim
    cannot rot while the PostgreSQL schema is still being ported.
    """

    def __init__(self, raw: Any, dialect: str) -> None:
        self.raw = raw
        self.dialect = dialect
        self.placeholder = PLACEHOLDERS.get(dialect, "?")

    @property
    def is_sqlite(self) -> bool:
        return self.dialect in SQLITE_SCHEMES

    def execute(self, sql: str, params: Any = ()) -> Any:
        return self.raw.execute(translate(sql, self.placeholder), params)

    def executemany(self, sql: str, rows: Any) -> Any:
        return self.raw.executemany(translate(sql, self.placeholder), rows)

    def executescript(self, script: str) -> Any:
        if not self.is_sqlite:
            raise DatabaseDenied(
                "executescript runs a sqlite schema; a PostgreSQL schema is applied "
                "statement by statement in one transaction"
            )
        return self.raw.executescript(script)

    def close(self) -> None:
        self.raw.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)


def open_target(target: Target) -> Connection:
    """A connection to this target, or a refusal naming why not.

    Only sqlite has a schema here. Opening a PostgreSQL connection would produce a
    handle that cannot run a single statement of ``db.py``, so the refusal happens
    before the driver is asked for anything.
    """
    if target.scheme in POSTGRES_SCHEMES:
        return _open_postgres(target)
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
    return Connection(conn, target.scheme)


def _open_postgres(target: Target) -> Connection:
    """A server connection, or the refusal that names what is missing.

    Autocommit is on because the application manages its own transactions and the
    server would otherwise keep one open from the first INSERT until close.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ModuleNotFoundError as missing:
        raise DatabaseDenied(POSTGRES_DRIVER_MISSING) from missing
    raw = psycopg.connect(target.location, autocommit=True, row_factory=dict_row)
    return Connection(raw, "postgres")


def describe(target: Target) -> dict[str, Any]:
    """What an operator sees when the install reports where it is writing."""
    return {"scheme": target.scheme, "target": target.label, "sqlite": target.is_sqlite}
