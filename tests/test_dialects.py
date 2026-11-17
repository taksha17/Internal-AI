"""One place names the database, and a scheme it cannot serve is refused.

The value here is the refusal as much as the parsing: an install pointed at
PostgreSQL must stop with a reason it can act on, not open a handle and fail on
the first PRAGMA.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai import dialects
from internal_ai.config import Settings
from internal_ai.context import Platform
from internal_ai.db import SCHEMA, init_db, transaction
from internal_ai.dialects import (
    Connection,
    DatabaseDenied,
    Target,
    describe,
    open_target,
    parse,
    postgres_schema,
    split_statements,
    translate,
)
from internal_ai.store import Store


def test_a_bare_path_is_sqlite(tmp_path: Path) -> None:
    target = parse(tmp_path / "db.sqlite")

    assert target.is_sqlite
    assert target.path == tmp_path / "db.sqlite"


def test_a_sqlite_url_names_the_same_file(tmp_path: Path) -> None:
    target = parse(f"sqlite://{tmp_path / 'db.sqlite'}")

    assert target.is_sqlite
    assert target.path == tmp_path / "db.sqlite"


def test_an_in_memory_url_is_recognised() -> None:
    target = parse("sqlite:///:memory:")

    assert target.is_memory
    with pytest.raises(DatabaseDenied, match="no path"):
        _ = target.path


def test_postgresql_is_a_target_now_that_the_schema_is_generated() -> None:
    target = parse("postgresql://user:secret@db.internal:5432/internal_ai")

    assert target.scheme == "postgresql"
    assert not target.is_sqlite
    assert "secret" not in target.label


def test_a_server_url_without_a_host_is_refused() -> None:
    with pytest.raises(DatabaseDenied, match="names no host"):
        parse("postgresql:///internal_ai")


def test_opening_a_server_target_without_the_driver_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("psycopg"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_psycopg)

    with pytest.raises(DatabaseDenied) as refused:
        open_target(parse("postgresql://u:p@127.0.0.1:5433/internal_ai"))

    message = str(refused.value)
    assert "psycopg" in message
    assert "internal-ai[postgres]" in message


def test_an_unknown_scheme_is_refused_rather_than_guessed() -> None:
    with pytest.raises(DatabaseDenied, match="unknown database scheme"):
        parse("mysql://host/db")


def test_an_empty_target_is_refused() -> None:
    with pytest.raises(DatabaseDenied, match="empty"):
        parse("   ")


def test_a_password_is_not_printed_in_the_label() -> None:
    target = Target("postgresql", "postgresql://user:secret@db.internal:5432/x")

    assert "secret" not in target.label
    assert "user" in target.label


def test_describe_reports_the_scheme_for_an_operator(tmp_path: Path) -> None:
    report = describe(parse(tmp_path / "db.sqlite"))

    assert report["scheme"] == "sqlite"
    assert report["sqlite"] is True


def test_opening_a_sqlite_target_creates_the_file_and_its_directory(tmp_path: Path) -> None:
    conn = open_target(parse(tmp_path / "nested" / "db.sqlite"))
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO t (id) VALUES (1)")
        assert conn.execute("SELECT id FROM t").fetchone()["id"] == 1
    finally:
        conn.close()
    assert (tmp_path / "nested" / "db.sqlite").exists()


def test_a_store_accepts_a_url_and_writes(tmp_path: Path) -> None:
    store = Store(f"sqlite://{tmp_path / 'url.db'}")
    try:
        assert store.conn.execute("SELECT COUNT(*) AS n FROM datasets").fetchone()["n"] == 0
    finally:
        store.close()
    assert (tmp_path / "url.db").exists()


def test_the_settings_default_is_the_sqlite_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(dialects.ENV_DATABASE_URL, raising=False)
    settings = Settings.from_env(tmp_path)

    assert settings.database_url == f"sqlite://{settings.db_path}"
    assert parse(settings.database_url).path == settings.db_path


def test_a_configured_url_wins_over_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(dialects.ENV_DATABASE_URL, "sqlite:///:memory:")
    settings = Settings.from_env(tmp_path)

    assert settings.database_url == "sqlite:///:memory:"


def test_a_platform_refuses_a_url_it_cannot_serve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("psycopg"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    monkeypatch.setenv(dialects.ENV_DATABASE_URL, "postgresql://u:p@127.0.0.1:5433/ai")

    with pytest.raises(DatabaseDenied):
        Platform.create(root=tmp_path)


def test_a_platform_still_opens_the_default_database(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path) as platform:
        assert platform.store.target.is_sqlite
        assert platform.settings.db_path.exists()


def test_placeholders_become_percent_s_for_a_server_dialect() -> None:
    assert translate("SELECT a FROM t WHERE b = ?", "%s") == "SELECT a FROM t WHERE b = %s"


def test_a_sqlite_statement_is_never_rewritten() -> None:
    sql = "SELECT '?' AS q WHERE a = ?"

    assert translate(sql, "?") == sql


def test_a_question_mark_inside_a_literal_is_data() -> None:
    sql = 'SELECT \'{"q":"why?"}\' AS j WHERE id = ?'

    rewritten = translate(sql, "%s")

    assert '\'{"q":"why?"}\'' in rewritten
    assert rewritten.endswith("WHERE id = %s")


def test_a_doubled_quote_does_not_end_the_literal() -> None:
    rewritten = translate("SELECT 'it''s ?' AS s WHERE id = ?", "%s")

    assert "'it''s ?'" in rewritten
    assert rewritten.endswith("%s")


def test_a_quoted_identifier_is_untouched() -> None:
    rewritten = translate('SELECT "weird?column" FROM t WHERE id = ?', "%s")

    assert '"weird?column"' in rewritten
    assert rewritten.endswith("%s")


def test_comments_are_untouched() -> None:
    assert translate("SELECT 1 -- why?\n", "%s") == "SELECT 1 -- why?\n"
    assert translate("/* why? */ SELECT 1", "%s") == "/* why? */ SELECT 1"


def test_the_wrapper_passes_sqlite_through_unchanged(tmp_path: Path) -> None:
    conn = open_target(parse(tmp_path / "x.db"))
    try:
        conn.execute("CREATE TABLE t (id INTEGER, question TEXT)")
        conn.execute("INSERT INTO t (id, question) VALUES (?, ?)", (1, "why?"))
        found = conn.execute("SELECT question FROM t WHERE id = ?", (1,)).fetchone()
        assert found["question"] == "why?"
    finally:
        conn.close()


def test_the_wrapper_exposes_transaction_state(tmp_path: Path) -> None:
    conn = open_target(parse(tmp_path / "x.db"))
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert conn.in_transaction
        conn.execute("COMMIT")
        assert not conn.in_transaction
    finally:
        conn.close()


def test_executescript_refuses_a_non_sqlite_dialect() -> None:
    conn = Connection(raw=object(), dialect="postgres")

    with pytest.raises(DatabaseDenied, match="executescript"):
        conn.executescript("CREATE TABLE t (id int);")


def test_a_server_dialect_wrapper_rewrites_what_it_is_handed() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def execute(self, sql: str, params: object = ()) -> None:
            self.calls.append((sql, params))

    raw = Recorder()
    conn = Connection(raw=raw, dialect="postgres")

    conn.execute("SELECT '?' AS q WHERE id = ?", (7,))

    assert raw.calls == [("SELECT '?' AS q WHERE id = %s", (7,))]


def test_the_server_schema_keeps_no_sqlite_spelling() -> None:
    ddl = postgres_schema(SCHEMA)

    for leftover in ("AUTOINCREMENT", "RAISE(", "sqlite_master", "REAL "):
        assert leftover not in ddl, leftover


def test_the_audit_sequence_becomes_an_identity_column() -> None:
    ddl = postgres_schema(SCHEMA)

    assert "seq BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY" in ddl


def test_the_audit_table_is_append_only_on_the_server_too() -> None:
    ddl = postgres_schema(SCHEMA)

    assert "audit_events_append_only" in ddl
    assert "CREATE TRIGGER audit_events_no_update" in ddl
    assert "CREATE TRIGGER audit_events_no_delete" in ddl
    assert "BEFORE UPDATE ON audit_events" in ddl


def test_every_table_of_the_schema_reaches_the_server() -> None:
    import re as _re

    names = set(_re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA))
    assert len(names) == 31

    ddl = postgres_schema(SCHEMA)
    missing = sorted(name for name in names if name not in ddl)
    assert missing == []


def test_the_generated_schema_is_statement_terminated() -> None:
    ddl = postgres_schema(SCHEMA)

    assert ddl.count("CREATE TABLE") == 31
    assert not ddl.strip().endswith(";\n\n")
    assert "END;" in ddl


class _FakeServer:
    """A server connection that records what would be sent."""

    is_sqlite = False
    dialect = "postgres"

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: object = ()) -> _FakeServer:
        self.statements.append((sql, tuple(params) if params else ()))
        return self

    def close(self) -> None:
        return None


def test_a_function_body_is_one_statement() -> None:
    sql = (
        "CREATE FUNCTION f() RETURNS trigger AS $$ BEGIN RAISE EXCEPTION 'x'; END; $$ "
        "LANGUAGE plpgsql; CREATE TABLE t (id int);"
    )

    statements = split_statements(sql)

    assert len(statements) == 2
    assert statements[0].endswith("LANGUAGE plpgsql")
    assert statements[1] == "CREATE TABLE t (id int)"


def test_a_semicolon_inside_a_literal_or_comment_does_not_split() -> None:
    statements = split_statements("INSERT INTO t VALUES ('a;b'); SELECT 1 -- why;\n")

    assert len(statements) == 2
    assert "'a;b'" in statements[0]


def test_the_generated_schema_splits_into_its_parts() -> None:
    statements = split_statements(postgres_schema(SCHEMA))

    assert sum(1 for s in statements if s.startswith("CREATE TABLE")) == 31
    assert sum(1 for s in statements if "audit_events_append_only" in s) >= 1


def test_a_server_database_is_created_statement_by_statement() -> None:
    conn = _FakeServer()

    init_db(conn)

    sent = "\n".join(sql for sql, _ in conn.statements)
    assert "CREATE TABLE IF NOT EXISTS tasks" in sent
    assert "audit_events_append_only" in sent
    assert "AUTOINCREMENT" not in sent
    assert len(conn.statements) > 30
    last_sql, last_params = conn.statements[-1]
    assert last_sql.startswith("INSERT INTO schema_meta")
    assert last_params == ("version", "1")


def test_a_server_transaction_does_not_take_sqlite_s_write_lock() -> None:
    conn = _FakeServer()

    with transaction(conn):
        pass

    assert [sql for sql, _ in conn.statements] == ["BEGIN", "COMMIT"]


def test_a_sqlite_transaction_still_opens_immediately(tmp_path: Path) -> None:
    conn = open_target(parse(tmp_path / "tx.db"))
    try:
        with transaction(conn):
            conn.execute("CREATE TABLE t (id INTEGER)")
        assert conn.execute("SELECT COUNT(*) AS n FROM t").fetchone()["n"] == 0
    finally:
        conn.close()


def test_no_statement_uses_a_sqlite_only_date_function() -> None:
    """`strftime` lives in SQL and in Python; only the Python one is portable.

    SQLite's `strftime()` has no equivalent in a server query, so a call site that
    reaches SQL has to be found before a cluster does. The only place it is used
    now is a console label, which is Python's own formatter.
    """
    source = Path(__file__).resolve().parents[1] / "src" / "internal_ai"
    allowed = {Path("console/app.py"), Path("console/lists.py")}
    offenders = [
        path.relative_to(source)
        for path in source.rglob("*.py")
        if "strftime(" in path.read_text(encoding="utf-8")
        and path.relative_to(source) not in allowed
    ]

    assert offenders == [], f"strftime outside a Python label: {offenders}"
    assert "strftime" not in postgres_schema(SCHEMA)


def test_the_home_variable_still_decides_where_the_file_goes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "elsewhere"
    monkeypatch.setenv("INTERNAL_AI_HOME", str(home))
    monkeypatch.delenv(dialects.ENV_DATABASE_URL, raising=False)

    with Platform.create(root=tmp_path) as platform:
        assert platform.settings.db_path == home / "internal_ai.db"
