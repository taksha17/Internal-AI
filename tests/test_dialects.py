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
from internal_ai.dialects import DatabaseDenied, Target, describe, open_target, parse
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


def test_postgresql_is_refused_with_the_reason() -> None:
    with pytest.raises(DatabaseDenied) as refused:
        parse("postgresql://user:secret@db.internal:5432/internal_ai")

    assert "not ported" in str(refused.value)
    assert "INTERNAL_AI_DATABASE_URL" in str(refused.value)
    assert "secret" not in str(refused.value)


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
    monkeypatch.setenv(dialects.ENV_DATABASE_URL, "postgresql://u:p@localhost/ai")

    with pytest.raises(DatabaseDenied):
        Platform.create(root=tmp_path)


def test_a_platform_still_opens_the_default_database(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path) as platform:
        assert platform.store.target.is_sqlite
        assert platform.settings.db_path.exists()


def test_the_home_variable_still_decides_where_the_file_goes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "elsewhere"
    monkeypatch.setenv("INTERNAL_AI_HOME", str(home))
    monkeypatch.delenv(dialects.ENV_DATABASE_URL, raising=False)

    with Platform.create(root=tmp_path) as platform:
        assert platform.settings.db_path == home / "internal_ai.db"
