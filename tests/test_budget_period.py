"""Closing a spend period, and the chargeback export a closed period allows.

Two properties matter: a period closes once, and an export only ever describes a
window whose total has stopped moving.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.budget import BudgetPeriodError
from internal_ai.context import Platform
from internal_ai.domain import BudgetScope
from internal_ai.services import governance


def window() -> tuple[str, str]:
    now = datetime.now(UTC)
    return (now - timedelta(hours=1)).isoformat(), (now + timedelta(hours=1)).isoformat()


def a_spend(platform: Platform, amount: float = 1.25) -> None:
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 100.0)
    platform.budget.record_spend(amount, agent="builder", notes="one call")


def test_a_closed_period_freezes_one_total_per_scope(platform: Platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pia Lead", ["program_lead"])
    a_spend(platform, 1.25)
    a_spend(platform, 0.75)
    starts, ends = window()

    period = governance.close_budget_period(platform, actor, starts, ends)

    assert period["totals"]["global:global"] == {"usd": 2.0, "records": 2}
    # One call lands in global, project, agent and weekly; the bill is one of them.
    assert period["billed_usd"] == 2.0
    assert period["scope_total_usd"] == 8.0
    assert platform.budget.periods()[0]["starts_at"] == starts


def test_closing_the_same_window_twice_is_refused(platform: Platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pia Lead", ["program_lead"])
    starts, ends = window()
    governance.close_budget_period(platform, actor, starts, ends)

    with pytest.raises(BudgetPeriodError, match="already closed"):
        governance.close_budget_period(platform, actor, starts, ends)


def test_a_backwards_window_is_not_a_period(platform: Platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pia Lead", ["program_lead"])
    starts, ends = window()

    with pytest.raises(BudgetPeriodError, match="is not a period"):
        governance.close_budget_period(platform, actor, ends, starts)


def test_an_open_period_cannot_be_exported(platform: Platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pia Lead", ["program_lead"])
    starts, _ends = window()

    with pytest.raises(BudgetPeriodError, match="close it before exporting"):
        governance.budget_export(platform, actor, starts)


def test_the_export_carries_the_rows_behind_the_total(platform: Platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pia Lead", ["program_lead"])
    a_spend(platform, 2.5)
    starts, ends = window()
    governance.close_budget_period(platform, actor, starts, ends)

    export = governance.budget_export(platform, actor, starts)

    assert export["totals"]["global:global"]["usd"] == 2.5
    # One call is four scoped rows, which is what a chargeback reads line by line.
    assert len(export["rows"]) == 4
    assert {row["scope"] for row in export["rows"]} == {"global", "project", "agent", "weekly"}
    assert all(row["agent"] == "builder" for row in export["rows"])


def test_closing_a_period_needs_the_role(platform: Platform, enroll_person) -> None:
    outsider = enroll_person(platform, "Dana Expert", ["domain_expert"])
    starts, ends = window()

    with pytest.raises(PermissionError, match="close budget period needs"):
        governance.close_budget_period(platform, outsider, starts, ends)
    with pytest.raises(PermissionError, match="export spend needs"):
        governance.budget_export(platform, outsider, starts)


def test_the_scope_lock_is_a_no_op_on_sqlite(platform: Platform) -> None:
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 10.0)

    # `BEGIN IMMEDIATE` already excludes every writer here; the call exists so the
    # server path is the same code, and it must not attempt a syntax sqlite rejects.
    platform.store.lock_budget_limit(BudgetScope.GLOBAL, "global")
    platform.store.lock_reservations("rsv_none")


@contextmanager
def console(tmp_path: Path) -> Iterator[tuple[TestClient, Any]]:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        yield client, app.state.platform


def test_the_console_closes_a_period(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        lead = enroll(platform, "Pia Lead", ["program_lead"])
        sign_in(client, lead.name)
        a_spend(platform, 3.0)
        starts, ends = window()

        response = client.post(
            "/console/admin/budget/close",
            data={"csrf_token": session_csrf(client), "starts_at": starts, "ends_at": ends},
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "closed, billing $3.0000" in response.text


def test_the_console_exports_a_closed_period_as_csv(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        lead = enroll(platform, "Pia Lead", ["program_lead"])
        sign_in(client, lead.name)
        a_spend(platform, 3.0)
        starts, ends = window()
        client.post(
            "/console/admin/budget/close",
            data={"csrf_token": session_csrf(client), "starts_at": starts, "ends_at": ends},
        )

        response = client.get(f"/console/admin/budget/periods/{starts}.csv")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        lines = response.text.strip().splitlines()
        assert lines[0] == "scope,key,amount_usd,agent,task_id,job_id,created_at"
        # One call, four scopes, ordered by scope name.
        assert len(lines) == 5
        assert any(line.startswith("global,global,3.0,builder,") for line in lines)


def test_an_outsider_gets_a_refusal_instead_of_a_csv(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        lead = enroll(platform, "Pia Lead", ["program_lead"])
        starts, ends = window()
        starts_at = starts
        app_platform = platform
        governance.close_budget_period(
            app_platform, enroll(app_platform, "Pia Two", ["program_lead"]), starts_at, ends
        )
        expert = enroll(platform, "Dana Expert", ["domain_expert"])
        sign_in(client, expert.name)

        response = client.get(
            f"/console/admin/budget/periods/{starts_at}.csv", follow_redirects=True
        )

        assert response.status_code == 200
        assert "export spend needs compliance or program_lead" in response.text
        assert lead.name  # the lead exists; the refusal is about the reader


@pytest.mark.skipif(
    not os.environ.get("INTERNAL_AI_TEST_PG_URL"),
    reason="needs a running PostgreSQL to prove a row lock blocks a second writer",
)
def test_the_server_lock_blocks_a_second_reservation() -> None:
    """The whole point of `FOR UPDATE`: the second caller waits, it does not fit too."""
    import psycopg

    url = os.environ["INTERNAL_AI_TEST_PG_URL"]
    first = psycopg.connect(url, autocommit=True)
    second = psycopg.connect(url, autocommit=False)
    first.execute("CREATE TABLE IF NOT EXISTS budget_lock_probe (scope TEXT PRIMARY KEY, key TEXT)")
    first.execute(
        "INSERT INTO budget_lock_probe (scope, key) VALUES ('global', 'global') "
        "ON CONFLICT (scope) DO NOTHING"
    )
    try:
        first.execute("BEGIN")
        first.execute("SELECT key FROM budget_lock_probe WHERE scope = %s FOR UPDATE", ("global",))
        second.execute("SET statement_timeout = '300ms'")
        with pytest.raises(psycopg.errors.QueryCanceled):
            second.execute(
                "SELECT key FROM budget_lock_probe WHERE scope = %s FOR UPDATE", ("global",)
            )
    finally:
        if not first.closed:
            first.rollback()
        second.rollback()
        first.execute("DROP TABLE IF EXISTS budget_lock_probe")
        first.close()
        second.close()
