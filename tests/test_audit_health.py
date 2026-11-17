"""A broken chain is announced at open, not discovered by whoever reads the page next.

The trigger that makes `audit_events` append-only can be dropped by anyone with
the file, so this is the case that matters: the rows were edited, the triggers are
gone, and the platform is started again on top of it.
"""

from __future__ import annotations

from pathlib import Path

from internal_ai.context import Platform


def test_health_is_the_last_verification(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        first = platform.audit.health()
        second = platform.audit.health()

        assert first is second, "a page render must not walk the chain again"
        assert first.ok


def test_a_tampered_chain_is_alerted_when_the_platform_opens(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        platform.audit.record(actor="a", action="test.one", subject="x")
        # What an attacker with the database file does first.
        platform.store.conn.execute("DROP TRIGGER audit_events_no_update")
        platform.store.conn.execute("UPDATE audit_events SET subject = 'tampered'")

    reopened = Platform.create(root=tmp_path, llm_kind="scripted")
    try:
        health = reopened.audit.health()
        assert not health.ok
        assert "hash mismatch" in health.reason

        alerts = reopened.store.conn.execute(
            "SELECT kind, severity, subject, message FROM alerts WHERE kind = 'audit_integrity'"
        ).fetchall()
        assert alerts, "an integrity failure has to leave an alert, not just a banner"
        assert alerts[0]["severity"] == "critical"
        assert "broken" in alerts[0]["message"]
    finally:
        reopened.close()


def test_a_check_is_due_until_one_has_run(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        platform.audit.verify()

        assert not platform.audit.check_due(24.0)
        assert platform.audit.check_due(0.0)
        assert platform.store.kv_get("audit_checked_ok", "") == "1"


def test_an_unreadable_stamp_is_due(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        platform.store.kv_set("audit_checked_at", "last tuesday")

        assert platform.audit.check_due(24.0), "a stamp nobody can parse proves nothing"


def test_a_clean_install_raises_no_integrity_alert(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        platform.audit.record(actor="a", action="test.one", subject="x")
        rows = platform.store.conn.execute(
            "SELECT count(*) AS n FROM alerts WHERE kind = 'audit_integrity'"
        ).fetchone()

        assert rows["n"] == 0
