"""A chain proves the rows it holds; an anchor proves the tail was not deleted.

The tests are the four ways that claim can fail: no key, a forged anchor, an anchor
pointing at an event that is gone, and a person without the role trying to anchor.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai import signing
from internal_ai.api import create_app
from internal_ai.context import Platform
from internal_ai.services.actor import Actor
from internal_ai.services.audit_head import anchor_head, anchor_status, may_anchor
from internal_ai.signing import AuditSigner, SigningRefused


def a_key(tmp_path: Path) -> AuditSigner:
    """The key an install reads on open: ``<home>/audit.key``, where home is var/."""
    return signing.create_key(tmp_path / "var" / "audit.key")


def anchors_on_disk(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "var" / "audit_anchors").glob("*.json"))


@contextmanager
def platform_with_key(tmp_path: Path) -> Iterator[tuple[Platform, AuditSigner]]:
    key = a_key(tmp_path)
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        assert platform.audit.signer is not None, "the key at home/audit.key must be picked up"
        yield platform, key


def test_a_short_key_is_refused() -> None:
    with pytest.raises(SigningRefused, match="at least 32 bytes"):
        AuditSigner(b"too short", "id")


def test_a_key_is_written_once_and_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "var" / "audit.key"
    signing.create_key(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(SigningRefused, match="refusing to replace"):
        signing.create_key(path)


def test_an_install_without_a_key_cannot_anchor(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        assert platform.audit.signer is None
        with pytest.raises(SigningRefused, match="INTERNAL_AI_AUDIT_KEY_FILE"):
            platform.audit.anchor(actor="someone")
        result = platform.audit.verify()
        assert result.ok
        assert result.anchored is False
        assert "no anchor yet" in result.anchor_note


def test_an_anchor_is_written_and_verifies(tmp_path: Path) -> None:
    with platform_with_key(tmp_path) as (platform, key):
        platform.audit.record(actor="a", action="test.one", subject="x")
        anchor = platform.audit.anchor(actor="a")

        files = anchors_on_disk(tmp_path)
        assert len(files) == 1
        stored = json.loads(files[0].read_text())
        assert stored["key_id"] == key.key_id
        assert stored["seq"] == anchor.seq

        result = platform.audit.verify()
        assert result.ok
        assert result.anchored is True
        assert result.anchor_seq == anchor.seq
        assert key.key_id in result.anchor_note
        assert platform.store.list_audit(action="audit.anchored")


def test_a_forged_anchor_is_caught(tmp_path: Path) -> None:
    with platform_with_key(tmp_path) as (platform, _key):
        platform.audit.record(actor="a", action="test.one", subject="x")
        platform.audit.anchor(actor="a")
        path = anchors_on_disk(tmp_path)[0]
        payload = json.loads(path.read_text())
        payload["count"] = 99
        path.write_text(json.dumps(payload))

        result = platform.audit.verify()

        assert not result.ok
        assert "signature" in result.reason


def test_an_anchor_naming_a_missing_event_is_caught(tmp_path: Path) -> None:
    with platform_with_key(tmp_path) as (platform, key):
        platform.audit.record(actor="a", action="test.one", subject="x")
        rolled_back = key.anchor(seq=999, hash_="deadbeef", count=9, signed_at="2026-01-01")
        path = tmp_path / "var" / "audit_anchors" / f"{999:012d}-deadbeef.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rolled_back.to_dict()))

        result = platform.audit.verify()

        assert not result.ok
        assert "truncated or rewritten" in result.reason


def test_anchoring_needs_the_evidence_roles(tmp_path: Path) -> None:
    with platform_with_key(tmp_path) as (platform, _key):
        analyst = Actor("a1", "Analyst", frozenset({"analyst"}), frozenset(), True)
        compliance = Actor("c1", "Cora", frozenset({"compliance"}), frozenset(), True)
        assert may_anchor(analyst) is False
        assert may_anchor(compliance) is True

        with pytest.raises(Exception, match="anchor audit"):
            anchor_head(platform, analyst)
        denials = platform.store.list_audit(action="console.denied")
        assert denials and str(denials[0].outcome) == "denied"

        record = anchor_head(platform, compliance)
        assert record["seq"] >= 1


def test_the_console_anchors_for_a_compliance_person(tmp_path: Path) -> None:
    a_key(tmp_path)
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        platform = app.state.platform
        assert platform.audit.signer is not None
        cora = enroll(platform, "Cora Compliance", ["compliance"])
        assert sign_in(client, cora.name).status_code == 303

        page = client.get("/console/audit")
        assert page.status_code == 200
        assert "Anchor the current head" in page.text

        response = client.post(
            "/console/audit/anchor",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=False,
        )

        assert response.status_code == 303
        assert anchors_on_disk(tmp_path)
        after = client.get("/console/audit")
        assert "anchored and signed" in after.text


def test_the_console_refuses_to_anchor_for_an_analyst(tmp_path: Path) -> None:
    a_key(tmp_path)
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        platform = app.state.platform
        anna = enroll(platform, "Anna Analyst", ["analyst"])
        assert sign_in(client, anna.name).status_code == 303

        page = client.get("/console/audit")
        assert "Anchoring needs security or compliance" in page.text

        client.post(
            "/console/audit/anchor",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=False,
        )

        assert not anchors_on_disk(tmp_path)
        denials = platform.store.list_audit(action="console.denied")
        assert denials and str(denials[0].outcome) == "denied"


def test_the_page_says_why_an_unsigned_install_cannot_anchor(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        platform = app.state.platform
        cora = enroll(platform, "Cora Compliance", ["compliance"])
        assert sign_in(client, cora.name).status_code == 303

        page = client.get("/console/audit")

        assert "No signing key on this install" in page.text
        assert "Anchor the current head" not in page.text


def test_status_reports_the_directory_and_the_key(tmp_path: Path) -> None:
    with platform_with_key(tmp_path) as (platform, key):
        platform.audit.record(actor="a", action="test.one", subject="x")
        platform.audit.anchor(actor="a")

        status: dict[str, Any] = anchor_status(platform)

        assert status["signing"] is True
        assert status["key_id"] == key.key_id
        assert status["anchors"] == 1
        assert status["latest_seq"] >= 1
        assert "audit_anchors" in status["directory"]
