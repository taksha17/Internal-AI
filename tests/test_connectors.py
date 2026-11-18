"""A connector reports an ACL or the source is refused."""

from __future__ import annotations

import pytest
from conftest import enroll
from fastapi.testclient import TestClient
from test_console_e2e import Person

from internal_ai.api import create_app
from internal_ai.data.connectors import AclUnreported, Pulled, pull, require_acl
from internal_ai.services.build import preview_connector
from internal_ai.services.permissions import PermissionDenied


def _tree(root) -> None:
    root.mkdir()
    (root / "notes.md").write_text("the policy is reviewed yearly\n", encoding="utf-8")
    (root / "skip.bin").write_bytes(b"\x00\x01")
    nested = root / "team"
    nested.mkdir()
    (nested / "faq.txt").write_text("ask the owner\n", encoding="utf-8")


def test_git_reports_the_groups_and_a_second_pull_is_empty(tmp_path) -> None:
    tree = tmp_path / "repo"
    _tree(tree)
    rows, cursor = pull("git", str(tree), ["legal", "finance"])
    assert {row.key for row in rows} == {"notes.md", "team/faq.txt"}
    assert all(row.acl == ("legal", "finance") for row in rows)
    again, same = pull("git", str(tree), ["legal", "finance"], cursor)
    assert again == []
    assert same == cursor


def test_an_unreported_acl_is_refused() -> None:
    with pytest.raises(AclUnreported, match="ACL"):
        require_acl([Pulled(key="doc", text="body", acl=None)])


def test_a_remote_connector_refuses_before_it_reads(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    for kind in ("sharepoint", "confluence", "erp"):
        with pytest.raises(AclUnreported, match="account"):
            pull(kind, str(missing), ["legal"])
    assert not missing.exists()


def test_finance_cannot_preview_a_connector(platform, tmp_path) -> None:
    tree = tmp_path / "repo"
    _tree(tree)
    owner = enroll(platform, "Dana Owner", ["data_owner"])
    finance = enroll(platform, "Finn Books", ["finance"])
    seen = preview_connector(platform, owner, "git", str(tree), ["legal"])
    assert seen["records"] == 2
    assert seen["groups"] == ["legal"]
    with pytest.raises(PermissionDenied):
        preview_connector(platform, finance, "git", str(tree), ["legal"])


def test_console_previews_a_git_tree_and_refuses_sharepoint(tmp_path) -> None:
    tree = tmp_path / "repo"
    _tree(tree)
    app = create_app(root=tmp_path / "home", llm_kind="scripted")
    with TestClient(app):
        enroll(app.state.platform, "Dana Owner", ["data_owner"])
        dana = Person(app, "Dana Owner")
        page = dana.get("/console/sources")
        assert "Read a connector" in page
        ok = dana.post("/console/sources/connector", kind="git", path=str(tree), acl="legal")
        assert "connector git reported 2 record(s) for legal" in ok
        refused = dana.post(
            "/console/sources/connector", kind="sharepoint", path=str(tree), acl="legal", ok=False
        )
        assert "cannot report an ACL" in refused
