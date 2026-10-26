"""Operator secrets and a backup that can be checked without replacing the live database."""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.cli import main
from internal_ai.llm import AnthropicAdapter
from internal_ai.sandbox import scrub_env
from internal_ai.secrets import (
    ANTHROPIC_API_KEY,
    CONTROL_PLANE_KEY,
    operator_secret,
    write_operator_secrets,
)


def test_secrets_file_beats_the_environment_and_stays_out_of_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-environment")
    monkeypatch.setenv("INTERNAL_AI_API_KEY", "env-control-plane")
    write_operator_secrets(
        tmp_path,
        {ANTHROPIC_API_KEY: "from-the-file", CONTROL_PLANE_KEY: "file-control-plane"},
    )
    mode = (tmp_path / "secrets.json").stat().st_mode & 0o777
    assert mode == 0o600
    assert operator_secret(tmp_path, ANTHROPIC_API_KEY) == "from-the-file"
    assert operator_secret(tmp_path, CONTROL_PLANE_KEY) == "file-control-plane"
    assert scrub_env({"ANTHROPIC_API_KEY": "from-the-file", "PATH": "/usr/bin"}) == {
        "PATH": "/usr/bin"
    }
    adapter = AnthropicAdapter(api_key="from-the-file")
    with pytest.raises(RuntimeError, match="operator secret"):
        adapter.complete("the key is from-the-file")


def test_missing_secrets_file_keeps_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERNAL_AI_API_KEY", "env-control-plane")
    assert operator_secret(tmp_path, CONTROL_PLANE_KEY) == "env-control-plane"


def test_backup_restore_verifies_the_audit_chain(tmp_path: Path) -> None:
    root = tmp_path / "live"
    backup = tmp_path / "backup.db"
    assert main(["--root", str(root), "--llm", "scripted", "init"]) == 0
    live = root / "var" / "internal_ai.db"
    assert (
        main(["--root", str(root), "--llm", "scripted", "db", "backup", "--out", str(backup)]) == 0
    )
    before = live.read_bytes()
    assert (
        main(
            [
                "--root",
                str(root),
                "--llm",
                "scripted",
                "db",
                "restore",
                "--check",
                "--from",
                str(backup),
            ]
        )
        == 0
    )
    assert live.read_bytes() == before
