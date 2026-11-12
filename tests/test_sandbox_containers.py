"""Production runs agent code in containers, not bare subprocesses.

ContainerSandbox renders an exact docker command: network none, memory, cpu
and pids limits, the workdir mounted at /workspace, and the same allowlisted
environment. It refuses instead of silently downgrading when docker is missing
and strict mode is on, and the backend used is recorded on the result.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from internal_ai.sandbox import (
    ContainerSandbox,
    Sandbox,
    SandboxDenied,
    make_sandbox,
)

IMAGE = "python:3.12-slim"


def fixed_environ(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        os,
        "environ",
        {
            "PATH": "/bin",
            "HOME": "/home/someone",
            "SECRET_TOKEN": "must-never-leak",
        },
    )


def docker_ok(monkeypatch, argv_log: list) -> MagicMock:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = "hi\n"
    proc.stderr = ""
    run = MagicMock(return_value=proc)

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        argv_log.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return run


def test_the_docker_command_is_exact(tmp_path: Path, monkeypatch) -> None:
    fixed_environ(monkeypatch)
    argv_log: list = []
    docker_ok(monkeypatch, argv_log)
    workdir = tmp_path / "job"
    box = ContainerSandbox(tmp_path, image=IMAGE)
    result = box.run(["echo", "hi"], workdir=workdir)
    assert result.exit_code == 0
    assert argv_log, "nothing was executed"
    assert argv_log[0] == [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--memory=4096m",
        "--cpus=1.00",
        "--pids-limit=256",
        "-v",
        f"{workdir}:/workspace:rw",
        "-w",
        "/workspace",
        "-e",
        "PATH=/bin",
        "-e",
        "HOME=/workspace",
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "PYTHONHASHSEED=0",
        "-e",
        "INTERNAL_AI_NETWORK=deny",
        IMAGE,
        "echo",
        "hi",
    ]
    assert result.metadata["backend"] == "docker"


def test_user_env_cannot_reenable_the_network(tmp_path: Path, monkeypatch) -> None:
    fixed_environ(monkeypatch)
    argv_log: list = []
    docker_ok(monkeypatch, argv_log)
    box = ContainerSandbox(tmp_path, image=IMAGE)
    box.run(["echo", "hi"], workdir=tmp_path / "job", env={"INTERNAL_AI_NETWORK": "allow"})
    network_flags = [flag for cmd in argv_log for flag in cmd if flag == "--network"]
    assert network_flags == ["--network"]
    assert "none" in argv_log[0]


def test_strict_mode_refuses_when_docker_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    run = MagicMock(side_effect=AssertionError("must never execute"))
    monkeypatch.setattr(subprocess, "run", run)
    box = ContainerSandbox(tmp_path, image=IMAGE)
    with pytest.raises(SandboxDenied, match="docker"):
        box.run(["echo", "hi"], workdir=tmp_path / "job")


def test_non_strict_mode_falls_back_to_a_subprocess(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    box = ContainerSandbox(tmp_path, image=IMAGE, strict=False)
    result = box.run(["echo", "hi"], workdir=tmp_path / "job")
    assert result.exit_code == 0
    assert result.stdout.strip() == "hi"
    assert result.metadata["backend"] == "subprocess"


def test_the_backend_is_chosen_by_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("INTERNAL_AI_SANDBOX", raising=False)
    assert isinstance(make_sandbox(tmp_path), Sandbox)
    assert not isinstance(make_sandbox(tmp_path), ContainerSandbox)
    monkeypatch.setenv("INTERNAL_AI_SANDBOX", "container")
    box = make_sandbox(tmp_path)
    assert isinstance(box, ContainerSandbox)
    assert box.network == "none"


def docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return probe.returncode == 0


needs_docker = pytest.mark.skipif(
    not docker_ready(), reason="needs a docker daemon with the sandbox image"
)


@needs_docker
def test_a_live_container_runs_and_reports_docker(tmp_path: Path) -> None:
    box = ContainerSandbox(tmp_path, image=IMAGE)
    result = box.run(["python", "-c", "print('hi')"], workdir=tmp_path / "job")
    assert result.exit_code == 0
    assert result.stdout.strip() == "hi"
    assert result.metadata["backend"] == "docker"


@needs_docker
def test_a_live_container_cannot_reach_the_network(tmp_path: Path) -> None:
    box = ContainerSandbox(tmp_path, image=IMAGE)
    result = box.run(
        [
            "python",
            "-c",
            "import socket; socket.create_connection(('8.8.8.8', 53), timeout=5)",
        ],
        workdir=tmp_path / "job",
    )
    assert result.exit_code != 0
