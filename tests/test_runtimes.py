"""A job may name gVisor, Kata or Firecracker, and no other runtime class."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.compute.runtimes import RuntimeRefused, runtime_for
from internal_ai.compute.scheduler import KubernetesBackend
from internal_ai.sandbox import ContainerSandbox, SandboxDenied
from internal_ai.services.compute_advice import select_runtime
from internal_ai.services.permissions import PermissionDenied


def test_each_class_names_its_handler_and_an_unknown_class_is_refused() -> None:
    assert runtime_for("gvisor").handler == "runsc"
    assert runtime_for("kata").isolation == "virtual machine"
    assert runtime_for("firecracker").handler == "firecracker"
    with pytest.raises(RuntimeRefused, match="unknown runtime class"):
        runtime_for("runc")


def test_the_manifest_names_a_known_class_and_refuses_any_other() -> None:
    from internal_ai.compute.scheduler import Job, JobSpec

    job = Job(
        id="job_runtime",
        spec=JobSpec(name="nightly", command=["python", "-c", "print(1)"], agent="trainer"),
        created_at="now",
    )
    manifest = KubernetesBackend(runtime_class="firecracker").render_job(job)
    assert "      runtimeClassName: firecracker\n" in manifest
    assert "runtimeClassName: gvisor" not in manifest
    with pytest.raises(RuntimeRefused, match="unknown runtime class"):
        KubernetesBackend(runtime_class="privileged")
    absent = KubernetesBackend(runtime_class=None).render_job(job)
    assert "runtimeClassName" not in absent


def test_a_named_runtime_is_the_docker_handler(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")
    argv: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        argv.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return proc

    monkeypatch.setattr(subprocess, "run", fake_run)
    box = ContainerSandbox(tmp_path, runtime="kata")
    result = box.run(["echo", "hi"], workdir=tmp_path / "job")
    assert "--runtime" in argv[0]
    assert argv[0][argv[0].index("--runtime") + 1] == "kata-runtime"
    assert result.metadata["runtime"] == "kata"
    with pytest.raises(RuntimeRefused, match="unknown runtime class"):
        ContainerSandbox(tmp_path, runtime="privileged")


def test_a_named_runtime_does_not_fall_back_to_a_subprocess(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    box = ContainerSandbox(tmp_path, runtime="firecracker", strict=False)
    with pytest.raises(SandboxDenied, match="firecracker"):
        box.run(["echo", "hi"], workdir=tmp_path / "job")


def test_finance_cannot_select_a_runtime(platform) -> None:
    infra = enroll(platform, "Ivy Infra", ["infra_engineer"])
    books = enroll(platform, "Finn Books", ["finance"])
    assert select_runtime(platform, infra, "kata") == "kata"
    events = platform.store.list_audit(action="compute.runtime_selected")
    assert events[0].details["handler"] == "kata-runtime"
    with pytest.raises(PermissionDenied):
        select_runtime(platform, books, "gvisor")


def test_the_compute_page_lists_the_classes_and_records_a_choice(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        enroll(app.state.platform, "Ivy Infra", ["infra_engineer"])
        assert sign_in(client, "Ivy Infra").status_code == 303
        page = client.get("/console/compute")
        assert "kata-runtime" in page.text
        assert "firecracker" in page.text
        chosen = client.post(
            "/console/compute/runtime",
            data={"runtime": "firecracker", "csrf_token": csrf_from(page.text)},
        )
        assert chosen.status_code == 200
        assert "runtime firecracker selected" in chosen.text
