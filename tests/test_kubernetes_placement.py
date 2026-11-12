"""Kubernetes placement is rendered exactly and runs where a cluster exists.

The Job manifest is golden-tested so cluster placement does not drift. Object
names are sanitised to RFC 1123, GPU jobs request a GPU, and a missing
kubectl is a refusal before anything is written. The live tests run only
where ``INTERNAL_AI_LIVE_K8S=1`` and a cluster answers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from internal_ai.compute.scheduler import Job, JobSpec, KubernetesBackend


def _job(tmp_path, device="gpu") -> Job:
    spec = JobSpec(
        name="nightly-replay",
        command=[sys.executable, "-c", "print('hello world')"],
        agent="infra_engineer",
        tokens=1000,
        parameters=1000,
        device=device,
        environment={"B_KEY": "b value", "A_KEY": "a"},
        workdir=str(tmp_path),
    )
    return Job(id="job_test", spec=spec, created_at="now")


def test_manifest_render_is_golden(tmp_path) -> None:
    job = _job(tmp_path)
    backend = KubernetesBackend(namespace="internal-ai", image="python:3.12-slim")

    assert backend.render_manifest(job) == (
        "apiVersion: batch/v1\n"
        "kind: Job\n"
        "metadata:\n"
        "  name: internal-ai-job-test\n"
        "  namespace: internal-ai\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      restartPolicy: Never\n"
        "      containers:\n"
        "        - name: runner\n"
        "          image: python:3.12-slim\n"
        f"          command: {json.dumps(job.spec.command)}\n"
        "          env:\n"
        "            - name: A_KEY\n"
        f"              value: {json.dumps('a')}\n"
        "            - name: B_KEY\n"
        f"              value: {json.dumps('b value')}\n"
        "          resources:\n"
        "            requests:\n"
        '              nvidia.com/gpu: "1"\n'
        "            limits:\n"
        '              nvidia.com/gpu: "1"\n'
    )


def test_cpu_jobs_request_no_gpu(tmp_path) -> None:
    manifest = KubernetesBackend().render_manifest(_job(tmp_path, device="cpu"))

    assert "nvidia.com/gpu" not in manifest


def test_a_missing_kubectl_is_a_refusal_before_anything_is_written(tmp_path, monkeypatch) -> None:
    job = _job(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert shutil.which("kubectl") is None

    with pytest.raises(RuntimeError, match="kubectl"):
        KubernetesBackend().submit(job, sandbox=None)
    assert list(tmp_path.glob("*.job.yaml")) == []


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="needs a Kubernetes cluster")
def test_live_submit_when_a_cluster_exists(tmp_path) -> None:
    _kubectl("create", "namespace", "internal-ai")
    result = KubernetesBackend().submit(_job(tmp_path), sandbox=None)
    assert result["exit_code"] == 0


def _kubectl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _live_cluster_answers() -> bool:
    if os.environ.get("INTERNAL_AI_LIVE_K8S") != "1":
        return False
    if shutil.which("kubectl") is None:
        return False
    return _kubectl("cluster-info").returncode == 0


@pytest.mark.skipif(
    not _live_cluster_answers(), reason="needs INTERNAL_AI_LIVE_K8S=1 and a cluster"
)
def test_live_job_runs_to_completion_and_logs_output(tmp_path) -> None:
    backend = KubernetesBackend()
    spec = JobSpec(
        name="live-proof",
        command=["python", "-c", "print('hello world')"],
        agent="infra_engineer",
        tokens=1,
        parameters=1,
        device="cpu",
        environment={},
        workdir=str(tmp_path),
    )
    job = Job(id="job_live_proof", spec=spec, created_at="now")
    name = backend.object_name(job)
    _kubectl("create", "namespace", backend.namespace)
    _kubectl("delete", "job", name, "-n", backend.namespace)

    applied = backend.submit(job, sandbox=None)
    assert applied["exit_code"] == 0, applied["stderr"]

    try:
        waited = _kubectl(
            "wait",
            "--for=condition=complete",
            f"job/{name}",
            "-n",
            backend.namespace,
            "--timeout=300s",
        )
        assert waited.returncode == 0, waited.stderr
        logs = _kubectl("logs", f"job/{name}", "-n", backend.namespace)
        assert "hello world" in logs.stdout
    finally:
        _kubectl("delete", "job", name, "-n", backend.namespace)
