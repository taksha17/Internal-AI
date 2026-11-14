"""Kubernetes placement is rendered exactly and runs where a cluster exists.

The Job manifest is golden-tested so cluster placement does not drift. Object
names are sanitised to RFC 1123, GPU jobs request a GPU, the pod cannot escalate
or write its own root, and its egress policy allows nothing. A missing kubectl
is a refusal before anything is written. The live tests run only where
``INTERNAL_AI_LIVE_K8S=1`` and a cluster answers.
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
    job_document, policy_document = backend.render_manifest(job).split("---\n")

    assert job_document == (
        "apiVersion: batch/v1\n"
        "kind: Job\n"
        "metadata:\n"
        "  name: internal-ai-job-test\n"
        "  namespace: internal-ai\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      restartPolicy: Never\n"
        "      automountServiceAccountToken: false\n"
        "      runtimeClassName: gvisor\n"
        "      securityContext:\n"
        "        runAsNonRoot: true\n"
        "        runAsUser: 65532\n"
        "        runAsGroup: 65532\n"
        "        fsGroup: 65532\n"
        "        seccompProfile:\n"
        "          type: RuntimeDefault\n"
        "      containers:\n"
        "        - name: runner\n"
        "          image: python:3.12-slim\n"
        f"          command: {json.dumps(job.spec.command)}\n"
        "          env:\n"
        "            - name: A_KEY\n"
        f"              value: {json.dumps('a')}\n"
        "            - name: B_KEY\n"
        f"              value: {json.dumps('b value')}\n"
        "          securityContext:\n"
        "            allowPrivilegeEscalation: false\n"
        "            readOnlyRootFilesystem: true\n"
        "            capabilities:\n"
        "              drop:\n"
        "                - ALL\n"
        "          resources:\n"
        "            requests:\n"
        '              nvidia.com/gpu: "1"\n'
        "            limits:\n"
        '              nvidia.com/gpu: "1"\n'
        "          volumeMounts:\n"
        "            - name: scratch\n"
        "              mountPath: /tmp\n"
        "      volumes:\n"
        "        - name: scratch\n"
        "          emptyDir: {}\n"
    )
    assert policy_document == (
        "apiVersion: networking.k8s.io/v1\n"
        "kind: NetworkPolicy\n"
        "metadata:\n"
        "  name: internal-ai-job-test-egress\n"
        "  namespace: internal-ai\n"
        "spec:\n"
        "  podSelector:\n"
        "    matchLabels:\n"
        "      job-name: internal-ai-job-test\n"
        "  policyTypes:\n"
        "    - Egress\n"
        "  egress: []\n"
    )


def test_the_job_pod_cannot_escalate_or_write_its_own_root(tmp_path) -> None:
    """A trainer reads inputs and writes checkpoints; every other permission is attack surface."""
    manifest = KubernetesBackend().render_manifest(_job(tmp_path))

    assert "allowPrivilegeEscalation: false" in manifest
    assert "readOnlyRootFilesystem: true" in manifest
    assert "runAsNonRoot: true" in manifest
    assert "automountServiceAccountToken: false" in manifest
    assert "                - ALL" in manifest
    # A read-only root without somewhere to write would fail every trainer.
    assert "mountPath: /tmp" in manifest


def test_the_egress_policy_allows_nothing(tmp_path) -> None:
    """An empty egress rule list is a deny; omitting the object would allow everything."""
    policy = KubernetesBackend().egress_policy(_job(tmp_path))

    assert "  egress: []\n" in policy
    assert "  policyTypes:\n    - Egress\n" in policy
    assert "      job-name: internal-ai-job-test\n" in policy


def test_the_runtime_class_is_configurable_and_can_be_absent(tmp_path) -> None:
    chosen = KubernetesBackend(runtime_class="kata").render_manifest(_job(tmp_path))
    assert "      runtimeClassName: kata\n" in chosen

    unisolated = KubernetesBackend(runtime_class=None).render_manifest(_job(tmp_path))
    assert "runtimeClassName" not in unisolated


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


def test_every_live_test_is_gated_on_the_switch_not_on_kubectl_alone() -> None:
    """kubectl is on every hosted runner, so a cluster probe may not stand in for the switch.

    Without this, a live test runs on a machine that has the client and no
    cluster, fails, and blocks every pull request behind a red check.
    """
    module = sys.modules[__name__]
    live = [getattr(module, name) for name in dir(module) if name.startswith("test_live_")]
    assert live
    for test in live:
        reasons = [
            mark.kwargs.get("reason", "") for mark in test.pytestmark if mark.name == "skipif"
        ]
        assert any("INTERNAL_AI_LIVE_K8S" in reason for reason in reasons), test.__name__


# A single-node proof cluster has no gVisor RuntimeClass, so the live tests ask
# for the placement without it; the isolation claims are tested in the manifest.
@pytest.mark.skipif(
    not _live_cluster_answers(), reason="needs INTERNAL_AI_LIVE_K8S=1 and a cluster"
)
def test_live_submit_when_a_cluster_exists(tmp_path) -> None:
    _kubectl("create", "namespace", "internal-ai")
    result = KubernetesBackend(runtime_class=None).submit(_job(tmp_path), sandbox=None)
    assert result["exit_code"] == 0


@pytest.mark.skipif(
    not _live_cluster_answers(), reason="needs INTERNAL_AI_LIVE_K8S=1 and a cluster"
)
def test_live_job_runs_to_completion_and_logs_output(tmp_path) -> None:
    backend = KubernetesBackend(runtime_class=None)
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
        _kubectl("delete", "networkpolicy", f"{name}-egress", "-n", backend.namespace)
