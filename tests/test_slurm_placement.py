"""Slurm placement is rendered exactly, quoted safely, and never pretends to run.

The sbatch script is golden-tested so cluster placement does not drift. A job
command is shell-quoted, GPU jobs request a GPU, and a missing scheduler is a
refusal before anything is written. The live test runs only where
``INTERNAL_AI_LIVE_SLURM=1`` and a scheduler answers, and it names the
partition and account from the environment, because a cluster's shape is the
cluster's business: a single-node lab has neither a ``gpu`` partition nor an
accounting association, and a test that hardcodes them fails on every real
install rather than on a real defect.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time

import pytest

from internal_ai.compute.scheduler import Job, JobSpec, SlurmBackend


def _job(tmp_path, device="gpu", account="ai") -> Job:
    spec = JobSpec(
        name="nightly-replay",
        command=[sys.executable, "-c", "print('hello world')"],
        agent="infra_engineer",
        tokens=1000,
        parameters=1000,
        device=device,
        workdir=str(tmp_path),
    )
    return Job(id="job_test", spec=spec, created_at="now"), account


def test_sbatch_render_is_golden(tmp_path) -> None:
    job, account = _job(tmp_path)
    backend = SlurmBackend(partition="gpu", account=account)

    assert backend.render_sbatch(job) == (
        "#!/bin/bash\n"
        "#SBATCH --job-name=nightly-replay\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --account=ai\n"
        "#SBATCH --gres=gpu:1\n"
        f"#SBATCH --output={tmp_path / 'job_test.out'}\n"
        "set -euo pipefail\n"
        f"{shlex.join(job.spec.command)}\n"
    )


def test_cpu_jobs_request_no_gpu(tmp_path) -> None:
    job, _account = _job(tmp_path, device="cpu", account="")
    backend = SlurmBackend(partition="cpu", account="")

    script = backend.render_sbatch(job)
    assert "--gres=" not in script
    assert "--account=" not in script


def test_a_missing_scheduler_is_a_refusal_before_anything_is_written(tmp_path, monkeypatch) -> None:
    job, account = _job(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert shutil.which("sbatch") is None

    with pytest.raises(RuntimeError, match="sbatch"):
        SlurmBackend(partition="gpu", account=account).submit(job, sandbox=None)
    assert list(tmp_path.glob("*.sbatch")) == []


def _live_scheduler_answers() -> bool:
    if os.environ.get("INTERNAL_AI_LIVE_SLURM") != "1":
        return False
    if shutil.which("sbatch") is None or shutil.which("sinfo") is None:
        return False
    return subprocess.run(["sinfo"], capture_output=True, timeout=60, check=False).returncode == 0


@pytest.mark.skipif(
    not _live_scheduler_answers(), reason="needs INTERNAL_AI_LIVE_SLURM=1 and a scheduler"
)
def test_live_job_is_accepted_by_a_real_scheduler(tmp_path) -> None:
    job, _account = _job(tmp_path, device="cpu", account="")
    backend = SlurmBackend(
        partition=os.environ.get("INTERNAL_AI_SLURM_PARTITION", "debug"), account=""
    )
    result = backend.submit(job, sandbox=None)
    assert result["exit_code"] == 0, result["stderr"]
    assert "Submitted batch job" in result["stdout"]


@pytest.mark.skipif(
    not _live_scheduler_answers(), reason="needs INTERNAL_AI_LIVE_SLURM=1 and a scheduler"
)
def test_live_job_runs_to_completion_and_writes_its_output(tmp_path) -> None:
    output = tmp_path / "job_live_proof.out"
    spec = JobSpec(
        name="live-proof",
        command=["sh", "-c", "echo SLURM_LIVE_PROOF_OK; nproc"],
        agent="infra_engineer",
        tokens=1,
        parameters=1,
        device="cpu",
        workdir=str(tmp_path),
    )
    job = Job(id="job_live_proof", spec=spec, created_at="now")
    backend = SlurmBackend(partition=os.environ.get("INTERNAL_AI_SLURM_PARTITION", "debug"))

    submitted = backend.submit(job, sandbox=None)
    assert submitted["exit_code"] == 0, submitted["stderr"]
    job_id = submitted["stdout"].split()[-1]

    deadline = time.monotonic() + 180
    state = ""
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        state = probe.stdout.strip()
        if not state:
            break
        time.sleep(3)
    assert state == "", f"job {job_id} never left the queue: {state}"
    assert output.exists(), f"job {job_id} wrote no output at {output}"
    assert "SLURM_LIVE_PROOF_OK" in output.read_text(encoding="utf-8")
