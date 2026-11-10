"""Slurm placement is rendered exactly, quoted safely, and never pretends to run.

The sbatch script is golden-tested so cluster placement does not drift. A job
command is shell-quoted, GPU jobs request a GPU, and a missing scheduler is a
refusal before anything is written. The live submit runs only where Slurm exists.
"""

from __future__ import annotations

import shlex
import shutil
import sys

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


@pytest.mark.skipif(shutil.which("sbatch") is None, reason="needs a Slurm scheduler")
def test_live_submit_when_slurm_exists(tmp_path) -> None:
    job, account = _job(tmp_path)
    result = SlurmBackend(partition="gpu", account=account).submit(job, sandbox=None)
    assert result["exit_code"] == 0
