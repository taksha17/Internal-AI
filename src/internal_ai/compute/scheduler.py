"""Compute plane: job scheduling with quotas, priorities and cost caps.

The PRD keeps the compute decision open in section 10 and defines only what
the platform needs. This module implements that contract: a scheduler that
enforces per-job cost caps, per-agent quotas and priority ordering, and runs
jobs through a pluggable backend. A local backend executes jobs in the sandbox
so the platform works without a cluster; a Slurm or Kubernetes backend slots
in behind the same interface.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from internal_ai.audit import AuditLog
from internal_ai.budget import BudgetManager
from internal_ai.domain import AuditOutcome
from internal_ai.ids import new_id, now_iso
from internal_ai.sandbox import Sandbox
from internal_ai.training.cost import CostEstimate, estimate_cpu_run, estimate_gpu_run


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class JobSpec:
    name: str
    command: list[str]
    agent: str
    tokens: int = 0
    parameters: int = 0
    priority: int = 5
    quota: str = "default"
    cost_cap_usd: float = 5.0
    device: str = "cpu"
    environment: dict[str, str] = field(default_factory=dict)
    workdir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "agent": self.agent,
            "tokens": self.tokens,
            "parameters": self.parameters,
            "priority": self.priority,
            "quota": self.quota,
            "cost_cap_usd": self.cost_cap_usd,
            "device": self.device,
            "workdir": self.workdir,
        }


@dataclass(slots=True)
class Job:
    id: str
    spec: JobSpec
    state: JobState = JobState.QUEUED
    estimate: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "spec": self.spec.to_dict(),
            "state": str(self.state),
            "estimate": self.estimate,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "rejection_reason": self.rejection_reason,
        }


class JobBackend(Protocol):
    name: str

    def submit(self, job: Job, sandbox: Sandbox) -> dict[str, Any]: ...


class LocalBackend:
    """Runs the job in a sandboxed subprocess on this machine."""

    name = "local"

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def submit(self, job: Job, sandbox: Sandbox) -> dict[str, Any]:
        workdir = sandbox.prepare(job.id)
        result = sandbox.run(job.spec.command, workdir=workdir, env=job.spec.environment)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "duration_s": result.duration_s,
        }


class SlurmBackend:
    """Submits to Slurm through sbatch.

    Requires a reachable scheduler. The class is the integration point for
    the compute decision in PRD section 10; it is not exercised in local runs.
    """

    name = "slurm"

    def __init__(self, partition: str = "gpu", account: str = "") -> None:
        self.partition = partition
        self.account = account

    def submit(self, job: Job, sandbox: Sandbox) -> dict[str, Any]:
        script = Path(job.spec.workdir or ".") / f"{job.id}.sbatch"
        script.parent.mkdir(parents=True, exist_ok=True)
        account = f"#SBATCH --account={self.account}\n" if self.account else ""
        script.write_text(
            f"#!/bin/bash\n#SBATCH --job-name={job.spec.name}\n"
            f"#SBATCH --partition={self.partition}\n{account}"
            f"#SBATCH --output={script.with_suffix('.out')}\n"
            f"set -euo pipefail\n{' '.join(job.spec.command)}\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["sbatch", str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if proc.returncode != 0:
            return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        return {"exit_code": 0, "stdout": proc.stdout, "stderr": proc.stderr, "script": str(script)}


class JobScheduler:
    """Queues jobs, enforces caps and quotas, and orders by priority."""

    def __init__(
        self,
        audit: AuditLog,
        budget: BudgetManager,
        sandbox: Sandbox,
        backend: JobBackend | None = None,
    ) -> None:
        self.audit = audit
        self.budget = budget
        self.sandbox = sandbox
        self.backend = backend or LocalBackend(sandbox)
        self.jobs: dict[str, Job] = {}
        self.quota_limits: dict[str, int] = {}

    def set_quota(self, quota: str, max_concurrent: int) -> None:
        self.quota_limits[quota] = max_concurrent

    def estimate(self, spec: JobSpec) -> CostEstimate:
        if spec.device == "cpu":
            return estimate_cpu_run(spec.parameters, spec.tokens, tokens_per_second=20_000.0)
        return estimate_gpu_run(spec.parameters, spec.tokens)

    def submit(self, spec: JobSpec, actor: str = "infra_engineer") -> Job:
        job = Job(id=new_id("job"), spec=spec, created_at=now_iso())
        estimate = self.estimate(spec)
        job.estimate = estimate.to_dict()

        decision = self.budget.check(spec.agent, job_id=job.id)
        if not decision.allowed:
            job.state = JobState.REJECTED
            job.rejection_reason = decision.reason
            self.jobs[job.id] = job
            self.audit.record(
                actor=actor,
                action="job.rejected",
                subject=job.id,
                outcome=AuditOutcome.DENIED,
                details={"reason": decision.reason, "spec": spec.to_dict()},
            )
            return job
        if estimate.estimated_cost_usd > spec.cost_cap_usd:
            job.state = JobState.REJECTED
            job.rejection_reason = (
                f"estimated cost {estimate.estimated_cost_usd:.4f} USD exceeds the job cap "
                f"{spec.cost_cap_usd:.4f} USD"
            )
            self.jobs[job.id] = job
            self.audit.record(
                actor=actor,
                action="job.rejected",
                subject=job.id,
                outcome=AuditOutcome.DENIED,
                details={"reason": job.rejection_reason, "estimate": job.estimate},
            )
            return job

        self.jobs[job.id] = job
        self.audit.record(
            actor=actor,
            action="job.submitted",
            subject=job.id,
            outcome=AuditOutcome.REQUESTED,
            details={"spec": spec.to_dict(), "estimate": job.estimate},
        )
        return job

    def queue(self) -> list[Job]:
        return sorted(
            (job for job in self.jobs.values() if job.state == JobState.QUEUED),
            key=lambda job: (-job.spec.priority, job.created_at, job.id),
        )

    def running_count(self, quota: str) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.state == JobState.RUNNING and job.spec.quota == quota
        )

    def run_next(self, actor: str = "infra_engineer") -> Job | None:
        for job in self.queue():
            limit = self.quota_limits.get(job.spec.quota)
            if limit is not None and self.running_count(job.spec.quota) >= limit:
                continue
            return self.run(job, actor=actor)
        return None

    def run(self, job: Job, actor: str = "infra_engineer") -> Job:
        job.state = JobState.RUNNING
        job.started_at = now_iso()
        result = self.backend.submit(job, self.sandbox)
        job.exit_code = int(result.get("exit_code", 1))
        job.stdout = str(result.get("stdout", ""))
        job.stderr = str(result.get("stderr", ""))
        job.state = JobState.SUCCEEDED if job.exit_code == 0 else JobState.FAILED
        job.finished_at = now_iso()
        spend = float(job.estimate.get("estimated_cost_usd", 0.0))
        if spend > 0:
            self.budget.record_spend(
                spend, agent=job.spec.agent, job_id=job.id, notes=f"job:{job.spec.name}"
            )
        self.audit.record(
            actor=actor,
            action="job.finished",
            subject=job.id,
            outcome=AuditOutcome.SUCCESS
            if job.state == JobState.SUCCEEDED
            else AuditOutcome.FAILURE,
            details={
                "exit_code": job.exit_code,
                "state": str(job.state),
                "stderr_tail": job.stderr[-500:],
            },
            cost_usd=spend,
        )
        return job

    def cancel(self, job_id: str, actor: str) -> Job:
        job = self.jobs[job_id]
        if job.state in (JobState.SUCCEEDED, JobState.FAILED):
            raise ValueError(f"job {job_id} already finished")
        job.state = JobState.CANCELLED
        job.finished_at = now_iso()
        self.audit.record(
            actor=actor,
            actor_kind="human",
            action="job.cancelled",
            subject=job_id,
            details={"reason": "cancelled by request"},
        )
        return job

    def list_jobs(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda job: (job.created_at, job.id))


def python_command(script: str) -> list[str]:
    return [sys.executable, "-c", script]


class KubernetesBackend:
    """Submits jobs to Kubernetes via kubectl as a Job.

    Integration point for containerized cluster execution.
    """

    name = "kubernetes"

    def __init__(self, namespace: str = "internal-ai", image: str = "python:3.12-slim") -> None:
        self.namespace = namespace
        self.image = image

    def submit(self, job: Job, sandbox: Sandbox) -> dict[str, Any]:
        import yaml

        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": f"internal_ai-{job.id}", "namespace": self.namespace},
            "spec": {
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "runner",
                                "image": self.image,
                                "command": job.spec.command,
                                "env": [
                                    {"name": k, "value": v} for k, v in job.spec.environment.items()
                                ],
                            }
                        ],
                    }
                }
            },
        }
        manifest_path = Path(job.spec.workdir or ".") / f"{job.id}.job.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        proc = subprocess.run(
            ["kubectl", "apply", "-f", str(manifest_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if proc.returncode != 0:
            return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        return {
            "exit_code": 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "manifest": str(manifest_path),
        }
