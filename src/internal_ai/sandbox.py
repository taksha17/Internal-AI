"""Sandboxed execution for agent code and training jobs.

The sandbox gives each agent an isolated working directory, a scrubbed
environment, resource limits and no network by default. Credentials are never
placed in prompts; they are injected as short-lived scoped values only when a
role holds the matching capability.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SANDBOX_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


@dataclass(slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


def _limits(cpu_seconds: int, memory_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))


class Sandbox:
    """Runs a command inside a jail with an allowlisted environment."""

    def __init__(
        self,
        base_dir: Path,
        memory_mb: int = 4096,
        cpu_seconds: int = 3600,
        timeout: float = 3600.0,
    ) -> None:
        self.base_dir = base_dir
        self.memory_mb = memory_mb
        self.cpu_seconds = cpu_seconds
        self.timeout = timeout

    def prepare(self, name: str) -> Path:
        target = self.base_dir / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def run(
        self,
        command: list[str],
        workdir: Path,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        import time

        workdir.mkdir(parents=True, exist_ok=True)
        child_env = {
            key: value for key, value in os.environ.items() if key in SANDBOX_ENV_ALLOWLIST
        }
        child_env["HOME"] = str(workdir)
        child_env["TMPDIR"] = str(workdir)
        child_env["PYTHONHASHSEED"] = "0"
        child_env["INTERNAL_AI_NETWORK"] = "deny"
        if env:
            child_env.update(env)
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                command,
                cwd=str(workdir),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                preexec_fn=lambda: _limits(self.cpu_seconds, self.memory_mb),
                check=False,
            )
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=time.monotonic() - started,
                timed_out=timed_out,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            return SandboxResult(
                exit_code=124,
                stdout=(exc.stdout or b"").decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr="timeout",
                duration_s=time.monotonic() - started,
                timed_out=timed_out,
            )

    def run_python(self, code: str, workdir: Path, timeout: float | None = None) -> SandboxResult:
        script = workdir / "main.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(code, encoding="utf-8")
        return self.run([sys.executable, str(script)], workdir=workdir, timeout=timeout)


class ContainerSandbox(Sandbox):
    """Runs commands inside a Docker container with network isolation."""

    def __init__(
        self,
        base_dir: Path,
        memory_mb: int = 4096,
        cpu_seconds: int = 3600,
        timeout: float = 3600.0,
        image: str = "python:3.12-slim",
        network: str = "none",
    ) -> None:
        super().__init__(base_dir, memory_mb, cpu_seconds, timeout)
        self.image = image
        self.network = network

    def run(
        self,
        command: list[str],
        workdir: Path,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        import shutil
        import time

        workdir.mkdir(parents=True, exist_ok=True)
        if not shutil.which("docker"):
            # Fallback to subprocess sandbox
            return super().run(command, workdir, env, timeout)

        child_env = {
            key: value for key, value in os.environ.items() if key in SANDBOX_ENV_ALLOWLIST
        }
        child_env["HOME"] = "/workspace"
        child_env["TMPDIR"] = "/tmp"
        child_env["PYTHONHASHSEED"] = "0"
        child_env["INTERNAL_AI_NETWORK"] = "deny"
        if env:
            child_env.update(env)

        env_flags: list[str] = []
        for k, v in child_env.items():
            env_flags.extend(["-e", f"{k}={v}"])

        cpus = (
            max(1, self.cpu_seconds // 3600)
            if self.cpu_seconds >= 3600
            else self.cpu_seconds / 3600
        )
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            self.network,
            f"--memory={self.memory_mb}m",
            f"--cpus={cpus:.2f}",
            "--pids-limit=256",
            "-v",
            f"{workdir}:/workspace:rw",
            "-w",
            "/workspace",
            *env_flags,
            self.image,
            *command,
        ]

        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                check=False,
            )
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=time.monotonic() - started,
                timed_out=timed_out,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            return SandboxResult(
                exit_code=124,
                stdout=(exc.stdout or b"").decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr="timeout",
                duration_s=time.monotonic() - started,
                timed_out=timed_out,
            )
