"""What a dataset needs before a from-scratch run, and which machine can take it.

The hours use the same 6 x parameters x tokens figure as the rest of the
platform. Attaching a backend does not start a run. A later job submit uses
whatever was attached.
"""

from __future__ import annotations

import json
from typing import Any

from internal_ai.compute.scheduler import JobBackend, LocalBackend, SlurmBackend
from internal_ai.context import Platform
from internal_ai.sandbox import Sandbox
from internal_ai.training.cost import estimate_gpu_run
from internal_ai.training.trainer import TOKENS_PER_PARAMETER_FLOOR

BACKEND_KEY = "compute_backend"
SIZES = (("small", 10_000_000), ("1b", 1_000_000_000), ("7b", 7_000_000_000))
LOCAL_HOUR_LIMIT = 1.0


def attached_backend(platform: Platform) -> str:
    value = platform.store.kv_get(BACKEND_KEY) or "local"
    return value if value in {"local", "slurm"} else "local"


def attach_backend(platform: Platform, name: str, actor: str = "infra_engineer") -> str:
    """Record the machine that showed up. Training stays a separate command."""
    if name not in {"local", "slurm"}:
        raise ValueError("a backend is local or slurm")
    platform.store.kv_set(BACKEND_KEY, name)
    platform.audit.record(
        actor=actor,
        actor_kind="human",
        action="compute.attached",
        subject=name,
        details={"backend": name},
    )
    return name


def make_backend(platform: Platform, sandbox: Sandbox) -> JobBackend:
    if attached_backend(platform) == "slurm":
        return SlurmBackend()
    return LocalBackend(sandbox)


def advise_dataset(
    platform: Platform, dataset_id: str, actor: str = "researcher"
) -> dict[str, Any]:
    """GPU hours for the sizes this platform already names, stored for the dataset."""
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    tokens = int(dataset.token_count)
    kind = str(dataset.stats.get("token_count_kind") or "estimate")
    measured = kind == "measured"
    backend = attached_backend(platform)
    sizes = [_size_row(name, parameters, tokens, measured, backend) for name, parameters in SIZES]
    advice = {
        "dataset_id": dataset_id,
        "tokens": tokens,
        "token_count_kind": kind,
        "measured": measured,
        "backend": backend,
        "sizes": sizes,
    }
    platform.store.kv_set(f"compute_advice:{dataset_id}", json.dumps(advice, sort_keys=True))
    platform.audit.record(
        actor=actor,
        actor_kind="agent",
        action="compute.advised",
        subject=dataset_id,
        details={
            "tokens": tokens,
            "token_count_kind": kind,
            "backend": backend,
            "sizes": [
                {
                    "size": row["size"],
                    "planning_gpu_hours": row["planning_gpu_hours"],
                    "runnable": row["runnable"],
                }
                for row in sizes
            ],
        },
    )
    return advice


def _size_row(
    name: str, parameters: int, tokens: int, measured: bool, backend: str
) -> dict[str, Any]:
    estimate = estimate_gpu_run(parameters, tokens)
    ratio = tokens / parameters if parameters else 0.0
    meets_floor = ratio >= TOKENS_PER_PARAMETER_FLOOR
    local = estimate.planning_gpu_hours <= LOCAL_HOUR_LIMIT
    machine = "local" if local else "H100-class"
    runnable = measured and meets_floor and (local or backend == "slurm")
    return {
        "size": name,
        "parameters": parameters,
        "planning_gpu_hours": round(estimate.planning_gpu_hours, 4),
        "estimated_cost_usd": round(estimate.estimated_cost_usd, 4),
        "tokens_per_parameter": round(ratio, 4),
        "meets_floor": meets_floor,
        "machine": machine,
        "runnable": runnable,
    }
