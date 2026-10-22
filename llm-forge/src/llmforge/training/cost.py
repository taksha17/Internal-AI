"""Compute and cost estimation for training runs (PRD section 10, FR-13).

Training FLOPs are approximately 6 x parameters x tokens. Planning figures use a
3x to 5x multiplier for failures and ablations, as the PRD requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

H100_SUSTAINED_TFLOPS = 400.0
DEFAULT_GPU_HOUR_USD = 2.0
DEFAULT_CPU_HOUR_USD = 0.35
DEFAULT_OVERHEAD_FACTOR = 3.0


@dataclass(slots=True)
class CostEstimate:
    parameters: int
    tokens: int
    training_flops: float
    ideal_gpu_hours: float
    planning_gpu_hours: float
    estimated_cost_usd: float
    device: str
    overhead_factor: float = DEFAULT_OVERHEAD_FACTOR
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "tokens": self.tokens,
            "training_flops": self.training_flops,
            "ideal_gpu_hours": round(self.ideal_gpu_hours, 4),
            "planning_gpu_hours": round(self.planning_gpu_hours, 4),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "device": self.device,
            "overhead_factor": self.overhead_factor,
            "notes": self.notes,
            **self.extras,
        }


def training_flops(parameters: int, tokens: int) -> float:
    return 6.0 * parameters * tokens


def gpu_hours(flops: float, sustained_tflops: float = H100_SUSTAINED_TFLOPS) -> float:
    return flops / (sustained_tflops * 1e12) / 3600.0


def estimate_gpu_run(
    parameters: int,
    tokens: int,
    gpu_hour_usd: float = DEFAULT_GPU_HOUR_USD,
    overhead_factor: float = DEFAULT_OVERHEAD_FACTOR,
    sustained_tflops: float = H100_SUSTAINED_TFLOPS,
) -> CostEstimate:
    flops = training_flops(parameters, tokens)
    ideal = gpu_hours(flops, sustained_tflops)
    planning = ideal * overhead_factor
    return CostEstimate(
        parameters=parameters,
        tokens=tokens,
        training_flops=flops,
        ideal_gpu_hours=ideal,
        planning_gpu_hours=planning,
        estimated_cost_usd=planning * gpu_hour_usd,
        device="gpu",
        overhead_factor=overhead_factor,
        notes="H100-class sustained throughput assumption from PRD section 10.1",
    )


def estimate_cpu_run(
    parameters: int,
    tokens: int,
    tokens_per_second: float,
    cpu_hour_usd: float = DEFAULT_CPU_HOUR_USD,
    overhead_factor: float = DEFAULT_OVERHEAD_FACTOR,
) -> CostEstimate:
    flops = training_flops(parameters, tokens)
    seconds = tokens / max(tokens_per_second, 1.0)
    hours = seconds / 3600.0
    planning = hours * overhead_factor
    return CostEstimate(
        parameters=parameters,
        tokens=tokens,
        training_flops=flops,
        ideal_gpu_hours=hours,
        planning_gpu_hours=planning,
        estimated_cost_usd=planning * cpu_hour_usd,
        device="cpu",
        overhead_factor=overhead_factor,
        notes=(
            f"CPU throughput assumption of {tokens_per_second:.0f} tokens/s; "
            "no GPU is available in this environment"
        ),
        extras={"tokens_per_second": tokens_per_second, "cpu_hour_usd": cpu_hour_usd},
    )


def cost_cap_from_estimate(estimate: CostEstimate, multiplier: float = 1.25) -> float:
    return round(estimate.estimated_cost_usd * multiplier, 4)


def within_cap(estimate: CostEstimate, cap_usd: float) -> bool:
    return estimate.estimated_cost_usd <= cap_usd
