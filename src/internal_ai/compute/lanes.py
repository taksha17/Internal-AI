"""Which device a training job may ask for, and what that name means.

A label is a node resource, not a wish. NVIDIA, AMD and Intel each name their
own resource, image, precisions and collective. A label this table does not
list is refused, because guessing ``nvidia.com/gpu`` for an unknown device
would schedule the job onto the wrong card.
"""

from __future__ import annotations

from dataclasses import dataclass


class LaneRefused(ValueError):
    """The device label is not one this install knows how to place."""


@dataclass(frozen=True, slots=True)
class Lane:
    label: str
    vendor: str
    resource: str
    image: str
    precisions: tuple[str, ...]
    collective: str
    gres: str


LANES: tuple[Lane, ...] = (
    Lane(
        label="nvidia.com/gpu",
        vendor="nvidia",
        resource="nvidia.com/gpu",
        image="internal-ai-train:cuda",
        precisions=("fp32", "fp16", "bf16"),
        collective="nccl",
        gres="gpu:1",
    ),
    Lane(
        label="amd.com/gpu",
        vendor="amd",
        resource="amd.com/gpu",
        image="internal-ai-train:rocm",
        precisions=("fp32", "fp16"),
        collective="rccl",
        gres="gpu:1",
    ),
    Lane(
        label="gpu.intel.com/i915",
        vendor="intel",
        resource="gpu.intel.com/i915",
        image="internal-ai-train:xpu",
        precisions=("fp32", "fp16"),
        collective="oneccl",
        gres="gpu:1",
    ),
)

_BY_LABEL = {lane.label: lane for lane in LANES}


def lane_for(label: str) -> Lane:
    found = _BY_LABEL.get((label or "").strip())
    if found is None:
        known = ", ".join(lane.label for lane in LANES)
        raise LaneRefused(f"unknown device label {label!r}; known labels are {known}")
    return found


def described() -> list[dict[str, str]]:
    """Rows for the compute page, in table order."""
    return [
        {
            "label": lane.label,
            "vendor": lane.vendor,
            "resource": lane.resource,
            "image": lane.image,
            "precisions": ", ".join(lane.precisions),
            "collective": lane.collective,
        }
        for lane in LANES
    ]
