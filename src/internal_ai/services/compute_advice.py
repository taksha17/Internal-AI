"""Compute advice from the console, through the same functions the CLI calls.

The page does not compute hours. It asks ``compute/advise.py`` for the stored
advice and renders it, and the refusal text a waiting size shows is the text
``require_runnable_size`` raises -- because both read ``waiting_reasons``. A second
calculation on the page would eventually disagree with the one that refuses a run.
"""

from __future__ import annotations

from typing import Any

from internal_ai.compute.advise import (
    advise_dataset,
    attach_backend,
    attach_lane,
    attached_backend,
    attached_lane,
    stored_advice,
    waiting_reasons,
)
from internal_ai.compute.lanes import described
from internal_ai.context import Platform
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require


def advise(platform: Platform, actor: Actor, dataset_id: str) -> dict[str, Any]:
    require(platform.audit, actor, Action.ADVISE_COMPUTE, dataset_id)
    return advise_dataset(platform, dataset_id, actor=actor.id)


def attach(platform: Platform, actor: Actor, backend: str) -> str:
    require(platform.audit, actor, Action.ATTACH_COMPUTE, backend)
    return attach_backend(platform, backend, actor=actor.id)


def select_lane(platform: Platform, actor: Actor, label: str) -> str:
    require(platform.audit, actor, Action.ATTACH_COMPUTE, label)
    return attach_lane(platform, label, actor=actor.id)


def page(platform: Platform, dataset_id: str = "") -> dict[str, Any]:
    """Everything the template needs, with the waiting reasons already resolved."""
    datasets = [item for item in platform.store.list_datasets() if item.released]
    datasets.sort(key=lambda item: item.created_at, reverse=True)
    chosen = next((item for item in datasets if item.id == dataset_id), None)
    advice = stored_advice(platform, chosen.id) if chosen is not None else None
    rows: list[dict[str, Any]] = []
    if advice is not None:
        for row in advice.get("sizes", []):
            rows.append({**row, "reasons": waiting_reasons(advice, row)})
    return {
        "datasets": datasets,
        "dataset": chosen,
        "advice": advice,
        "rows": rows,
        "backend": attached_backend(platform),
        "lane": attached_lane(platform),
        "lanes": described(),
        "floor": _floor(),
    }


def _floor() -> int:
    from internal_ai.training.trainer import TOKENS_PER_PARAMETER_FLOOR

    return TOKENS_PER_PARAMETER_FLOOR
