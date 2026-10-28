"""Model card for an alpha adapter. The base is named; it is not our model."""

from __future__ import annotations

from typing import Any

from internal_ai.ids import now_iso
from internal_ai.training.alpha.pin import AlphaRecipe, PinnedBase


def alpha_card(
    base: PinnedBase,
    recipe: AlphaRecipe,
    dataset_id: str,
    dataset_hash: str | None,
    adapter_sha256: str,
) -> dict[str, Any]:
    return {
        "intended_use": "answer from cited internal documents",
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "training_config": {
            "rank": recipe.rank,
            "alpha": recipe.alpha,
            "max_seq_len": recipe.max_seq_len,
            "quantization": recipe.quantization,
            "stream_from_disk": recipe.stream_from_disk,
        },
        "known_limitations": [
            "The published base was not trained by this project.",
            "Answers may only repeat Ask hits, with each line kept on its own file.",
            "Alpha output is not training data for the from-scratch track.",
            "An improvement is a new released dataset version and a new run.",
        ],
        "trained_from_scratch": False,
        "pretrained_weights_used": True,
        "base_repo_id": base.repo_id,
        "base_revision": base.revision,
        "base_license": base.license,
        "adapter_sha256": adapter_sha256,
        "created_at": now_iso(),
    }
