"""One LoRA step on a matrix streamed from disk, then the matrix is dropped.

The 1650 cannot hold a 4B model. A step loads one saved matrix, updates only
the rank-8 adapter, and deletes the matrix before the next shard is opened.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from internal_ai.training.alpha.pin import RECIPE, AlphaRecipe


def lora_step(weight: Any, recipe: AlphaRecipe = RECIPE, steps: int = 1) -> dict[str, Any]:
    """Adapt a rank-8 pair against one streamed weight. The weight is not saved.

    The pair starts where LoRA starts: `A` carries the directions and `B`
    carries nothing, so the adapted matrix is the streamed one until the
    first step moves `B`. The step then drives the pair towards a rank-8
    approximation of the streamed matrix, and reports the loss it started
    and ended on. This is the machinery a fine-tune runs on; it is not a
    fine-tune, and the training loss on company text lives in `train.py`.
    """
    if recipe.quantization is not None:
        raise ValueError("alpha does not quantise the base; 4-bit crashed on this GPU")
    if not recipe.stream_from_disk:
        raise ValueError("alpha streams the base from disk")
    import torch

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
    matrix = weight.detach().float().to(device)
    out_features, in_features = matrix.shape
    left = torch.empty(recipe.rank, in_features, device=device)
    torch.nn.init.kaiming_uniform_(left, a=math.sqrt(5))
    left.requires_grad_(True)
    right = torch.zeros(out_features, recipe.rank, device=device, requires_grad=True)
    scale = recipe.alpha / recipe.rank
    optimiser = torch.optim.Adam([left, right], lr=0.05)
    first = 0.0
    for step in range(steps):
        optimiser.zero_grad()
        residual = matrix - scale * (right @ left)
        loss = residual.pow(2).mean()
        loss.backward()
        optimiser.step()
        if step == 0:
            first = float(loss.detach())
    with torch.no_grad():
        final = float((matrix - scale * (right @ left)).pow(2).mean())
    del matrix
    return {
        "A": left.detach(),
        "B": right.detach(),
        "rank": recipe.rank,
        "alpha": recipe.alpha,
        "max_seq_len": recipe.max_seq_len,
        "quantization": recipe.quantization,
        "loss_first": first,
        "loss_last": final,
    }


def stream_lora_file(path: Path, key: str, recipe: AlphaRecipe = RECIPE) -> dict[str, Any]:
    """Load one tensor from a checkpoint file, step, and drop it."""
    import torch

    payload = torch.load(path, weights_only=True, map_location="cpu")
    try:
        weight = payload[key]
    finally:
        del payload
    adapter = lora_step(weight, recipe)
    del weight
    return adapter
