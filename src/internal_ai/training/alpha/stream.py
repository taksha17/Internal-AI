"""One LoRA step on a matrix streamed from disk, then the matrix is dropped.

The 1650 cannot hold a 4B model. A step loads one saved matrix, updates only
the rank-8 adapter, and deletes the matrix before the next shard is opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from internal_ai.training.alpha.pin import RECIPE, AlphaRecipe


def lora_step(weight: Any, recipe: AlphaRecipe = RECIPE, steps: int = 1) -> dict[str, Any]:
    """Adapt a rank-8 pair against one streamed weight. The weight is not saved."""
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
    left = torch.zeros(recipe.rank, in_features, device=device, requires_grad=True)
    right = torch.zeros(out_features, recipe.rank, device=device, requires_grad=True)
    tokens = torch.randn(1, recipe.max_seq_len, in_features, device=device)
    scale = recipe.alpha / recipe.rank
    for _ in range(steps):
        adapted = tokens @ matrix.T + scale * ((tokens @ left.T) @ right.T)
        loss = adapted.float().pow(2).mean()
        loss.backward()
        with torch.no_grad():
            left -= 0.01 * left.grad
            right -= 0.01 * right.grad
            left.grad = None
            right.grad = None
    del matrix
    return {
        "A": left.detach(),
        "B": right.detach(),
        "rank": recipe.rank,
        "alpha": recipe.alpha,
        "max_seq_len": recipe.max_seq_len,
        "quantization": recipe.quantization,
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
