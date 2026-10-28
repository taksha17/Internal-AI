"""Streamed LoRA on a prefix of the pinned Qwen3 model, using released text.

The 4 GB card cannot hold the whole model. Each layer is loaded, used, and
dropped. Only the query projection is adapted. The embedding stays frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from internal_ai.training.alpha.pin import RECIPE, PinnedBase, verify_snapshot
from internal_ai.training.alpha.qwen import (
    layer_forward,
    load_layer,
    load_shape,
    load_tensor,
    tensor_index,
)


def _encode(directory: Path, text: str, seq_len: int) -> list[int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    return tokenizer.encode(text).ids[:seq_len]


def train_prefix(
    base: PinnedBase,
    root: Path,
    texts: list[str],
    out: Path,
    layers: int = 2,
    sequences: int = 4,
) -> dict[str, Any]:
    """Adapt the first ``layers`` of the pinned base on ``texts``."""
    import torch
    import torch.nn.functional as F

    directory = verify_snapshot(base, root)
    if not (directory / "config.json").is_file() or not (directory / "tokenizer.json").is_file():
        raise FileNotFoundError("pinned snapshot is missing config.json or tokenizer.json")
    shape = load_shape(directory)
    depth = max(1, min(layers, shape.layers))
    index = tensor_index(base, root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_name = "model.embed_tokens.weight"
    embed = load_tensor(index[embed_name], embed_name, "cpu")
    embed = embed.half().to(device) if device == "cuda" else embed.to(device)
    query_out = shape.heads * shape.head_dim
    scale = RECIPE.alpha / RECIPE.rank
    torch.manual_seed(0)
    pairs: dict[int, tuple[Any, Any]] = {}
    for layer in range(depth):
        left = torch.randn(RECIPE.rank, shape.hidden, device=device) * 0.02
        right = torch.zeros(query_out, RECIPE.rank, device=device)
        pairs[layer] = (left, right)
    encoded = [_encode(directory, text, RECIPE.max_seq_len) for text in texts[:sequences]]
    encoded = [ids for ids in encoded if len(ids) >= 2]
    if not encoded:
        raise ValueError("released text did not produce a trainable sequence")
    losses: list[float] = []
    for ids in encoded:
        tokens = torch.tensor(ids, device=device)
        hidden = embed[tokens].float().unsqueeze(0)
        inputs: list[Any] = []
        for layer in range(depth):
            inputs.append(hidden.detach())
            weights = load_layer(index, layer, device)
            left, right = pairs[layer]
            with torch.no_grad():
                hidden = layer_forward(hidden, weights, shape, left.detach(), right.detach(), scale)
            del weights
        final = hidden.detach().requires_grad_(True)
        logits = torch.nn.functional.linear(final.half(), embed).float()
        loss = F.cross_entropy(logits[0, :-1], tokens[1:])
        loss.backward()
        grad = final.grad
        losses.append(float(loss.detach().cpu()))
        print(f"sequence {len(losses)} loss {losses[-1]:.3f}", flush=True)
        del logits, loss, final
        for layer in reversed(range(depth)):
            weights = load_layer(index, layer, device)
            left, right = pairs[layer]
            left.requires_grad_(True)
            right.requires_grad_(True)
            source = inputs[layer].detach().requires_grad_(True)
            output = layer_forward(source, weights, shape, left, right, scale)
            torch.autograd.backward(output, grad)
            with torch.no_grad():
                if left.grad is not None:
                    left -= 0.01 * left.grad
                    left.grad = None
                if right.grad is not None:
                    right -= 0.01 * right.grad
                    right.grad = None
            grad = source.grad.detach() if source.grad is not None else grad
            del weights, output, source
        if device == "cuda":
            torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    saved_layers = {
        str(layer): {"A": left.detach().cpu(), "B": right.detach().cpu()}
        for layer, (left, right) in pairs.items()
    }
    torch.save(
        {
            "layers": saved_layers,
            "base_repo_id": base.repo_id,
            "base_revision": base.revision,
            "rank": RECIPE.rank,
            "max_seq_len": RECIPE.max_seq_len,
            "quantization": RECIPE.quantization,
            "losses": losses,
            "trained_layers": depth,
        },
        out,
    )
    return {
        "base_repo_id": base.repo_id,
        "adapter": str(out),
        "sequences": len(encoded),
        "layers": depth,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
