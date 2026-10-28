"""Load one matrix from a verified pinned snapshot, then close the shard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from internal_ai.training.alpha.pin import PinnedBase, verify_snapshot


def _choose(names: list[str]) -> str | None:
    layer_name = "layers.0.self_attn.q_proj.weight"
    layer = next((name for name in names if name.endswith(layer_name)), None)
    if layer is not None:
        return layer
    return next((name for name in names if name.endswith(".weight")), None)


def first_matrix(base: PinnedBase, root: Path) -> Any:
    """Stream one layer matrix from the pinned shards, then close the file."""
    directory = verify_snapshot(base, root)
    from safetensors import safe_open

    chosen: tuple[str, str] | None = None
    for shard in base.shards:
        handle = safe_open(directory / shard.name, framework="pt", device="cpu")
        try:
            name = _choose(list(handle.keys()))
        finally:
            del handle
        if name is not None and (chosen is None or name.endswith("q_proj.weight")):
            chosen = (shard.name, name)
            if name.endswith("q_proj.weight"):
                break
    if chosen is None:
        raise ValueError(f"no matrix in pinned snapshot {base.repo_id}")
    shard_name, tensor_name = chosen
    handle = safe_open(directory / shard_name, framework="pt", device="cpu")
    try:
        return handle.get_tensor(tensor_name)
    finally:
        del handle
