"""One Qwen3 decoder layer, streamed. Frozen weights are loaded and then dropped."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from internal_ai.training.alpha.pin import PinnedBase, verify_snapshot


@dataclass(frozen=True, slots=True)
class QwenShape:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    eps: float
    rope_theta: float

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> QwenShape:
        return cls(
            hidden=int(payload["hidden_size"]),
            layers=int(payload["num_hidden_layers"]),
            heads=int(payload["num_attention_heads"]),
            kv_heads=int(payload["num_key_value_heads"]),
            head_dim=int(payload["head_dim"]),
            intermediate=int(payload["intermediate_size"]),
            vocab=int(payload["vocab_size"]),
            eps=float(payload["rms_norm_eps"]),
            rope_theta=float(payload["rope_theta"]),
        )


def load_shape(directory: Path) -> QwenShape:
    payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return QwenShape.from_config(payload)


def tensor_index(base: PinnedBase, root: Path) -> dict[str, Path]:
    """Map each tensor name to its shard. The shard stays closed after the names are read."""
    directory = verify_snapshot(base, root)
    from safetensors import safe_open

    found: dict[str, Path] = {}
    for shard in base.shards:
        path = directory / shard.name
        handle = safe_open(path, framework="pt", device="cpu")
        try:
            names = list(handle.keys())
        finally:
            del handle
        for name in names:
            found[name] = path
    return found


def load_tensor(path: Path, name: str, device: str) -> Any:
    import torch
    from safetensors import safe_open

    handle = safe_open(path, framework="pt", device="cpu")
    try:
        tensor = handle.get_tensor(name)
    finally:
        del handle
    return tensor.to(device=device, dtype=torch.float32)


def rms_norm(hidden: Any, weight: Any, eps: float) -> Any:
    import torch

    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normalised = hidden.float() * torch.rsqrt(variance + eps)
    return normalised * weight.float()


def _rotate_half(hidden: Any) -> Any:
    import torch

    half = hidden.shape[-1] // 2
    first, second = hidden[..., :half], hidden[..., half:]
    return torch.cat((-second, first), dim=-1)


def rope_angles(seq_len: int, head_dim: int, theta: float, device: str) -> tuple[Any, Any]:
    import torch

    exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta**exponents)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    angles = torch.cat((freqs, freqs), dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(states: Any, cos: Any, sin: Any) -> Any:
    cos_b = cos[None, :, None, :]
    sin_b = sin[None, :, None, :]
    return states * cos_b + _rotate_half(states) * sin_b


def linear(
    hidden: Any,
    weight: Any,
    left: Any | None = None,
    right: Any | None = None,
    scale: float = 1.0,
) -> Any:
    import torch

    projected = torch.nn.functional.linear(hidden, weight)
    if left is None or right is None:
        return projected
    return projected + scale * ((hidden @ left.transpose(0, 1)) @ right.transpose(0, 1))


def layer_forward(
    hidden: Any,
    weights: dict[str, Any],
    shape: QwenShape,
    left: Any | None = None,
    right: Any | None = None,
    scale: float = 1.0,
) -> Any:
    """One decoder layer. ``left`` and ``right`` are the rank-8 pair on the query projection."""
    import torch.nn.functional as F

    residual = hidden
    normed = rms_norm(hidden, weights["input_layernorm.weight"], shape.eps)
    query = linear(normed, weights["self_attn.q_proj.weight"], left, right, scale)
    key = linear(normed, weights["self_attn.k_proj.weight"])
    value = linear(normed, weights["self_attn.v_proj.weight"])
    batch, seq_len, _ = query.shape
    query = query.view(batch, seq_len, shape.heads, shape.head_dim)
    key = key.view(batch, seq_len, shape.kv_heads, shape.head_dim)
    value = value.view(batch, seq_len, shape.kv_heads, shape.head_dim)
    query = rms_norm(query, weights["self_attn.q_norm.weight"], shape.eps)
    key = rms_norm(key, weights["self_attn.k_norm.weight"], shape.eps)
    cos, sin = rope_angles(seq_len, shape.head_dim, shape.rope_theta, str(query.device))
    query = apply_rope(query, cos, sin)
    key = apply_rope(key, cos, sin)
    groups = shape.heads // shape.kv_heads
    key = key.repeat_interleave(groups, dim=2)
    value = value.repeat_interleave(groups, dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    hidden = residual + linear(attended, weights["self_attn.o_proj.weight"])
    residual = hidden
    normed = rms_norm(hidden, weights["post_attention_layernorm.weight"], shape.eps)
    gate = F.silu(linear(normed, weights["mlp.gate_proj.weight"]))
    up = linear(normed, weights["mlp.up_proj.weight"])
    hidden = residual + linear(gate * up, weights["mlp.down_proj.weight"])
    return hidden


LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def load_layer(index: dict[str, Path], layer: int, device: str) -> dict[str, Any]:
    prefix = f"model.layers.{layer}."
    loaded: dict[str, Any] = {}
    for suffix in LAYER_SUFFIXES:
        name = prefix + suffix
        loaded[suffix] = load_tensor(index[name], name, device)
    return loaded
