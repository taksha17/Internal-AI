"""A transformer implemented from scratch.

Weights are randomly initialized and trained only on a released internal
dataset. A change is a new dataset version, not an edit. This from-scratch
track does not load a published checkpoint. The only loader for the pinned
open base is ``internal_ai.training.alpha``. Alpha output is not training
data for this track (PRD non-goals 2.2, policy P3).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for training. Install it with: uv pip install torch "
        "--index-url https://download.pytorch.org/whl/cpu"
    ) from exc

PAD_ID = 0


@dataclass(slots=True)
class TransformerConfig:
    vocab_size: int = 8_000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    causal: bool = False
    tie_embeddings: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransformerConfig:
        fields = {key: payload[key] for key in cls.__slots__ if key in payload}
        return cls(**fields)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.causal = config.causal
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, seq, dim)
        return self.out(self.dropout(attended))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TinyTransformer(nn.Module):
    """Encoder for embedding models, causal decoder for language models."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = (
            nn.Linear(config.d_model, config.vocab_size, bias=False)
            if config.tie_embeddings
            else nn.Linear(config.d_model, config.vocab_size)
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _batch, seq = input_ids.shape
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

    def embed(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pooled contextual embeddings for retrieval."""
        x = self.token_embedding(input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if mask is None:
            mask = (input_ids != PAD_ID).unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, dim=-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def count_parameters(config: TransformerConfig) -> int:
    with torch.random.fork_rng():
        model = TinyTransformer(config)
        return model.parameter_count()


def spec_violations(model: TinyTransformer, config: TransformerConfig) -> list[str]:
    """Where the built model disagrees with the config it was built from.

    Training will happily run through a mis-shaped layer and report a poor model,
    so the shapes are compared with the spec before the first step.
    """
    expected: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (config.vocab_size, config.d_model),
        "position_embedding.weight": (config.max_seq_len, config.d_model),
        "norm.weight": (config.d_model,),
    }
    if not config.tie_embeddings:
        expected["lm_head.weight"] = (config.vocab_size, config.d_model)
    shapes = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
    violations = [
        f"{name} is {'.'.join(str(size) for size in shapes.get(name, ()))} but the spec says "
        f"{'.'.join(str(size) for size in shape)}"
        for name, shape in expected.items()
        if shapes.get(name) != shape
    ]
    built_layers = sum(1 for name in shapes if name.endswith("attention.qkv.weight"))
    if built_layers != config.n_layers:
        violations.append(
            f"the model has {built_layers} attention layers; the spec says {config.n_layers}"
        )
    expected_count = count_parameters(config)
    actual_count = model.parameter_count()
    if actual_count != expected_count:
        violations.append(f"parameter count is {actual_count} but the spec says {expected_count}")
    return violations


def non_finite_parameters(model: TinyTransformer) -> list[str]:
    """Parameters holding a NaN or an infinity before training even starts."""
    return [
        name
        for name, parameter in model.named_parameters()
        if not bool(torch.isfinite(parameter).all())
    ]


def build_model(config: TransformerConfig, seed: int = 0) -> TinyTransformer:
    torch.manual_seed(seed)
    return TinyTransformer(config)


def cosine_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)
