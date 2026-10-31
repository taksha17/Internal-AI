"""The pinned base, adapted, answering on this machine with nothing else involved.

Until now every answer came from a runner someone else had to start:
`/v1/answer` refused outright when `INTERNAL_AI_LOCAL_RUNNER` was unset. That is
the gap between "runs on your machines" and "we call a server on your LAN": the
weights, the adapter and the tokenizer are all already on disk and pinned by
hash, so the answer can be produced here.

This is a reference generator, not a serving engine. It recomputes the prefix
for every token and streams each layer from disk per step, because that is what
the 4 GB recipe can do; it is deterministic by construction, since a temperature
would make the same question answer differently twice.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.llm import Completion, LLMAdapter
from internal_ai.training.alpha.pin import PINNED, RECIPE, PinnedBase, verify_snapshot
from internal_ai.training.alpha.qwen import (
    layer_forward,
    linear,
    load_layer,
    load_shape,
    load_tensor,
    rms_norm,
    tensor_index,
)
from internal_ai.training.provenance import require_open_weights

EMBED_NAME = "model.embed_tokens.weight"
NORM_NAME = "model.norm.weight"
HEAD_NAME = "lm_head.weight"


class LocalGenerationUnavailable(RuntimeError):
    """The pieces this machine would need to answer locally are not installed."""


def base_for(model: ModelVersion) -> PinnedBase:
    """The pinned base the card names, never one that merely happens to be on disk."""
    repo_id = str(model.card.get("base_repo_id") or "")
    revision = str(model.card.get("base_revision") or "")
    for base in PINNED:
        if base.repo_id == repo_id and base.revision == revision:
            return base
    raise ValueError(
        f"model {model.id} does not name a pinned base this package can load "
        f"(card says {repo_id or 'nothing'} at {revision or 'nothing'})"
    )


def load_adapter(path: Path) -> dict[int, tuple[Any, Any]]:
    """The rank-8 pairs `train_prefix` saved, as floats the forward pass can use."""
    import torch

    payload = torch.load(path, weights_only=True, map_location="cpu")
    pairs = {
        int(layer): (value["A"].float(), value["B"].float())
        for layer, value in dict(payload.get("layers") or {}).items()
    }
    if not pairs:
        raise ValueError(f"{path.name} holds no adapted layers")
    return pairs


def generate(
    base: PinnedBase,
    root: Path,
    prompt: str,
    adapter_path: Path | None = None,
    max_tokens: int = 128,
) -> str:
    """Greedy tokens from the base, with the adapter on the layers it covers."""
    import torch

    directory = verify_snapshot(base, root)
    if adapter_path is not None and adapter_path.is_dir():
        raise LocalGenerationUnavailable(
            "this adapter was saved by the full-depth backend as a directory, and the "
            "reference generator here only replays the streamed rank-8 format; serve it "
            "with the local runner (set INTERNAL_AI_LOCAL_RUNNER)"
        )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - the alpha extra is optional
        raise LocalGenerationUnavailable(
            "generating from a base needs the tokenizer the base ships with: "
            "install the alpha extra (pip install 'internal-ai[alpha]')"
        ) from exc

    shape = load_shape(directory)
    index = tensor_index(base, root)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = load_tensor(index[EMBED_NAME], EMBED_NAME, device)
    head = load_tensor(index[HEAD_NAME], HEAD_NAME, device) if HEAD_NAME in index else embed
    norm = load_tensor(index[NORM_NAME], NORM_NAME, device)
    pairs = load_adapter(adapter_path) if adapter_path is not None else {}
    scale = RECIPE.alpha / RECIPE.rank
    ids = tokenizer.encode(prompt).ids[-RECIPE.max_seq_len :]
    if not ids:
        raise ValueError("the prompt encoded to no tokens")
    produced: list[int] = []
    for _ in range(max(1, max_tokens)):
        window = (ids + produced)[-RECIPE.max_seq_len :]
        hidden = embed[torch.tensor(window, device=device)].float().unsqueeze(0)
        for layer in range(shape.layers):
            weights = load_layer(index, layer, device)
            if layer in pairs:
                left, right = pairs[layer]
                hidden = layer_forward(hidden, weights, shape, left, right, scale)
            else:
                hidden = layer_forward(hidden, weights, shape)
            del weights
        hidden = rms_norm(hidden, norm, shape.eps)
        logits = linear(hidden[:, -1, :], head)
        produced.append(int(torch.argmax(logits, dim=-1).item()))
    return tokenizer.decode(produced).strip()


class LocalModelAdapter:
    """The base plus adapter on this machine. `LLMAdapter` so `cited_answer` cannot tell."""

    name = "local-base"

    def __init__(self, platform: Platform, model: ModelVersion, max_tokens: int = 128) -> None:
        self.base = base_for(model)
        self.root = platform.settings.root
        self.adapter_path = Path(model.path) if model.path else None
        self.model_name = f"{self.base.repo_id}+{model.name}-v{model.version}"
        self.max_tokens = max_tokens

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Completion:
        if temperature:
            raise ValueError("local generation is greedy; a temperature would make it unrepeatable")
        text = generate(
            self.base,
            self.root,
            f"{system.strip()}\n{prompt}".strip() if system else prompt,
            self.adapter_path,
            min(max_tokens, self.max_tokens),
        )
        return Completion(
            text=text,
            model=self.model_name,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )


def answer_adapter(platform: Platform, model: ModelVersion) -> LLMAdapter:
    """A configured runner if the lab has one, otherwise the weights on this disk.

    Reading pretrained weights is the act provenance mode governs, so the local
    path asks before it loads anything. The runner path does not, because the
    process that already owns those weights is not this one.
    """
    runner_configured = os.environ.get("INTERNAL_AI_LOCAL_RUNNER", "").strip()
    if runner_configured:
        from internal_ai.llm import LocalRunnerAdapter

        return LocalRunnerAdapter()
    require_open_weights(platform, "answering from the base model")
    return LocalModelAdapter(platform, model)
