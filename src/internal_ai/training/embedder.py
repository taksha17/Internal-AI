"""Embedding wrapper so trained checkpoints can be scored by the eval harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from internal_ai.data.tokenizer import Tokenizer
from internal_ai.training.model import PAD_ID, TinyTransformer, TransformerConfig

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required to serve embeddings") from exc


class TorchEmbedder:
    """Mean-pooled contextual embeddings from a trained checkpoint."""

    def __init__(
        self,
        model_path: Path,
        tokenizer: Tokenizer,
        max_seq_len: int = 256,
        name: str = "torch-embedder",
    ) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        payload = torch.load(model_path, weights_only=True, map_location="cpu")
        self.model_config = TransformerConfig.from_dict(payload["model_config"])
        self.model = TinyTransformer(self.model_config)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model_config.d_model), dtype=np.float32)
        encoded = [
            list(self.tokenizer.encode(text, max_tokens=self.max_seq_len - 1)) or [1]
            for text in texts
        ]
        width = max(len(row) for row in encoded)
        padded = [row + [PAD_ID] * (width - len(row)) for row in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long)
        mask = (input_ids != PAD_ID).unsqueeze(-1)
        with torch.no_grad():
            vectors = self.model.embed(input_ids, mask=mask)
        return vectors.cpu().numpy().astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_config": self.model_config.to_dict(),
            "vocab_size": self.tokenizer.vocab_size,
            "objective": "causal" if self.model_config.causal else "mlm",
        }
