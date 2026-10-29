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

# One feed-forward activation for the whole corpus was about 6.3 GB. A batch of
# 64 keeps that tensor near 30 MB, which fits beside the loaded documents.
EMBED_BATCH_SIZE = 64


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

    def embed(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model_config.d_model), dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        encoded = [
            self.tokenizer.content_ids(list(self.tokenizer.encode(text)))[: self.max_seq_len - 1]
            or [1]
            for text in texts
        ]
        # Padding participates in attention, so every batch uses the same width
        # a single forward over the whole list would have used.
        width = max(len(row) for row in encoded)
        parts = [
            self._embed_encoded(encoded[start : start + batch_size], width)
            for start in range(0, len(encoded), batch_size)
        ]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def _embed_encoded(self, encoded: list[list[int]], width: int) -> np.ndarray:
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
