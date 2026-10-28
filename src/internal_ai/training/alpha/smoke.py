"""One streamed LoRA step on the pinned base, then the matrix is dropped."""

from __future__ import annotations

from pathlib import Path

import torch

from internal_ai.training.alpha.loader import first_matrix
from internal_ai.training.alpha.pin import available_base
from internal_ai.training.alpha.stream import lora_step


def _step(base, root: Path, out: Path) -> dict[str, str]:
    weight = first_matrix(base, root)
    adapter = lora_step(weight)
    del weight
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "A": adapter["A"],
            "B": adapter["B"],
            "base_repo_id": base.repo_id,
            "base_revision": base.revision,
            "rank": adapter["rank"],
            "max_seq_len": adapter["max_seq_len"],
            "quantization": adapter["quantization"],
        },
        out,
    )
    return {"base_repo_id": base.repo_id, "base_revision": base.revision, "adapter": str(out)}


def smoke(root: Path, out: Path) -> dict[str, str]:
    """One streamed step. A GPU out-of-memory error uses the local 3B pin."""
    from internal_ai.training.alpha.pin import FALLBACK, verify_snapshot

    base = available_base(root)
    try:
        return _step(base, root, out)
    except RuntimeError as exc:
        if base.repo_id == FALLBACK.repo_id or "out of memory" not in str(exc).lower():
            raise
        verify_snapshot(FALLBACK, root)
        return _step(FALLBACK, root, out)
