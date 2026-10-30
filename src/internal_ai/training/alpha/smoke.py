"""One streamed LoRA step on the pinned base, then the matrix is dropped.

The adapter this writes proves the machine can hold the pinned base and take a
gradient step on a matrix that came off disk. It is not a fine-tune, and the
artifact says so, so it is never mistaken for a model trained on company text.
"""

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
            "trained_on_company_text": False,
        },
        out,
    )
    return {
        "base_repo_id": base.repo_id,
        "base_revision": base.revision,
        "adapter": str(out),
        "kind": "smoke",
    }


def smoke(root: Path, out: Path) -> dict[str, str]:
    """One streamed step. Running out of memory does not swap in a research licence."""
    from internal_ai.training.alpha.pin import (
        DEFAULT_ALLOWED,
        FALLBACK,
        BaseRefused,
        verify_snapshot,
    )

    base = available_base(root)
    try:
        return _step(base, root, out)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if FALLBACK.license_class not in DEFAULT_ALLOWED:
            raise BaseRefused(
                f"{base.repo_id} ran out of memory and {FALLBACK.repo_id} is "
                f"{FALLBACK.license} ({FALLBACK.license_class}); using it needs an approved "
                "base_model_exception"
            ) from exc
        verify_snapshot(FALLBACK, root)
        return _step(FALLBACK, root, out)
