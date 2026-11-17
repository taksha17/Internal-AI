"""Fold a trained adapter back into its base and save one complete model.

Training keeps the base frozen and learns an adapter; serving the pair means
carrying both everywhere and trusting the pairing. The merge resolves the
adapter into the weights once, so the artifact downstream is a single model
directory: weights, config and tokenizer together, with a record naming the
base revision, the adapter directory hash, and the merged weight hash.

The base loads in full precision on the CPU for this, even when training held
it quantised: merging into 4-bit blocks is not supported, and the merge is a
one-off cost while every later load pays for a smaller file. The merged model
is the thing the trial serves, so the record it writes is what the trial cites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from internal_ai.ids import now_iso, tree_hash
from internal_ai.training.alpha.pin import PinnedBase, verify_snapshot

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def _copy_tokenizer(directory: Path, out: Path) -> None:
    for name in TOKENIZER_FILES:
        source = directory / name
        if source.is_file():
            (out / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def merge_adapter(base: PinnedBase, root: Path, adapter: Path, out: Path) -> dict[str, Any]:
    """Resolve adapter into base and save the complete model to out."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not adapter.is_dir():
        raise ValueError("no adapter directory at " + str(adapter))
    directory = verify_snapshot(base, root)
    # bfloat16 rather than float32: the base is published in bf16, so nothing is
    # lost, and a float32 merge of a 5.1B model needs about 20 GB of RAM and again
    # as much disk on a machine that has 31 GB in total.
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    merged.config.save_pretrained(out)
    _copy_tokenizer(directory, out)
    record = {
        "base_repo_id": base.repo_id,
        "base_revision": base.revision,
        "base_license": base.license,
        "adapter": str(adapter),
        "adapter_sha256": tree_hash(adapter),
        "merged_model": str(out),
        "merged_sha256": tree_hash(out),
        "created_at": now_iso(),
    }
    (out / "merge_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + chr(10))
    return record
