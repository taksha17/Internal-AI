"""Full-depth LoRA on the pinned base, on the objective generation runs.

`train.py` streams a prefix of the model because a 4 GB card cannot hold the
whole thing: it adapts two layers and scores them through the embedding matrix,
skipping the remaining layers and the final normalisation. That objective is not
the one `local.py` runs at inference, which is why `workflows` refuses to promote
an adapter trained that way. This module is the backend that can be promoted —
every layer, the real LM head, an optimiser with a schedule — at the cost of
needing a machine that can hold the model.

It is a reference backend, not a fast one: one sequence at a time, no packing,
and it refuses to run without the `alpha` extra rather than pretending.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from internal_ai.ids import tree_hash
from internal_ai.training.alpha.pin import RECIPE, PinnedBase, verify_snapshot

BACKEND = "peft"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DIVERGENCE_FACTOR = 3.0


class AdaptationUnavailable(RuntimeError):
    """The packages this backend trains with are not installed."""


def adaptation_available() -> bool:
    """Whether this install can adapt at full depth at all."""
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _bf16_is_native(torch: Any) -> bool:
    """Whether this card computes bf16 itself rather than through emulation.

    The plain question answers yes on Turing, where the emulation is six times
    slower than fp16 on a matmul, so it is the wrong question to load a model on.
    """
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return False


def _dtype(torch: Any, device: str) -> Any:
    """Half precision on a card, fp32 on the CPU.

    fp16 is the fallback rather than fp32 because the pinned base at 8.04 GB
    needs 14.99 GiB in fp32, which overruns a T4's 14.56 GiB outright: a card
    that cannot compute bf16 natively can still hold the model in fp16.
    """
    if device != "cuda":
        return torch.float32
    if _bf16_is_native(torch):
        return torch.bfloat16
    return torch.float16


def _loss_scaling(torch: Any, dtype: Any) -> Any:
    """Scale fp16 gradients so a step does not underflow into a refused run.

    Half precision cannot hold the small gradients LoRA produces on a 4B model,
    and the guard below reads the resulting non-finite loss as divergence, so
    without this the first step of an fp16 run fails for a reason that has
    nothing to do with the data.
    """
    if dtype is not torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _load_model(directory: Path, dtype: Any, device: str) -> Any:
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=dtype)
    except TypeError:  # transformers 4 spells the same argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, torch_dtype=dtype
        )
    return model.to(device)


def _load_tokenizer(directory: Path) -> Any:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if (directory / "tokenizer.json").is_file():
        return PreTrainedTokenizerFast(tokenizer_file=str(directory / "tokenizer.json"))
    try:
        return AutoTokenizer.from_pretrained(directory, local_files_only=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"the pinned snapshot at {directory} ships no usable tokenizer") from exc


def _encode(tokenizer: Any, text: str, max_seq_len: int) -> list[int]:
    encoded = tokenizer(text, truncation=True, max_length=max_seq_len, add_special_tokens=False)
    return [int(token) for token in encoded["input_ids"]]


def adapt(
    base: PinnedBase,
    root: Path,
    texts: list[str],
    out: Path,
    rank: int = RECIPE.rank,
    alpha: int = RECIPE.alpha,
    max_seq_len: int = RECIPE.max_seq_len,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    seed: int = 0,
) -> dict[str, Any]:
    """Adapt every layer of the pinned base on ``texts`` and save a PEFT adapter.

    The text is a document, not a prompt/completion pair, so the loss is the
    ordinary causal objective over every token rather than completion-only: there
    is no assistant turn here to mask down to.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the alpha extra is optional
        raise AdaptationUnavailable(
            "adapting at full depth needs torch: install the alpha extra "
            "(pip install 'internal-ai[alpha]')"
        ) from exc
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - the alpha extra is optional
        raise AdaptationUnavailable(
            "adapting at full depth needs transformers and peft: install the alpha "
            "extra (pip install 'internal-ai[alpha]')"
        ) from exc

    directory = verify_snapshot(base, root)
    tokenizer = _load_tokenizer(directory)
    sequences = [
        ids for ids in (_encode(tokenizer, text, max_seq_len) for text in texts) if len(ids) > 1
    ]
    if not sequences:
        raise ValueError("the training text did not produce a trainable sequence")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    dtype = _dtype(torch, device)
    model = _load_model(directory, dtype, device)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(TARGET_MODULES),
        ),
    )
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    steps = max(1, int(epochs)) * len(sequences)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, steps // 10), num_training_steps=steps
    )
    scaler = _loss_scaling(torch, dtype)
    losses: list[float] = []
    diverged_at: int | None = None
    model.train()
    for _epoch in range(max(1, int(epochs))):
        for ids in sequences:
            batch = torch.tensor([ids], device=device)
            loss = model(input_ids=batch, labels=batch).loss
            if not bool(torch.isfinite(loss)):
                diverged_at = len(losses)
                break
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if losses[0] > 0 and losses[-1] > DIVERGENCE_FACTOR * losses[0]:
                diverged_at = len(losses)
                break
        if diverged_at is not None:
            break
    if diverged_at is not None and len(losses) < len(sequences):
        raise ValueError(
            f"the loss diverged at step {diverged_at}; the run produced nothing worth keeping"
        )
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    return {
        "base_repo_id": base.repo_id,
        "adapter": str(out),
        "adapter_backend": BACKEND,
        "adapter_sha256": tree_hash(out),
        "sequences": len(sequences),
        "layers": int(getattr(model.config, "num_hidden_layers", 0) or 0),
        "epochs": max(1, int(epochs)),
        "trainable_parameters": trainable,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "diverged_at": diverged_at,
    }
