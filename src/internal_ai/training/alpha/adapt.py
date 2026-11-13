"""Full-depth LoRA on the pinned base, on the objective generation runs.

`train.py` streams a prefix of the model because a 4 GB card cannot hold the
whole thing: it adapts two layers and scores them through the embedding matrix,
skipping the remaining layers and the final normalisation. That objective is not
the one `local.py` runs at inference, which is why `workflows` refuses to promote
an adapter trained that way. This module is the backend that can be promoted —
every layer, the real LM head, an optimiser with a schedule — at the cost of
needing a machine that can hold the model.

It is a reference backend rather than a tuned one: sequences are grouped by
length and padded, not packed, and a multi-card lane gets a pipeline of layers
rather than data parallelism, so the second card takes turns instead of sharing
the step. The run reports tokens per second because the lane, not this
docstring, decides whether a change to the step actually helped.

It refuses to run without the `alpha` extra rather than pretending.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from internal_ai.ids import tree_hash
from internal_ai.training.alpha.pin import RECIPE, PinnedBase, verify_snapshot

BACKEND = "peft"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DIVERGENCE_FACTOR = 3.0
DIVERGENCE_WINDOW = 5
# A tokenizer loaded from a bare `tokenizer.json` ships no pad token. The
# positions this fills are masked out of attention and out of the loss, so the
# value only has to be a real row of the embedding.
PAD_ID = 0
IGNORE_INDEX = -100
HOST_MEMINFO = Path("/proc/meminfo")


def _batches(sequences: list[list[int]], batch_size: int) -> list[list[list[int]]]:
    """Group sequences into batches of similar length.

    Sorting by length before grouping is what makes a batch worth having: a batch
    drawn in file order pads every short sequence up to the longest one in the
    file, and on a lane billed by the hour the padding is the bill.
    """
    if batch_size <= 1:
        return [[ids] for ids in sequences]
    ordered = sorted(sequences, key=len)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def _padded(group: list[list[int]]) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """One batch as input ids, an attention mask and labels.

    A pad position is masked out of attention and set to -100 in the labels, so
    the cross entropy averages over the tokens that exist rather than over the
    padding, and a short sequence does not teach the model to predict pad.
    """
    width = max(len(ids) for ids in group)
    rows: list[list[int]] = []
    masks: list[list[int]] = []
    labels: list[list[int]] = []
    for ids in group:
        padding = width - len(ids)
        rows.append(ids + [PAD_ID] * padding)
        masks.append([1] * len(ids) + [0] * padding)
        labels.append(ids + [IGNORE_INDEX] * padding)
    return rows, masks, labels


def _landed_placement(model: Any) -> dict[str, str]:
    """Where the layers actually went, as opposed to the map that was asked for.

    ``parallelism`` names the arrangement a run requested, and on a lane with two
    visible cards it says "pipeline" whatever accelerate then decided: a 4B fits one
    T4, so the whole model may have sat on the first card with the second idle. A
    throughput number cannot be read without knowing which of those happened.
    """
    mapping = getattr(model, "hf_device_map", None)
    if not mapping:
        return {}
    return {str(key): str(value) for key, value in mapping.items()}


def _p50(seconds: list[float]) -> float:
    """The middle step time, because one slow first step is not the lane."""
    if not seconds:
        return 0.0
    ordered = sorted(seconds)
    return round(ordered[len(ordered) // 2], 4)


def _window_exploded(losses: list[float]) -> bool:
    """Whether the run's newest window sits far above its first window.

    Sequences differ in difficulty, so a single step three times the first one
    is ordinariness, not divergence; the mean of the newest window against the
    mean of the first window does not move for one hard sequence. A run needs
    more steps than a window to be judged by one.
    """
    if len(losses) < DIVERGENCE_WINDOW:
        return False
    first = sum(losses[:DIVERGENCE_WINDOW]) / DIVERGENCE_WINDOW
    recent = sum(losses[-DIVERGENCE_WINDOW:]) / DIVERGENCE_WINDOW
    return first > 0 and recent > DIVERGENCE_FACTOR * first


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


def _host_ram_gib() -> float | None:
    """The host memory a spilled load may claim, or nothing where it cannot be read.

    ``MemTotal`` is read before ``sysconf`` because a container can report the
    memory it is limited to rather than the memory the machine has: on the
    workstation that ran this, sysconf said 7.3 GiB where the host holds 15, and
    a budget that small sends the embedding tables to disk as meta tensors.
    """
    try:
        for line in HOST_MEMINFO.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
    except (OSError, IndexError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return pages * size / 1024**3


def _quantised_budget(torch: Any) -> dict[int | str, str] | None:
    """How much memory each card may claim before a quantised load spills to the host.

    Quantisation only reaches the linear layers. A multimodal base keeps its
    embedding tables in the compute dtype: Gemma 4 E2B holds 2.77B parameters
        there, about 5.5 GB in bf16, which does not sit beside the four-bit body on an
    8 GB card. Given a budget, accelerate spills those tables to the host and the
    run starts; without one it refuses the load outright.
    """
    try:
        if not torch.cuda.is_available():
            return None
        budget: dict[int | str, str] = {}
        for index in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(index)
            budget[index] = f"{free * 0.85 / 1024**3:.1f}GiB"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    host = _host_ram_gib()
    if host:
        budget["cpu"] = f"{host * 0.8:.1f}GiB"
    return budget


def _lora_targets(model: Any) -> list[str] | str:
    """The projections to wrap, or every linear layer when those names are not linears.

    A base may wrap each projection in a module of its own: Gemma 4 puts every one
    inside a clippable block, so ``q_proj`` names the wrapper and the layer sits at
    ``q_proj.linear``. peft refuses such a target rather than skipping it, and
    ``all-linear`` selects by type instead, reaching the quantised linear inside.
    The name alone is not enough to tell the two apart, so the module is checked.
    """
    import torch

    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in TARGET_MODULES and isinstance(module, torch.nn.Linear):
            return list(TARGET_MODULES)
    return "all-linear"


def _load_model(
    directory: Path,
    dtype: Any,
    device: str,
    load_in_4bit: bool = False,
    sharded: bool = False,
    torch_module: Any | None = None,
) -> Any:
    from transformers import AutoModelForCausalLM

    if load_in_4bit:
        # A card that cannot hold the weights at half precision quantises the
        # frozen base to nf4 instead: a 30B at fp16 needs about 60 GB, which no
        # free Colab GPU has, and at nf4 it needs about 18 GB. The adapter is
        # still trained in the compute dtype, so the arithmetic is the same.
        # auto rather than a fixed card, because Kaggle's free tier offers two
        # T4s in one session and a 30B at nf4 needs both of them.
        from transformers import BitsAndBytesConfig

        if torch_module is None:
            import torch as torch_module

        budget = _quantised_budget(torch_module)
        # A budget replaces the auto map rather than joining it: given both,
        # accelerate plans the load on meta and bitsandbytes then reads a
        # quantisation offset that was left there, which fails on the first step.
        placement: dict[str, Any] = (
            {"max_memory": budget} if budget is not None else {"device_map": "auto"}
        )
        return AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                # Bitsandbytes refuses any module that is dispatched away from a
                # card unless this is set, whatever the quantisation it applied.
                llm_int8_enable_fp32_cpu_offload=budget is not None,
            ),
            **placement,
        )

    if sharded:
        # Half precision needs 14.5 GB for a 7B, which no single T4 holds with room
        # for activations. A device map splits the layers across the cards; calling
        # .to() afterwards would gather them back onto one and undo the point.
        try:
            return AutoModelForCausalLM.from_pretrained(
                directory, local_files_only=True, dtype=dtype, device_map="auto"
            )
        except TypeError:  # transformers 4 spells the same argument torch_dtype
            return AutoModelForCausalLM.from_pretrained(
                directory, local_files_only=True, torch_dtype=dtype, device_map="auto"
            )
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
    load_in_4bit: bool = False,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
) -> dict[str, Any]:
    """Adapt every layer of the pinned base on ``texts`` and save a PEFT adapter.

    The text is a document, not a prompt/completion pair, so the loss is the
    ordinary causal objective over every token rather than completion-only: there
    is no assistant turn here to mask down to.

    ``load_in_4bit`` changes how the frozen weights are held, never which weights
    may be loaded: the pin, the licence class and the provenance mode still decide
    that, and a quantised run is recorded as quantised so a score can say what
    produced it.

    ``batch_size`` groups sequences that are close in length and masks the
    padding; ``gradient_accumulation`` keeps the effective batch the recipe asked
    for when one step cannot hold it. Neither changes which weights are loaded or
    what the objective is, so a run that uses them is still comparable with one
    that does not: it says which it used.

    The result reports the arrangement that was requested *and* the placement that
    happened (``device_map_landed``, ``devices_in_use``), because a lane with two
    cards does not necessarily use both and a tokens-per-second figure means
    nothing without that distinction.
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
    if load_in_4bit and device != "cuda":
        raise ValueError(
            "quantised loading needs a CUDA card; on this machine install the alpha extra "
            "and run at full precision, or use a shorter streamed run instead"
        )
    torch.manual_seed(seed)
    # More than one card means the weights are split rather than gathered, which
    # is the only way a half-precision 7B fits a dual-T4 session.
    sharded = device == "cuda" and torch.cuda.device_count() > 1
    dtype = _dtype(torch, device)
    model = _load_model(directory, dtype, device, load_in_4bit, sharded)
    # Read before peft wraps the model: the map lives on the loader's object, and
    # the record should describe the placement that holds the frozen weights.
    landed = _landed_placement(model)
    if load_in_4bit:
        # peft's k-bit preparation casts the frozen norms to fp32 and turns on
        # input gradients; without it an adapter over 4-bit blocks receives no
        # gradient at all, and the run would report a loss that never moves.
        #
        # This is also the one loading choice that changes the arithmetic rather
        # than only the memory it needs: the frozen weights are dequantised on
        # every matmul. A quantised run therefore needs a score against a frozen
        # set before it replaces an unquantised one; being faster is not enough.
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = _lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
        ),
    )
    if device == "cuda":
        # Recomputing activations on the backward pass costs time and buys memory.
        # The gradient is the same number computed twice, so this is the one knob
        # in this module that cannot change what the model learns.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    wrapped = targets if isinstance(targets, str) else list(targets)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    if device != "cuda":
        parallelism = "cpu"
    elif load_in_4bit or sharded:
        # The arrangement requested, which is not the placement achieved: see
        # ``device_map_landed`` for where the layers ended up.
        parallelism = "device_map_pipeline"
    else:
        parallelism = "single_card"
    groups = _batches(sequences, max(1, int(batch_size)))
    accumulation = max(1, int(gradient_accumulation))
    per_epoch = max(1, -(-len(groups) // accumulation))
    steps = max(1, int(epochs)) * per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, steps // 10), num_training_steps=steps
    )
    scaler = _loss_scaling(torch, dtype)
    losses: list[float] = []
    step_seconds: list[float] = []
    diverged_at: int | None = None
    optimizer_steps = 0
    tokens = 0
    started = time.perf_counter()
    last_step = started
    model.train()
    for _epoch in range(max(1, int(epochs))):
        pending = 0
        for index, group in enumerate(groups):
            rows, masks, labels = _padded(group)
            outputs = model(
                input_ids=torch.tensor(rows, device=device),
                attention_mask=torch.tensor(masks, device=device),
                labels=torch.tensor(labels, device=device),
            )
            loss = outputs.loss
            if not bool(torch.isfinite(loss)):
                diverged_at = len(losses)
                break
            scaled = loss / accumulation
            if scaler is None:
                scaled.backward()
            else:
                scaler.scale(scaled).backward()
            pending += 1
            tokens += sum(len(ids) for ids in group)
            if pending == accumulation or index == len(groups) - 1:
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                now = time.perf_counter()
                step_seconds.append(now - last_step)
                last_step = now
                losses.append(float(loss.detach()))
                pending = 0
                if _window_exploded(losses):
                    diverged_at = len(losses)
                    break
            del outputs, loss, scaled
        if diverged_at is not None:
            break
    elapsed = time.perf_counter() - started
    if diverged_at is not None:
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
        # Which layers the adapter actually reached, because a base that wraps its
        # projections trains a different set than the pinned names describe.
        "lora_targets": wrapped,
        "quantisation": "nf4-4bit" if load_in_4bit else "none",
        "device_map": "auto" if (load_in_4bit or sharded) else device,
        "device_map_landed": landed,
        "devices_in_use": len(set(landed.values())),
        "parallelism": parallelism,
        "batch_size": max(1, int(batch_size)),
        "gradient_accumulation": accumulation,
        "optimizer_steps": optimizer_steps,
        "tokens": tokens,
        # Measured here rather than quoted from a blog: a change to the step is
        # only an optimisation if this number went up on the lane that runs it.
        "step_seconds_p50": _p50(step_seconds),
        "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "diverged_at": diverged_at,
    }
