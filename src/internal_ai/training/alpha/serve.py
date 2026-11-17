"""Answer from a merged KYC model with plain transformers generate.

The reference generator in local.py only replays the streamed rank-8 format
and refuses full-depth adapters on purpose, so a merged model needs its own
loader: weights, config and tokenizer from one directory, greedy decoding so
the same prompt answers the same way twice. Quantised loading is the same
choice training makes for the same reason: a 5.1B model at full precision
needs about 10 GB and the 3070 has 8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def quantised_budget() -> dict[int | str, str] | None:
    """What a quantised load may claim, as the training loader works it out.

    Serving and training face the same card: quantisation reaches the linear
    layers, the embedding tables stay in the compute dtype, and the merged model
    is a gigabyte larger than the base it came from.
    """
    import torch

    from internal_ai.training.alpha.adapt import _quantised_budget

    return _quantised_budget(torch)


def load_for_serving(model_dir: Path, load_in_4bit: bool = False) -> tuple[Any, Any]:
    """Load a merged model directory for completion. Refuses a bare adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_dir.is_dir():
        raise ValueError("no model directory at " + str(model_dir))
    if (model_dir / "adapter_config.json").is_file():
        raise ValueError(
            "this is an adapter directory, not a merged model: run merge_adapter first, "
            "because serving the pair means trusting a pairing the record cannot see"
        )
    kwargs: dict[str, Any] = {"local_files_only": True}
    if load_in_4bit:
        import torch
        from transformers import BitsAndBytesConfig

        budget = quantised_budget()
        if budget is None:
            raise ValueError("quantised serving needs a CUDA card")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=budget is not None,
        )
        # A budget replaces the auto map: given both, the load is planned on meta
        # and bitsandbytes reads a quantisation offset that was left there.
        if budget is None:
            kwargs["device_map"] = "auto"
        else:
            kwargs["max_memory"] = budget
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return model, tokenizer


def complete(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 32) -> str:
    """Greedy completion: the same prompt answers the same way twice."""
    import torch

    ids = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    ids = {key: value.to(device) for key, value in ids.items()}
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad
        )
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()
