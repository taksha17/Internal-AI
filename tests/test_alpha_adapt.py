"""The full-depth backend is judged on the objective generation actually runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from internal_ai.training.alpha.adapt import (
    BACKEND,
    AdaptationUnavailable,
    _dtype,
    _loss_scaling,
    adapt,
)
from internal_ai.training.alpha.pin import PinnedBase, WeightShard
from internal_ai.workflows import PROMOTABLE_ADAPTER_BACKENDS

torch = pytest.importorskip("torch", reason="adaptation needs the alpha extra")
pytest.importorskip("transformers", reason="adaptation needs the alpha extra")
pytest.importorskip("peft", reason="adaptation needs the alpha extra")

WORDS = ("the", "ledger", "failed", "nightly", "reconciliation", "[UNK]")
TEXTS = ["the ledger failed", "nightly reconciliation failed", "the ledger reconciliation failed"]


def tiny_snapshot(root: Path) -> PinnedBase:
    """A two-layer Qwen3 saved to disk as the pinned layout, with its shard hashed."""
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    directory = root / "Example--tiny" / "rev1"
    directory.mkdir(parents=True)
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=len(WORDS),
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(directory, safe_serialization=True)
    path = directory / "model.safetensors"
    payload = path.read_bytes()
    return PinnedBase(
        repo_id="Example/tiny",
        revision="rev1",
        license="Apache-2.0",
        shards=(WeightShard(path.name, hashlib.sha256(payload).hexdigest(), len(payload)),),
    )


def test_training_lowers_the_loss_it_reports(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)

    result = adapt(
        base,
        tmp_path,
        TEXTS,
        tmp_path / "out",
        rank=8,
        alpha=16,
        max_seq_len=16,
        epochs=20,
        learning_rate=5e-3,
        seed=0,
    )

    assert result["loss_last"] < result["loss_first"]
    assert result["diverged_at"] is None
    assert result["trainable_parameters"] > 0
    assert result["layers"] == 2


def test_the_adapter_it_saves_names_its_backend(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)
    out = tmp_path / "out"

    result = adapt(base, tmp_path, TEXTS, out, max_seq_len=16, epochs=1, seed=0)

    assert result["adapter_backend"] == BACKEND
    assert (out / "adapter_config.json").is_file()
    assert (out / "adapter_model.safetensors").is_file()
    assert result["adapter_sha256"]
    assert BACKEND in PROMOTABLE_ADAPTER_BACKENDS


def test_a_tampered_shard_is_refused_before_training(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)
    shard = tmp_path / "Example--tiny" / "rev1" / "model.safetensors"
    shard.write_bytes(b"\0" * shard.stat().st_size)

    with pytest.raises(ValueError, match="pinned shard hash mismatch"):
        adapt(base, tmp_path, TEXTS, tmp_path / "out", max_seq_len=16)


def test_text_without_a_trainable_sequence_is_refused(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)

    with pytest.raises(ValueError, match="did not produce a trainable sequence"):
        adapt(base, tmp_path, ["", " "], tmp_path / "out", max_seq_len=16)


def test_the_record_beside_the_adapter_carries_the_backend(tmp_path: Path) -> None:
    """The service writes this file; here it only has to be the one promotion reads."""
    base = tiny_snapshot(tmp_path)
    out = tmp_path / "out"
    adapt(base, tmp_path, TEXTS, out, max_seq_len=16, epochs=1, seed=0)
    (out.parent / "reproducibility.json").write_text(
        json.dumps({"adapter_backend": BACKEND}), encoding="utf-8"
    )

    from internal_ai.workflows import adapter_backend_named_by

    assert adapter_backend_named_by({}, str(out)) == BACKEND


def test_adaptation_says_which_install_it_needs(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args, **kwargs):
        if name == "peft":
            raise ImportError("no peft here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(AdaptationUnavailable, match="alpha extra"):
        adapt(
            PinnedBase("Example/tiny", "rev1", "Apache-2.0", ()),
            Path("/nonexistent"),
            TEXTS,
            Path("/nonexistent-out"),
        )


class _Card:
    """A card that answers the plain bf16 question yes and the honest one no."""

    def __init__(self, *, native: bool) -> None:
        self.native = native

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        return self.native or including_emulation


class _Amp:
    @staticmethod
    def GradScaler(*_args: object, **_kwargs: object) -> str:
        return "scaler"


class _Torch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    amp = _Amp()

    def __init__(self, card: object) -> None:
        self.cuda = card


def test_emulated_bf16_is_refused_in_favour_of_fp16() -> None:
    """The measured T4 case: emulated bf16 ran six times slower than fp16."""
    assert _dtype(_Torch(_Card(native=False)), "cuda") == "float16"


def test_a_card_without_native_bf16_is_never_given_fp32() -> None:
    """fp32 needs 14.99 GiB of a 14.56 GiB card, so the old fallback could not run."""
    assert _dtype(_Torch(_Card(native=False)), "cuda") != "float32"


def test_a_native_bf16_card_keeps_bf16() -> None:
    assert _dtype(_Torch(_Card(native=True)), "cuda") == "bfloat16"


def test_a_torch_that_cannot_exclude_emulation_is_not_trusted() -> None:
    class _Unguarded:
        def is_bf16_supported(self) -> bool:
            return True

    assert _dtype(_Torch(_Unguarded()), "cuda") == "float16"


def test_the_cpu_stays_in_fp32() -> None:
    assert _dtype(_Torch(_Card(native=True)), "cpu") == "float32"


def test_half_precision_steps_are_scaled_and_fp32_ones_are_not() -> None:
    """Unscaled fp16 gradients read as divergence, not as an underflow."""
    stub = _Torch(_Card(native=False))

    assert _loss_scaling(stub, _dtype(stub, "cuda")) == "scaler"
    assert _loss_scaling(stub, _dtype(stub, "cpu")) is None
    assert _loss_scaling(stub, stub.bfloat16) is None
