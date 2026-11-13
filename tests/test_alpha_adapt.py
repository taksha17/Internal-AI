"""The full-depth backend is judged on the objective generation actually runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from internal_ai.training.alpha import adapt as alpha_adapt
from internal_ai.training.alpha.adapt import (
    BACKEND,
    DIVERGENCE_WINDOW,
    IGNORE_INDEX,
    PAD_ID,
    TARGET_MODULES,
    AdaptationUnavailable,
    _batches,
    _dtype,
    _landed_placement,
    _load_model,
    _lora_targets,
    _loss_scaling,
    _p50,
    _padded,
    _window_exploded,
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


def test_a_divergence_in_a_later_epoch_refuses_the_run(tmp_path: Path) -> None:
    """The B6 repro: one text, six epochs, an absurd learning rate.

    The tiny model holds a normal learning rate no matter how large, so the loss
    stays finite until a step the schedule has wound up; the divergence then
    lands after the first pass, where the old guard read it as a finished run and
    saved the adapter anyway.
    """
    base = tiny_snapshot(tmp_path)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="the loss diverged"):
        adapt(
            base,
            tmp_path,
            ["the ledger failed"],
            out,
            max_seq_len=16,
            epochs=6,
            learning_rate=2e4,
            seed=0,
        )

    assert not out.exists()


def test_a_batch_pads_the_short_sequence_and_ignores_the_padding() -> None:
    """Padding teaches the prediction of pad unless it is masked and ignored."""
    rows, masks, labels = _padded([[1, 2, 3], [4, 5]])

    assert rows == [[1, 2, 3], [4, 5, PAD_ID]]
    assert masks == [[1, 1, 1], [1, 1, 0]]
    assert labels == [[1, 2, 3], [4, 5, IGNORE_INDEX]]


def test_batches_are_grouped_by_length_so_padding_stays_cheap() -> None:
    """File order would pad every short sequence up to the longest in the file."""
    groups = _batches([[1], [2, 3, 4], [5, 6]], 2)

    assert [[len(ids) for ids in group] for group in groups] == [[1, 2], [3]]
    assert sorted(len(ids) for group in groups for ids in group) == [1, 2, 3]


def test_one_sequence_per_step_is_unchanged_by_default() -> None:
    """The unbatched order is the order the text arrived in."""
    groups = _batches([[1], [2, 3, 4], [5, 6]], 1)

    assert groups == [[[1]], [[2, 3, 4]], [[5, 6]]]


def test_a_batched_run_steps_once_per_accumulated_group(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)

    result = adapt(
        base,
        tmp_path,
        TEXTS,
        tmp_path / "out",
        max_seq_len=16,
        epochs=1,
        seed=0,
        batch_size=2,
        gradient_accumulation=2,
    )

    assert result["sequences"] == 3
    assert result["batch_size"] == 2
    assert result["gradient_accumulation"] == 2
    # Three sequences group into two batches, and both batches make one step.
    assert result["optimizer_steps"] == 1
    assert result["tokens"] > 0
    assert result["tokens_per_second"] > 0
    assert result["parallelism"] == "cpu"
    assert result["diverged_at"] is None


def test_an_unbatched_run_steps_once_per_sequence(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)

    result = adapt(base, tmp_path, TEXTS, tmp_path / "out", max_seq_len=16, epochs=1, seed=0)

    assert result["optimizer_steps"] == result["sequences"] == 3
    assert result["batch_size"] == 1
    assert result["gradient_accumulation"] == 1
    assert len([result["step_seconds_p50"]]) == 1
    assert _p50([]) == 0.0


def test_the_divergence_guard_reads_a_window_not_a_single_sequence() -> None:
    """One hard sequence is not divergence; a run whose newest window is far above
    its first window is.
    """
    assert not _window_exploded([2.0, 8.0, 2.0, 2.0, 2.0, 2.0])
    assert _window_exploded([2.0] * DIVERGENCE_WINDOW + [8.0] * DIVERGENCE_WINDOW)


class _Placed:
    """A model that reports where its layers went, the way an accelerate load does."""

    def __init__(self, mapping: dict[str, object] | None) -> None:
        if mapping is not None:
            self.hf_device_map = mapping


def test_the_record_says_where_the_layers_landed_not_only_what_was_asked() -> None:
    """The 4B A/B recorded `auto` and a pipeline; neither says which card worked."""
    placed = _Placed({"model.embed_tokens": 0, "model.layers.0": "cuda:1"})

    assert _landed_placement(placed) == {
        "model.embed_tokens": "0",
        "model.layers.0": "cuda:1",
    }


def test_a_model_placed_on_one_card_reports_one_device() -> None:
    """A quantised 7B fits a single T4, so a run claiming two cards is wrong."""
    landed = _landed_placement(_Placed({"": 0}))

    assert list(landed.values()) == ["0"]


def test_a_model_without_a_device_map_reports_nothing_rather_than_a_guess() -> None:
    assert _landed_placement(_Placed(None)) == {}


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


def _recording_loader(seen: dict[str, object]):
    class _Loader:
        @staticmethod
        def from_pretrained(directory: object, **kwargs: object) -> str:
            seen.update(kwargs)
            seen["directory"] = directory
            return "model"

    return _Loader


def _card(*, present: bool = True, free_mib: int = 6852, total_mib: int = 8192):
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return present

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            return (free_mib * 1024**2, total_mib * 1024**2)

    class _Torch:
        cuda = _Cuda()

    return _Torch()


def test_a_card_too_small_for_the_bf16_embedding_tables_spills_instead_of_refusing(
    monkeypatch,
) -> None:
    """Gemma 4 E2B keeps 2.77B parameters in bf16 embeddings, ~5.5 GB beside the body.

    No 8 GB card holds both, and without a budget accelerate answers with a refusal
    rather than a placement, so the run never starts.
    """
    import transformers

    seen: dict[str, object] = {}
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _recording_loader(seen))

    _load_model(Path("base"), "bfloat16", "cuda", load_in_4bit=True, torch_module=_card())

    budget = seen["max_memory"]
    assert "device_map" not in seen
    assert budget[0] == "5.7GiB"
    assert float(budget["cpu"].removesuffix("GiB")) > 0
    assert seen["quantization_config"].llm_int8_enable_fp32_cpu_offload is True


def test_a_quantised_load_without_a_card_asks_for_no_budget(monkeypatch) -> None:
    import transformers

    seen: dict[str, object] = {}
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _recording_loader(seen))

    _load_model(
        Path("base"), "bfloat16", "cuda", load_in_4bit=True, torch_module=_card(present=False)
    )

    assert "max_memory" not in seen
    assert seen["device_map"] == "auto"
    assert seen["quantization_config"].llm_int8_enable_fp32_cpu_offload is False


def test_the_host_budget_is_read_from_the_machine_not_a_container_limit(
    tmp_path: Path, monkeypatch
) -> None:
    """sysconf said 7.3 GiB on a 15 GiB box, and that budget spilled the tables to disk."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       15728640 kB\nMemFree:  1024 kB\n", encoding="utf-8")
    monkeypatch.setattr(alpha_adapt, "HOST_MEMINFO", meminfo)

    assert alpha_adapt._host_ram_gib() == pytest.approx(15.0, abs=1e-6)


def test_a_wrapped_projection_is_reached_through_all_linear() -> None:
    """Gemma 4 names every projection ``linear`` inside its own clippable block, so
    peft finds no ``q_proj`` to wrap and refuses the run instead of skipping it."""

    class _Wrapped:
        def named_modules(self):
            yield "", object()
            yield "model.language_model.layers.0.self_attn.q_proj.linear", object()
            yield "model.language_model.layers.0.mlp.down_proj.linear", object()

    assert _lora_targets(_Wrapped()) == "all-linear"


def test_a_base_that_names_its_projections_keeps_the_pinned_targets() -> None:
    class _Plain:
        def named_modules(self):
            yield "", object()
            yield "model.layers.0.self_attn.q_proj", object()
            yield "model.layers.0.mlp.down_proj", object()

    assert _lora_targets(_Plain()) == list(TARGET_MODULES)


def test_half_precision_steps_are_scaled_and_fp32_ones_are_not() -> None:
    """Unscaled fp16 gradients read as divergence, not as an underflow."""
    stub = _Torch(_Card(native=False))

    assert _loss_scaling(stub, _dtype(stub, "cuda")) == "scaler"
    assert _loss_scaling(stub, _dtype(stub, "cpu")) is None
    assert _loss_scaling(stub, stub.bfloat16) is None
