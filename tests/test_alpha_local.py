"""The answer can be produced here, from the pinned weights, with nothing else running."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from internal_ai.domain import ModelVersion
from internal_ai.ids import now_iso
from internal_ai.llm import LocalRunnerAdapter
from internal_ai.training.alpha.local import (
    LocalGenerationUnavailable,
    LocalModelAdapter,
    answer_adapter,
    base_for,
    generate,
    load_adapter,
)
from internal_ai.training.alpha.pin import PRIMARY, PinnedBase, WeightShard
from internal_ai.training.provenance import ProvenanceRefused

torch = pytest.importorskip("torch", reason="local generation needs the alpha extra")
save_file = pytest.importorskip(
    "safetensors.torch", reason="local generation needs the alpha extra"
).save_file

WORDS = ("hello", "world", "the", "ledger", "failed", "[UNK]")


def write_tokenizer(directory: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))


def tiny_base(root: Path, *, with_tokenizer: bool = False) -> PinnedBase:
    """One 8-wide decoder layer, enough to run a forward pass and produce a token."""
    directory = root / "Example--tiny" / "rev1"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        """{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
"num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16, "vocab_size": 6,
"rms_norm_eps": 1e-6, "rope_theta": 10000.0}""",
        encoding="utf-8",
    )
    if with_tokenizer:
        write_tokenizer(directory)
    torch.manual_seed(0)
    tensors = {
        "model.embed_tokens.weight": torch.randn(6, 8) * 0.02,
        "model.norm.weight": torch.ones(8),
        "model.layers.0.input_layernorm.weight": torch.ones(8),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(8),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(4),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8) * 0.02,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8) * 0.02,
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8) * 0.02,
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8) * 0.02,
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16) * 0.02,
    }
    path = directory / "model.safetensors"
    save_file(tensors, str(path))
    payload = path.read_bytes()
    return PinnedBase(
        repo_id="Example/tiny",
        revision="rev1",
        license="Apache-2.0",
        shards=(WeightShard(path.name, hashlib.sha256(payload).hexdigest(), len(payload)),),
    )


def card_model(model_id: str = "mdl_z", **card) -> ModelVersion:
    values = {"base_repo_id": PRIMARY.repo_id, "base_revision": PRIMARY.revision}
    values.update(card)
    return ModelVersion(
        id=model_id,
        name="qwen3-4b",
        version=1,
        content_hash="h",
        created_at=now_iso(),
        status="candidate",
        config={},
        card=values,
        path="",
    )


def test_generation_reads_the_weights_on_disk(tmp_path: Path) -> None:
    base = tiny_base(tmp_path, with_tokenizer=True)

    first = generate(base, tmp_path, "hello world", max_tokens=3)
    second = generate(base, tmp_path, "hello world", max_tokens=3)

    assert first == second
    assert all(word in WORDS for word in first.split())


def test_an_adapter_is_read_as_pairs_per_layer(tmp_path: Path) -> None:
    base = tiny_base(tmp_path, with_tokenizer=True)
    adapter = tmp_path / "adapter.pt"
    torch.save({"layers": {"0": {"A": torch.randn(8, 8) * 0.02, "B": torch.zeros(8, 8)}}}, adapter)

    pairs = load_adapter(adapter)

    assert list(pairs) == [0]
    assert pairs[0][0].shape == (8, 8)
    with_adapter = generate(base, tmp_path, "hello", adapter, max_tokens=2)
    assert with_adapter == generate(base, tmp_path, "hello", adapter, max_tokens=2)


def test_an_adapter_with_no_layers_is_refused(tmp_path: Path) -> None:
    adapter = tmp_path / "empty.pt"
    torch.save({"layers": {}}, adapter)

    with pytest.raises(ValueError, match="no adapted layers"):
        load_adapter(adapter)


def test_a_card_must_name_a_pinned_base() -> None:
    model = ModelVersion(
        id="mdl_x",
        name="tiny",
        version=1,
        content_hash="h",
        created_at=now_iso(),
        status="candidate",
        config={},
        card={},
        path="",
    )

    with pytest.raises(ValueError, match="does not name a pinned base"):
        base_for(model)


def test_the_pinned_base_is_matched_on_revision_not_name() -> None:
    with pytest.raises(ValueError, match="does not name a pinned base"):
        base_for(card_model(base_revision="not-the-pin"))


def test_a_missing_snapshot_is_reported_before_generation(platform) -> None:
    adapter = LocalModelAdapter(platform, card_model())

    with pytest.raises(FileNotFoundError, match="missing pinned shard"):
        adapter.complete("hello")


def test_a_temperature_is_refused_because_generation_is_greedy(platform) -> None:
    with pytest.raises(ValueError, match="greedy"):
        LocalModelAdapter(platform, card_model()).complete("hello", temperature=0.7)


def test_the_local_path_refuses_while_the_install_is_scratch_only(platform) -> None:
    with pytest.raises(ProvenanceRefused, match="scratch_only"):
        answer_adapter(platform, card_model("mdl_s"))


def test_a_configured_runner_is_still_preferred(platform, monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", "http://127.0.0.1:11434/v1")

    assert isinstance(answer_adapter(platform, card_model("mdl_r")), LocalRunnerAdapter)


def test_the_refusal_names_the_extra_when_the_tokenizer_is_absent(monkeypatch, tmp_path) -> None:
    base = tiny_base(tmp_path, with_tokenizer=False)
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    with pytest.raises(LocalGenerationUnavailable, match="alpha extra"):
        generate(base, tmp_path, "hello", max_tokens=1)
