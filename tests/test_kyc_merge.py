"""The KYC pipeline mechanics: adapt a tiny base, merge it whole, serve it.

The weights here are random initialisations saved by the test, never a
pretrained base, so no provenance gate is involved: what this proves is the
machinery the 3070 run will use - adapter in, complete model out, greedy
tokens out of the merged directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_alpha_adapt import TEXTS, tiny_snapshot

from internal_ai.training.alpha import merge as merge_module
from internal_ai.training.alpha import serve as serve_module
from internal_ai.training.alpha.adapt import adapt

torch = pytest.importorskip("torch", reason="merging needs the alpha extra")
pytest.importorskip("transformers", reason="merging needs the alpha extra")
pytest.importorskip("peft", reason="merging needs the alpha extra")


def test_a_bare_adapter_is_not_a_model(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)
    adapter = tmp_path / "adapter"
    adapt(base, tmp_path, TEXTS, adapter, max_seq_len=16, epochs=1, seed=0)
    with pytest.raises(ValueError, match="not a merged model"):
        serve_module.load_for_serving(adapter)


def test_quantised_serving_asks_for_a_budget_not_an_auto_map(tmp_path: Path, monkeypatch) -> None:
    """A merged 5.1B is 10 GB in bf16 and the card is 8 GB, so the tables spill."""
    import transformers

    seen: dict[str, object] = {}
    model_dir = tmp_path / "merged"
    model_dir.mkdir()

    class _Loader:
        @staticmethod
        def from_pretrained(directory: object, **kwargs: object) -> str:
            seen.update(kwargs)
            return "model"

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Loader)
    monkeypatch.setattr(transformers, "AutoTokenizer", _Loader)
    monkeypatch.setattr(
        serve_module,
        "quantised_budget",
        lambda: {0: "5.7GiB", "cpu": "12.0GiB"},
        raising=False,
    )

    serve_module.load_for_serving(model_dir, load_in_4bit=True)

    assert seen["max_memory"] == {0: "5.7GiB", "cpu": "12.0GiB"}
    assert "device_map" not in seen
    assert seen["quantization_config"].llm_int8_enable_fp32_cpu_offload is True


def test_merge_refuses_a_missing_adapter(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)
    with pytest.raises(ValueError, match="no adapter directory"):
        merge_module.merge_adapter(base, tmp_path, tmp_path / "absent", tmp_path / "merged")


class StubTokenizer:
    """The ids the bare test tokenizer would give, without its loading quirk."""

    eos_token_id = 5
    pad_token_id = 0

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
        assert prompt
        return {"input_ids": torch.tensor([[0, 1]])}

    def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
        return "stubbed"


def test_adapt_merge_and_serve_round_trip(tmp_path: Path) -> None:
    base = tiny_snapshot(tmp_path)
    adapter = tmp_path / "adapter"
    adapt(base, tmp_path, TEXTS, adapter, max_seq_len=16, epochs=1, seed=0)
    merged = tmp_path / "merged"
    record = merge_module.merge_adapter(base, tmp_path, adapter, merged)
    assert record["base_repo_id"] == base.repo_id
    assert record["base_revision"] == base.revision
    assert record["merged_sha256"]
    assert (merged / "model.safetensors").is_file()
    assert (merged / "merge_record.json").is_file()
    assert (merged / "tokenizer.json").is_file()
    model, _tokenizer = serve_module.load_for_serving(merged)
    first = serve_module.complete(model, StubTokenizer(), "the ledger", max_new_tokens=4)
    second = serve_module.complete(model, StubTokenizer(), "the ledger", max_new_tokens=4)
    assert first == second == "stubbed"
