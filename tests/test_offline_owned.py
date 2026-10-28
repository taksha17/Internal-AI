"""Offline is the default, and training text is a released company dataset."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import httpx
import pytest
import torch

from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.llm import AnthropicAdapter, LocalRunnerAdapter, private_runner_url
from internal_ai.training.alpha.answer import cited_answer
from internal_ai.training.alpha.citations import AskHit
from internal_ai.training.alpha.fetch import fetch_base
from internal_ai.training.alpha.pin import PinnedBase, WeightShard, verify_snapshot
from internal_ai.training.alpha.qwen import QwenShape, layer_forward
from internal_ai.training.ownership import require_released_dataset


def test_anthropic_does_not_call_out_while_offline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "1")

    def boom(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("network")

    monkeypatch.setattr(httpx, "Client", boom)
    adapter = AnthropicAdapter(model="x", api_key="test-key")
    with pytest.raises(RuntimeError, match="offline"):
        adapter.complete("hello")


def test_fetch_does_not_download_while_offline(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "1")

    def boom(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    payload = b"pinned-weights"
    base = PinnedBase(
        repo_id="Example/Base",
        revision="abc",
        license="Apache-2.0",
        shards=(
            WeightShard("model.safetensors", hashlib.sha256(payload).hexdigest(), len(payload)),
        ),
    )
    directory = base.directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(payload)
    assert verify_snapshot(base, tmp_path) == directory
    with pytest.raises(RuntimeError, match="offline"):
        fetch_base(base, tmp_path)


def test_training_requires_a_released_dataset() -> None:
    released = DatasetVersion(
        id="ds_ok",
        name="company",
        version=1,
        content_hash="h",
        record_count=1,
        token_count=1,
        created_at=now_iso(),
        released=True,
    )
    assert require_released_dataset(released).id == "ds_ok"
    draft = DatasetVersion(
        id="ds_draft",
        name="company",
        version=2,
        content_hash="h2",
        record_count=1,
        token_count=1,
        created_at=now_iso(),
        released=False,
    )
    with pytest.raises(PermissionError, match="new version"):
        require_released_dataset(draft)
    with pytest.raises(KeyError, match="unknown dataset"):
        require_released_dataset(None)


def test_local_runner_rejects_a_public_host() -> None:
    with pytest.raises(ValueError, match="private"):
        private_runner_url("https://api.openai.com/v1")
    assert private_runner_url("http://192.168.1.68:11434/v1").endswith("/v1")
    assert private_runner_url("http://127.0.0.1:11434/v1").endswith("/v1")


def test_local_runner_posts_only_to_the_private_host(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    sent: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "from the lan"}}], "model": "local"}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
            return None

        def post(self, url: str, json: dict) -> FakeResponse:
            sent["url"] = url
            sent["body"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    adapter = LocalRunnerAdapter(base_url="http://10.0.0.8:11434/v1", model="qwen")
    completion = adapter.complete("where is the host allow-list?")
    assert completion.text == "from the lan"
    assert sent["url"] == "http://10.0.0.8:11434/v1/chat/completions"


def test_a_mixed_citation_is_replaced_by_the_hits() -> None:
    class Mixed:
        name = "fake"

        def complete(
            self, prompt: str, system: str = "", max_tokens: int = 512, temperature: float = 0.0
        ):  # type: ignore[no-untyped-def]
            from internal_ai.llm import Completion

            return Completion(
                text=(
                    "See Acmy_Website/src/app/api/payments/razorpay/create-order/route.ts "
                    "at line 36."
                ),
                model="fake",
            )

    hits = [
        AskHit("Acmy_Website/next.config.ts", 36),
        AskHit("Acmy_Website/src/app/api/payments/razorpay/create-order/route.ts", 8),
    ]
    answer = cited_answer(Mixed(), "where is the checkout host?", hits)
    assert answer.passed is False
    assert "next.config.ts:36" in answer.text
    assert "line 36" not in answer.text


def test_one_qwen_layer_keeps_its_shape() -> None:
    shape = QwenShape(8, 1, 4, 2, 4, 16, 32, 1e-6, 10000.0)
    weights = {
        "input_layernorm.weight": torch.ones(shape.hidden),
        "post_attention_layernorm.weight": torch.ones(shape.hidden),
        "self_attn.q_norm.weight": torch.ones(shape.head_dim),
        "self_attn.k_norm.weight": torch.ones(shape.head_dim),
        "self_attn.q_proj.weight": torch.randn(shape.heads * shape.head_dim, shape.hidden) * 0.02,
        "self_attn.k_proj.weight": torch.randn(shape.kv_heads * shape.head_dim, shape.hidden)
        * 0.02,
        "self_attn.v_proj.weight": torch.randn(shape.kv_heads * shape.head_dim, shape.hidden)
        * 0.02,
        "self_attn.o_proj.weight": torch.randn(shape.hidden, shape.heads * shape.head_dim) * 0.02,
        "mlp.gate_proj.weight": torch.randn(shape.intermediate, shape.hidden) * 0.02,
        "mlp.up_proj.weight": torch.randn(shape.intermediate, shape.hidden) * 0.02,
        "mlp.down_proj.weight": torch.randn(shape.hidden, shape.intermediate) * 0.02,
    }
    left = torch.randn(8, shape.hidden) * 0.02
    right = torch.zeros(shape.heads * shape.head_dim, 8)
    hidden = torch.randn(1, 4, shape.hidden)
    output = layer_forward(hidden, weights, shape, left, right, scale=2.0)
    assert output.shape == hidden.shape
