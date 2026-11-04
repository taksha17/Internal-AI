"""B1: the adapter the platform promotes is a PEFT directory, so this platform serves it.

Promotion accepts only the full-depth PEFT backend, whose adapter is a
directory hashed by its tree. The old serving path understood the streamed
rank-8 ``.pt`` the platform no longer promotes, refused to load the directory,
and wrote a manifest without ever verifying what the runner was running. The
tests here hold the local serving path to the card: a real PEFT adapter
directory, saved by peft beside a real pinned snapshot, loads only when its
tree hash matches, answers on this machine, and reports the hash that is
answering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_alpha_adapt import tiny_snapshot

from internal_ai.domain import ModelVersion
from internal_ai.ids import tree_hash
from internal_ai.training.provenance import OPEN_WEIGHT_ALLOWED

torch = pytest.importorskip("torch", reason="serving the adapter needs the alpha extra")
pytest.importorskip("transformers", reason="serving the adapter needs the alpha extra")
pytest.importorskip("peft", reason="serving the adapter needs the alpha extra")

HEALTH = {"backend": "transformers-peft"}


def enable_open_weights(platform) -> None:
    """An install that decided it may adapt a base (the P6 story)."""
    decision = platform.policy.get("P6")
    decision.current_value = OPEN_WEIGHT_ALLOWED
    decision.status = "decided"
    platform.store.upsert_policy(decision)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from internal_ai.training.alpha import backend

    monkeypatch.setattr(backend, "LOADED", {})


def peft_adapter(snapshot: Path, adapter_dir: Path) -> None:
    """Save what the promotable backend saves: a real PEFT adapter directory."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(snapshot)
    adapted = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    adapted.save_pretrained(adapter_dir)


def registered(platform, adapter_dir: Path) -> ModelVersion:
    return platform.store.add_model(
        ModelVersion(
            id="mdl_serve",
            name="tiny-alpha",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            path=str(adapter_dir),
            dataset_ids=["ds_alpha"],
            card={
                "registered_by": "ppl_lead",
                "base_repo_id": "Example/tiny",
                "base_revision": "rev1",
                "adapter_backend": "peft",
                "adapter_sha256": tree_hash(adapter_dir),
                "trained_from_scratch": False,
                "pretrained_weights_used": True,
            },
        )
    )


def _prepared(tmp_path: Path, platform, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, str]:
    """A scratch install whose pin names the two-layer Qwen3 on disk."""
    import internal_ai.training.alpha.local as local_module

    enable_open_weights(platform)
    base = tiny_snapshot(tmp_path)
    snapshot = tmp_path / "Example--tiny" / "rev1"
    adapter_dir = tmp_path / "adapter"
    peft_adapter(snapshot, adapter_dir)
    monkeypatch.setattr(local_module, "PINNED", (base,))
    return base, registered(platform, adapter_dir), tree_hash(adapter_dir)


def test_a_promoted_peft_directory_answers_on_this_machine(
    tmp_path: Path, platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    from internal_ai.training.alpha import backend
    from internal_ai.training.alpha.local import answer_adapter

    _base, model, digest = _prepared(tmp_path, platform, monkeypatch)

    adapter = answer_adapter(platform, model)
    completion = adapter.complete("Question: the ledger failed\nAnswer:")

    assert completion.text.strip(), completion.text
    assert completion.output_tokens > 0
    assert backend.LOADED["mdl_serve"] == digest
    assert adapter.name == "transformers-peft"


def test_a_directory_whose_tree_drifted_is_refused(
    tmp_path: Path, platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    from internal_ai.training.alpha.local import answer_adapter

    _base, model, _digest = _prepared(tmp_path, platform, monkeypatch)
    (Path(model.path) / "extra.bin").write_bytes(b"tampered")

    with pytest.raises(PermissionError, match="tree_hash"):
        answer_adapter(platform, model)


def test_a_card_that_names_no_hash_is_refused(
    tmp_path: Path, platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    from internal_ai.training.alpha.local import answer_adapter

    _base, model, _digest = _prepared(tmp_path, platform, monkeypatch)
    model.card.pop("adapter_sha256")
    platform.store.update_model_card(model.id, model.card)

    with pytest.raises(PermissionError, match="no adapter hash on the card"):
        answer_adapter(platform, model)


def test_health_names_the_hash_that_is_answering(
    tmp_path: Path, platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from internal_ai.serving.app import create_serving_app
    from internal_ai.serving.service import Principal
    from internal_ai.training.alpha.local import answer_adapter

    _base, model, digest = _prepared(tmp_path, platform, monkeypatch)
    answer_adapter(platform, model).complete("Question: the ledger failed\nAnswer:")

    app = create_serving_app(
        root=tmp_path,
        principals={"k": Principal(user_id="ada", groups=frozenset(), api_key="k")},
    )
    with TestClient(app) as client:
        health = client.get("/health").json()

    assert health["loaded_adapters"] == {"mdl_serve": digest}
