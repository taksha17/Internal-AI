"""Only an adapter trained at full depth may be promoted."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from internal_ai.workflows import (
    PROMOTABLE_ADAPTER_BACKENDS,
    adapter_backend_named_by,
    promote_model,
)


def test_only_a_full_depth_backend_is_promotable() -> None:
    assert "peft" in PROMOTABLE_ADAPTER_BACKENDS
    assert len(PROMOTABLE_ADAPTER_BACKENDS) == 1


def test_the_card_is_read_before_the_run_record(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"weights")
    (tmp_path / "reproducibility.json").write_text(
        json.dumps({"adapter_backend": "stream_from_disk"}), encoding="utf-8"
    )

    assert adapter_backend_named_by({"adapter_backend": "peft"}, str(adapter)) == "peft"


def test_a_card_that_names_nothing_falls_back_to_the_run_record(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"weights")
    (tmp_path / "reproducibility.json").write_text(
        json.dumps({"recipe": {"backend": "stream_from_disk"}}), encoding="utf-8"
    )

    assert adapter_backend_named_by({}, str(adapter)) == "stream_from_disk"


def test_an_adapter_with_no_card_and_no_record_is_not_promotable() -> None:
    assert adapter_backend_named_by({}) == ""
    assert adapter_backend_named_by({}) not in PROMOTABLE_ADAPTER_BACKENDS


def test_promotion_refuses_a_streamed_adapter_before_reading_a_report(
    platform, monkeypatch
) -> None:
    model = SimpleNamespace(
        id="mdl_streamed",
        card={"base_repo_id": "Qwen/Qwen3-4B", "adapter_backend": "stream_from_disk"},
        path="",
    )
    monkeypatch.setattr(platform.store, "get_model", lambda _id: model)

    with pytest.raises(PermissionError, match="truncated"):
        promote_model(platform, "mdl_streamed", gate_approved=True)


def test_a_full_depth_adapter_reaches_the_gate_check(platform, monkeypatch) -> None:
    model = SimpleNamespace(
        id="mdl_peft",
        card={"base_repo_id": "Qwen/Qwen3-4B", "adapter_backend": "peft"},
        path="",
    )
    monkeypatch.setattr(platform.store, "get_model", lambda _id: model)

    with pytest.raises(PermissionError, match="model_promotion gate"):
        promote_model(platform, "mdl_peft", gate_approved=False)


def test_a_from_scratch_model_is_not_subject_to_the_adapter_rule(platform, monkeypatch) -> None:
    model = SimpleNamespace(
        id="mdl_scratch",
        card={"trained_from_scratch": True, "pretrained_weights_used": False},
        path="",
    )
    monkeypatch.setattr(platform.store, "get_model", lambda _id: model)

    with pytest.raises(PermissionError, match="model_promotion gate"):
        promote_model(platform, "mdl_scratch", gate_approved=False)
