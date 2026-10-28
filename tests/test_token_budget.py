"""Training waits for a measured token count above the parameter floor."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from programme import record_ready_programme

from internal_ai.data.tokenizer import Tokenizer
from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.training.trainer import TrainConfig
from internal_ai.workflows import (
    _embedding_parameters,
    train_classifier_model,
    train_embedding_model,
)


def _ready(platform, enroll_person, tokens: int, kind: str) -> Tokenizer:
    owner = enroll_person(platform, "Mo Builder", ["ml_lead"])
    record_ready_programme(platform, owner)
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_tokens",
            name="tickets",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=tokens,
            created_at=now_iso(),
            released=True,
            stats={"token_count_kind": kind},
        )
    )
    platform.store.add_dataset_records(
        "ds_tokens",
        [{"record_key": "a", "payload": "the ledgerline feed failed during the warehouse audit"}],
    )
    return Tokenizer(vocab={"<pad>": 0, "<unk>": 1, "a": 2}, merges=[])


def _denied(platform) -> str:
    row = platform.store.conn.execute(
        "SELECT details FROM audit_events WHERE action = 'train.token_budget' ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    return row["details"]


def test_an_estimate_is_denied(platform, enroll_person) -> None:
    tokenizer = _ready(platform, enroll_person, tokens=10**9, kind="estimate")
    with pytest.raises(PermissionError, match="ds_tokens is still an estimate"):
        train_embedding_model(platform, "ds_tokens", tokenizer, TrainConfig(steps=1))
    assert "estimate" in _denied(platform)


def test_a_measured_corpus_under_the_floor_names_both_numbers(platform, enroll_person) -> None:
    tokenizer = _ready(platform, enroll_person, tokens=1, kind="measured")
    config = TrainConfig(steps=1, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    parameters = _embedding_parameters(config, tokenizer)
    with pytest.raises(PermissionError, match="1 tokens") as caught:
        train_embedding_model(platform, "ds_tokens", tokenizer, config)
    assert str(parameters) in str(caught.value)
    details = _denied(platform)
    assert '"tokens":1' in details.replace(" ", "")
    assert str(parameters) in details


def test_a_measured_corpus_above_the_floor_reaches_the_trainer(
    platform, enroll_person, monkeypatch
) -> None:
    tokenizer = _ready(platform, enroll_person, tokens=1, kind="measured")
    config = TrainConfig(steps=1, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    parameters = _embedding_parameters(config, tokenizer)
    dataset = platform.store.get_dataset("ds_tokens")
    stats = dict(dataset.stats)
    platform.store.update_dataset_tokens("ds_tokens", parameters * 21, stats)

    def reached(self, *args, **kwargs):
        raise RuntimeError("reached the trainer")

    monkeypatch.setattr("internal_ai.training.trainer.Trainer.train", reached)
    with pytest.raises(RuntimeError, match="reached the trainer"):
        train_embedding_model(platform, "ds_tokens", tokenizer, config)


def test_the_brief_can_raise_the_floor(platform, enroll_person) -> None:
    from programme import actor_named

    from internal_ai.brief import Metric, load, save

    tokenizer = _ready(platform, enroll_person, tokens=1, kind="measured")
    config = TrainConfig(steps=1, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    parameters = _embedding_parameters(config, tokenizer)
    dataset = platform.store.get_dataset("ds_tokens")
    stats = dict(dataset.stats)
    platform.store.update_dataset_tokens("ds_tokens", parameters * 21, stats)
    brief = load(platform.store)
    brief.metrics = [Metric(name="tokens_per_parameter", threshold="1000")]
    save(platform.store, platform.audit, actor_named(platform, "Mo Builder"), brief)
    with pytest.raises(PermissionError, match="1000 tokens per parameter"):
        train_embedding_model(platform, "ds_tokens", tokenizer, config)


def test_a_classifier_on_an_estimate_is_denied(platform, enroll_person, tmp_path: Path) -> None:
    _ready(platform, enroll_person, tokens=10**9, kind="")
    encoder = tmp_path / "encoder.pt"
    torch.save({"model_state": {"weight": torch.zeros(4, 4)}}, encoder)
    with pytest.raises(PermissionError, match="still an estimate"):
        train_classifier_model(
            platform, "ds_tokens", encoder, Tokenizer(vocab={"<pad>": 0}, merges=[])
        )
