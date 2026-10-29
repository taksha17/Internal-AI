"""A tokenizer is frozen with its hash, and a changed file cannot train."""

from __future__ import annotations

import hashlib

import pytest
from programme import record_ready_programme

from internal_ai.data.samples import generate_corpus
from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.training.trainer import TrainConfig
from internal_ai.workflows import train_embedding_model, train_tokenizer_for


def test_retraining_a_frozen_tokenizer_raises(platform) -> None:
    path = platform.settings.root / "data" / "freeze.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_corpus(path, records=8, duplicate_ratio=0, pii_ratio=0, seed=6)
    from internal_ai.workflows import build_corpus

    dataset_id = build_corpus(platform, [path], name="freeze").dataset_id
    train_tokenizer_for(platform, dataset_id, vocab_size=200)
    digest = platform.settings.models_dir / f"tokenizer-{dataset_id}.sha256"
    assert digest.is_file()
    with pytest.raises(PermissionError, match="frozen"):
        train_tokenizer_for(platform, dataset_id, vocab_size=200)


def test_a_mismatched_hash_and_a_blank_rationale_are_denied(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Mo Builder", ["ml_lead"])
    record_ready_programme(platform, owner)
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_freeze",
            name="tickets",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=10**12,
            created_at=now_iso(),
            released=True,
            stats={"token_count_kind": "measured"},
        )
    )
    platform.store.add_dataset_records(
        "ds_freeze",
        [{"record_key": "a", "payload": "the ledgerline feed failed during the warehouse audit"}],
    )
    from internal_ai.data.tokenizer import Tokenizer

    tokenizer = Tokenizer(vocab={"<pad>": 0, "<unk>": 1, "a": 2}, merges=[])
    path = platform.settings.models_dir / "tokenizer-ds_freeze.json"
    tokenizer.save(path)
    path.with_suffix(".sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    config = TrainConfig(
        steps=1,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        vocab_rationale="covers the ticket words",
    )
    with pytest.raises(PermissionError, match="does not match the freeze"):
        train_embedding_model(platform, "ds_freeze", tokenizer, config)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="vocabulary size"):
        train_embedding_model(
            platform,
            "ds_freeze",
            tokenizer,
            TrainConfig(steps=1, d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_rationale="  "),
        )
