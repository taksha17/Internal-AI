"""The acceptance run's cheap holes, closed.

Each test here exists because a report could look clean without measuring
anything: a token count estimated from characters, a contamination check that
passed when it could not run, and a permission-leak check that was a string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.records import Record
from internal_ai.evals.harness import EvalHarness
from internal_ai.training.trainer import TrainConfig, Trainer
from internal_ai.workflows import build_corpus, train_tokenizer_for


class HashEmbedder:
    """A deterministic stand-in, so the checks are what is under test."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(8, dtype=float)
            for index, char in enumerate(text):
                vector[index % 8] += (ord(char) % 13) / 13
            vectors.append(vector / (np.linalg.norm(vector) or 1.0))
        return np.array(vectors)


def record(key: str, text: str, acl: list[str] | None = None) -> Record:
    return Record(
        id=key,
        record_key=key,
        text=text,
        source_id="src",
        acl=["engineering"] if acl is None else acl,
        sensitivity="internal",
    )


def outcome_for(
    platform: Any,
    corpus: list[Record],
    training: list[Record],
) -> Any:
    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set(
        "permissions",
        [{"query": "how do I reset my vpn", "relevant_ids": [corpus[0].record_key]}],
    )
    return harness.evaluate_retrieval(HashEmbedder(), eval_set, corpus, training_records=training)


def test_the_token_count_is_measured_once_the_tokenizer_exists(platform, corpus_path: Path) -> None:
    built = build_corpus(platform, [corpus_path], name="tokens-measured")
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    assert dataset.stats["token_count_kind"] == "estimate"

    tokenizer, coverage = train_tokenizer_for(platform, built.dataset_id, vocab_size=300)
    measured = platform.store.get_dataset(built.dataset_id)
    assert measured is not None
    rows = platform.store.dataset_records(built.dataset_id)
    exact = sum(len(tokenizer.encode(row["payload"])) for row in rows)
    assert measured.token_count == exact
    assert measured.stats["token_count_kind"] == "measured"
    assert measured.stats["token_count_estimate"] == dataset.token_count
    assert coverage["tokens"] == exact


def test_a_run_records_how_many_tokens_it_had_per_parameter(platform, corpus_path: Path) -> None:
    built = build_corpus(platform, [corpus_path], name="budget")
    tokenizer, _coverage = train_tokenizer_for(platform, built.dataset_id, vocab_size=300)
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    rows = platform.store.dataset_records(built.dataset_id)
    records = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            sensitivity=row["sensitivity"],
            acl=row["acl"],
            tokens=row["tokens"],
            meta=row["meta"],
        )
        for row in rows
    ]
    trainer = Trainer(
        platform.store,
        platform.audit,
        platform.budget,
        platform.settings.artifacts_dir,
        root=platform.settings.root,
    )
    config = TrainConfig(
        steps=2,
        batch_size=2,
        seq_len=32,
        max_seq_len=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        log_interval=1,
        checkpoint_interval=2,
    )
    outcome = trainer.train(
        config,
        records,
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        resume=False,
    )
    assert outcome.token_budget["count_kind"] == "measured"
    assert outcome.token_budget["dataset_tokens"] == dataset.token_count
    assert outcome.token_budget["parameters"] > 0
    assert outcome.token_budget["meets_floor"] is False
    assert outcome.status == "completed"
    assert platform.store.list_audit(action="train.token_budget_short")


def test_contamination_fails_closed_when_it_cannot_run(platform) -> None:
    outcome = outcome_for(platform, [record("a", "vpn reset instructions for staff")], [])
    check = outcome.checks["contamination"]
    assert check["passed"] is False
    assert "could not be ruled out" in check["note"]
    assert outcome.passed is False


def test_contamination_passes_when_the_training_corpus_is_clean(platform) -> None:
    outcome = outcome_for(
        platform,
        [record("a", "vpn reset instructions for staff")],
        [record("b", "quarterly invoice reconciliation runbook")],
    )
    assert outcome.checks["contamination"]["passed"] is True


def test_a_document_without_an_access_list_fails_the_probe(platform) -> None:
    outcome = outcome_for(platform, [record("a", "vpn reset instructions", acl=[])], [])
    check = outcome.checks["permission_leak"]
    assert check["passed"] is False
    assert check["without_access_list"] == ["a"]
    assert outcome.passed is False


def test_the_probe_passes_when_only_the_granted_group_can_read(platform) -> None:
    outcome = outcome_for(platform, [record("a", "vpn reset instructions")], [])
    check = outcome.checks["permission_leak"]
    assert check["passed"] is True, check
    assert check["probed"] == 1


def test_the_release_audit_refuses_a_surviving_api_token(platform, corpus_path: Path) -> None:
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    built = build_corpus(platform, [corpus_path], name="secrets")
    dirty = platform.store.dataset_records(built.dataset_id)[0]
    platform.store.conn.execute(
        "UPDATE dataset_records SET payload = ? WHERE dataset_id = ? AND record_key = ?",
        (
            "deploy with sk-ABCDEFGHIJKLMNOPQRST and rotate later",
            built.dataset_id,
            dirty["record_key"],
        ),
    )
    with pytest.raises(PermissionError, match="release audit"):
        pipeline.release(built.dataset_id, "ml_lead", gate_approved=True)
