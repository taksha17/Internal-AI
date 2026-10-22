"""Evaluation harness, training loop and serving behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from llmforge.data.records import Record
from llmforge.evals.harness import EvalHarness
from llmforge.evals.metrics import (
    bm25_scores,
    expected_calibration_error,
    mean_reciprocal_rank,
    ndcg_at_k,
    perplexity,
    precision_recall_f1,
    recall_at_k,
)
from llmforge.serving.service import (
    ModelRouter,
    Principal,
    RateLimiter,
    RetrievalService,
    parse_api_keys,
)
from llmforge.training.cost import estimate_cpu_run, estimate_gpu_run, training_flops


class ConstantEmbedder:
    name = "constant"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = [self.vectors.get(text, [0.0, 0.0]) for text in texts]
        array = np.asarray(rows, dtype=np.float32)
        norms = np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-9, None)
        return array / norms


def test_retrieval_metrics() -> None:
    relevance = {"q1": {"d1", "d2"}, "q2": {"d3"}}
    rankings = {"q1": ["d1", "d9", "d2"], "q2": ["d9", "d3"]}
    assert recall_at_k(relevance, rankings, 1) == pytest.approx(0.25)
    assert mean_reciprocal_rank(relevance, rankings) == pytest.approx((1.0 + 0.5) / 2)
    expected_q1 = 1.5 / (1 + 1 / 1.58496)
    expected_q2 = 1 / 1.58496
    assert ndcg_at_k(relevance, rankings, 3) == pytest.approx(
        (expected_q1 + expected_q2) / 2, abs=1e-3
    )


def test_classifier_metrics_and_calibration() -> None:
    scores = precision_recall_f1(
        ["positive", "negative", "positive", "negative"],
        ["positive", "negative", "negative", "positive"],
    )
    assert scores["precision"] == pytest.approx(0.5)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(0.5)
    assert scores["accuracy"] == pytest.approx(0.5)
    assert 0 <= expected_calibration_error([0.9, 0.1], [1, 0]) <= 1
    assert perplexity([-1.0, -1.0]) == pytest.approx(2.718281828, rel=1e-6)


def test_bm25_prefers_lexical_overlap() -> None:
    documents = ["invoice reconciliation failed", "shipment tracking delayed"]
    scores = bm25_scores("invoice reconciliation", documents)
    assert scores[0] > scores[1]


def test_frozen_eval_set_cannot_be_written(forge) -> None:
    harness = EvalHarness(forge.store, forge.audit)
    eval_set = harness.create_eval_set("v1", [{"query": "q", "relevant_ids": ["d1"]}])
    with pytest.raises(PermissionError):
        harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=False)
    harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=True)
    with pytest.raises(PermissionError):
        harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=True)
    with pytest.raises(PermissionError):
        harness.write_to_frozen_set(eval_set.id, "data_scientist")
    denied = [
        event for event in forge.audit.recent(limit=20) if event.action == "eval_set.write_attempt"
    ]
    assert denied


def test_evaluation_requires_frozen_set(forge) -> None:
    from llmforge.evals.harness import EvalHarness
    from llmforge.workflows import evaluate_embedding_model

    harness = EvalHarness(forge.store, forge.audit)
    eval_set = harness.create_eval_set("v1", [{"query": "q", "relevant_ids": ["d1"]}])
    with pytest.raises(PermissionError, match="frozen"):
        evaluate_embedding_model(forge, __file__, eval_set.id, None, "ds_1")
    with pytest.raises(KeyError):
        evaluate_embedding_model(forge, __file__, "evs_missing", None, "ds_1")


def test_promotion_requires_passing_eval_and_gate(forge) -> None:
    from llmforge.workflows import promote_model

    model = forge.store.add_model(
        __import__("llmforge.domain", fromlist=["ModelVersion"]).ModelVersion(
            id="mdl_1",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="now",
        )
    )
    with pytest.raises(PermissionError):
        promote_model(forge, model.id, gate_approved=False)
    with pytest.raises(PermissionError):
        promote_model(forge, model.id, gate_approved=True)


def test_training_respects_cost_cap_and_records_reproducibility(forge, tmp_path) -> None:
    from llmforge.data.tokenizer import train_tokenizer
    from llmforge.training.trainer import TrainConfig, Trainer

    texts = [
        "the ledgerline feed failed during the warehouse audit reconciliation run today",
        "backorderqueue depth grows when the fulfilmenthub quotas are exceeded",
    ] * 8
    tokenizer = train_tokenizer(texts, vocab_size=200)
    records = [
        Record(id=str(index), record_key=str(index), text=text, source_id="s")
        for index, text in enumerate(texts)
    ]
    trainer = Trainer(forge.store, forge.audit, forge.budget, tmp_path / "artifacts", root=tmp_path)
    outcome = trainer.train(
        TrainConfig(steps=2, batch_size=2, seq_len=32, cost_cap_usd=0.0),
        records,
        tokenizer,
        dataset_id="ds_1",
        dataset_hash="hash",
    )
    assert outcome.status == "blocked_by_cost_cap"
    assert outcome.alerts


def test_training_run_completes_and_checkpoints(forge, tmp_path) -> None:
    from llmforge.data.tokenizer import train_tokenizer
    from llmforge.training.trainer import TrainConfig, Trainer

    texts = [
        "the ledgerline feed failed during the warehouse audit reconciliation run today",
        "backorderqueue depth grows when the fulfilmenthub quotas are exceeded",
    ] * 8
    tokenizer = train_tokenizer(texts, vocab_size=200)
    records = [
        Record(id=str(index), record_key=str(index), text=text, source_id="s")
        for index, text in enumerate(texts)
    ]
    trainer = Trainer(forge.store, forge.audit, forge.budget, tmp_path / "artifacts", root=tmp_path)
    outcome = trainer.train(
        TrainConfig(steps=3, batch_size=2, seq_len=32, cost_cap_usd=10.0),
        records,
        tokenizer,
        dataset_id="ds_1",
        dataset_hash="hash",
    )
    assert outcome.status in ("completed", "completed_with_warnings")
    assert outcome.steps == 3
    assert outcome.tokens_seen > 0
    assert outcome.reproducibility["seed"] == 17
    assert outcome.reproducibility["dataset_hash"] == "hash"
    from pathlib import Path

    assert Path(outcome.checkpoint_path).exists()
    assert Path(outcome.metrics_path).exists()


def test_cost_estimates_follow_prd_formula() -> None:
    estimate = estimate_gpu_run(100_000_000, 10_000_000_000)
    assert estimate.training_flops == training_flops(100_000_000, 10_000_000_000)
    assert estimate.ideal_gpu_hours == pytest.approx(6e18 / (400e12) / 3600, rel=1e-6)
    assert estimate.planning_gpu_hours == pytest.approx(estimate.ideal_gpu_hours * 3)
    cpu = estimate_cpu_run(1_000_000, 1_000_000, tokens_per_second=1000)
    assert cpu.device == "cpu"
    assert cpu.estimated_cost_usd > 0


def test_rate_limiter_blocks_after_limit() -> None:
    limiter = RateLimiter(requests_per_minute=2)
    assert limiter.allow("u")
    assert limiter.allow("u")
    assert not limiter.allow("u")
    assert limiter.allow("other")


def test_api_key_parsing() -> None:
    principals = parse_api_keys("k1:alice:support|engineering,k2:bob")
    assert principals["k1"].user_id == "alice"
    assert "support" in principals["k1"].groups
    assert principals["k2"].groups == frozenset()


def test_retrieval_enforces_document_permissions(forge) -> None:

    class Embedder:
        name = "test"

        def embed(self, texts):
            return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    forge.store.add_dataset(
        __import__("llmforge.domain", fromlist=["DatasetVersion"]).DatasetVersion(
            id="ds_1",
            name="c",
            version=1,
            content_hash="h",
            record_count=2,
            token_count=2,
            created_at="now",
        )
    )
    forge.store.add_dataset_records(
        "ds_1",
        [
            {
                "record_key": "public",
                "payload": "public ledgerline note",
                "source_id": "s",
                "acl": [],
            },
            {
                "record_key": "secret",
                "payload": "restricted billing note",
                "source_id": "s",
                "acl": ["finance"],
            },
        ],
    )
    service = RetrievalService(forge.store, forge.audit)
    support = Principal(user_id="alice", groups=frozenset({"support"}))
    hits = service.search(Embedder(), "ds_1", "ledgerline", support, top_k=5)
    assert {hit.document_id for hit in hits} == {"public"}
    finance = Principal(user_id="bob", groups=frozenset({"finance"}))
    hits = service.search(Embedder(), "ds_1", "ledgerline", finance, top_k=5)
    assert {hit.document_id for hit in hits} == {"public", "secret"}
    contractor = Principal(user_id="carol", groups=frozenset({"contractor"}))
    hits = service.search(Embedder(), "ds_1", "ledgerline", contractor, top_k=5)
    assert {hit.document_id for hit in hits} == {"public"}
    admin = Principal(user_id="root", groups=frozenset({"platform-admins"}))
    hits = service.search(Embedder(), "ds_1", "ledgerline", admin, top_k=5)
    assert len(hits) == 2


def test_model_router_rolls_back(forge) -> None:
    from llmforge.domain import ModelVersion

    for index in (1, 2):
        forge.store.add_model(
            ModelVersion(
                id=f"mdl_{index}",
                name="embedding-v1",
                version=index,
                content_hash="h",
                created_at="now",
                status="promoted",
            )
        )
    router = ModelRouter(forge.store, forge.audit)
    router.set_active("embedding-v1", "mdl_2", actor="admin", reason="deploy")
    assert router.resolve("embedding-v1").id == "mdl_2"
    router.rollback("embedding-v1", "mdl_1", actor="admin")
    assert router.resolve("embedding-v1").id == "mdl_1"
    events = [
        event
        for event in forge.audit.recent(limit=20)
        if event.action == "serving.version_activated"
    ]
    assert len(events) == 2
