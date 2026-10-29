"""Evaluation harness, training loop and serving behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from internal_ai.data.records import Record
from internal_ai.evals.harness import EvalHarness
from internal_ai.evals.metrics import (
    bm25_scores,
    expected_calibration_error,
    mean_reciprocal_rank,
    ndcg_at_k,
    perplexity,
    precision_recall_f1,
    recall_at_k,
)
from internal_ai.serving.service import (
    ModelRouter,
    Principal,
    RateLimiter,
    RetrievalService,
    parse_api_keys,
)
from internal_ai.training.cost import estimate_cpu_run, estimate_gpu_run, training_flops


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


def test_frozen_eval_set_cannot_be_written(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set("v1", [{"query": "q", "relevant_ids": ["d1"]}])
    with pytest.raises(PermissionError):
        harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=False)
    harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=True)
    with pytest.raises(PermissionError):
        harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=True)
    with pytest.raises(PermissionError):
        harness.write_to_frozen_set(eval_set.id, "data_scientist")
    denied = [
        event
        for event in platform.audit.recent(limit=20)
        if event.action == "eval_set.write_attempt"
    ]
    assert denied


def test_evaluation_requires_frozen_set(platform) -> None:
    from internal_ai.evals.harness import EvalHarness
    from internal_ai.workflows import evaluate_embedding_model

    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set("v1", [{"query": "q", "relevant_ids": ["d1"]}])
    with pytest.raises(PermissionError, match="frozen"):
        evaluate_embedding_model(platform, __file__, eval_set.id, None, "ds_1")
    with pytest.raises(KeyError):
        evaluate_embedding_model(platform, __file__, "evs_missing", None, "ds_1")


def test_promotion_requires_passing_eval_and_gate(platform) -> None:
    from internal_ai.workflows import promote_model

    model = platform.store.add_model(
        __import__("internal_ai.domain", fromlist=["ModelVersion"]).ModelVersion(
            id="mdl_1",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="now",
        )
    )
    with pytest.raises(PermissionError):
        promote_model(platform, model.id, gate_approved=False)
    with pytest.raises(PermissionError):
        promote_model(platform, model.id, gate_approved=True)


def test_a_passing_report_off_a_frozen_set_cannot_promote(platform) -> None:
    from internal_ai.domain import EvalReport, ModelVersion
    from internal_ai.evals.harness import EvalHarness
    from internal_ai.workflows import promote_model

    model = platform.store.add_model(
        ModelVersion(
            id="mdl_named_like_a_file",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="now",
        )
    )
    harness = EvalHarness(platform.store, platform.audit)
    open_set = harness.create_eval_set("open", [{"query": "q", "relevant_ids": ["d1"]}])

    def report(report_id: str, eval_set_id: str, created_at: str) -> None:
        platform.store.add_eval_report(
            EvalReport(
                id=report_id,
                model_id=model.id,
                eval_set_id=eval_set_id,
                created_at=created_at,
                metrics={"recall_at_10": 1.0},
                passed=True,
                notes="pass",
            )
        )

    report("rep_open", open_set.id, "2026-01-01T00:00:01+00:00")
    with pytest.raises(PermissionError, match="frozen"):
        promote_model(platform, model.id, gate_approved=True)
    harness.freeze_eval_set(open_set.id, "ml_lead", gate_approved=True)
    report("rep_frozen", open_set.id, "2026-01-01T00:00:02+00:00")
    report("rep_classifier", f"classifier::{model.id}", "2026-01-01T00:00:03+00:00")
    with pytest.raises(PermissionError, match="frozen"):
        promote_model(platform, model.id, gate_approved=True)
    report("rep_frozen_again", open_set.id, "2026-01-01T00:00:04+00:00")
    promoted = promote_model(platform, model.id, gate_approved=True)
    assert promoted.status == "promoted"
    assert promoted.eval_report_id == "rep_frozen_again"


def test_a_failed_report_on_a_frozen_set_cannot_promote(platform) -> None:
    from internal_ai.domain import EvalReport, ModelVersion
    from internal_ai.evals.harness import EvalHarness
    from internal_ai.workflows import promote_model

    model = platform.store.add_model(
        ModelVersion(
            id="mdl_failed",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="now",
        )
    )
    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set("v1", [{"query": "q", "relevant_ids": ["d1"]}])
    harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=True)
    platform.store.add_eval_report(
        EvalReport(
            id="rep_failed",
            model_id=model.id,
            eval_set_id=eval_set.id,
            created_at="2026-01-01T00:00:01+00:00",
            metrics={"recall_at_10": 1.0},
            baseline_metrics={"recall_at_10": 0.0},
            checks={
                "pii_regurgitation": {"passed": False},
                "contamination": {"passed": False},
            },
            passed=False,
            notes="PII regurgitation and training leakage both failed",
        )
    )
    with pytest.raises(PermissionError):
        promote_model(platform, model.id, gate_approved=True)
    assert platform.store.get_model(model.id).status == "candidate"


def test_serving_reports_the_model_that_is_actually_served(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from internal_ai.context import Platform
    from internal_ai.domain import ModelVersion
    from internal_ai.serving.app import create_serving_app
    from internal_ai.serving.service import Principal

    platform = Platform.create(root=tmp_path)
    platform.store.add_model(
        ModelVersion(
            id="mdl_candidate",
            name="embedding-v1",
            version=1,
            content_hash="h1",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    platform.store.add_model(
        ModelVersion(
            id="mdl_promoted",
            name="embedding-v1",
            version=2,
            content_hash="h2",
            created_at="2026-01-02T00:00:00+00:00",
            status="promoted",
        )
    )
    platform.close()

    app = create_serving_app(
        root=tmp_path,
        principals={
            "k": Principal(user_id="alice", groups=frozenset({"engineering"}), api_key="k")
        },
    )
    with TestClient(app) as client:
        listed = client.get("/v1/models", headers={"X-Internal-Key": "k"}).json()
        served = {entry["id"]: entry["active"] for entry in listed}
        assert served["mdl_promoted"] is True
        assert served["mdl_candidate"] is False

        with pytest.raises(PermissionError, match="promoted"):
            app.state.router.set_active("embedding-v1", "mdl_candidate", "admin", "rollback")
        listed = client.get("/v1/models", headers={"X-Internal-Key": "k"}).json()
        served = {entry["id"]: entry["active"] for entry in listed}
        assert served["mdl_promoted"] is True
        assert served["mdl_candidate"] is False

        app.state.router.set_active("embedding-v1", "mdl_promoted", "admin", "deploy")
        listed = client.get("/v1/models", headers={"X-Internal-Key": "k"}).json()
        served = {entry["id"]: entry["active"] for entry in listed}
        assert served["mdl_promoted"] is True
        assert served["mdl_candidate"] is False


def test_title_mode_pairs_use_first_line(platform) -> None:
    from internal_ai.domain import DatasetVersion
    from internal_ai.workflows import build_pair_dataset

    platform.store.add_dataset(
        DatasetVersion(
            id="ds_pairs",
            name="pairs",
            version=1,
            content_hash="h",
            record_count=3,
            token_count=3,
            created_at="now",
        )
    )
    platform.store.add_dataset_records(
        "ds_pairs",
        [
            {
                "record_key": "doc-1",
                "payload": "# Architecture\n\nDesign notes for the control plane.",
                "source_id": "s",
                "acl": [],
            },
            {
                "record_key": "doc-2",
                "payload": "import json\n\nfrom pathlib import Path",
                "source_id": "s",
                "acl": [],
            },
            {"record_key": "doc-3", "payload": "tiny", "source_id": "s", "acl": []},
        ],
    )
    pairs = build_pair_dataset(platform, "ds_pairs", mode="title")
    assert [pair["query"] for pair in pairs] == ["# Architecture", "import json"]
    assert all(pair["document"].startswith(pair["query"].split("\n")[0]) for pair in pairs)
    paraphrase = build_pair_dataset(platform, "ds_pairs", mode="paraphrase")
    assert paraphrase == []


def test_training_respects_cost_cap_and_records_reproducibility(platform, tmp_path) -> None:
    from internal_ai.data.tokenizer import train_tokenizer
    from internal_ai.training.trainer import TrainConfig, Trainer

    texts = [
        "the ledgerline feed failed during the warehouse audit reconciliation run today",
        "backorderqueue depth grows when the fulfilmenthub quotas are exceeded",
    ] * 8
    tokenizer = train_tokenizer(texts, vocab_size=200)
    records = [
        Record(id=str(index), record_key=str(index), text=text, source_id="s")
        for index, text in enumerate(texts)
    ]
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
    outcome = trainer.train(
        TrainConfig(steps=2, batch_size=2, seq_len=32, cost_cap_usd=0.0),
        records,
        tokenizer,
        dataset_id="ds_1",
        dataset_hash="hash",
    )
    assert outcome.status == "blocked_by_cost_cap"
    assert outcome.alerts


def test_training_run_completes_and_checkpoints(platform, tmp_path) -> None:
    from internal_ai.data.tokenizer import train_tokenizer
    from internal_ai.training.trainer import TrainConfig, Trainer

    texts = [
        "the ledgerline feed failed during the warehouse audit reconciliation run today",
        "backorderqueue depth grows when the fulfilmenthub quotas are exceeded",
    ] * 8
    tokenizer = train_tokenizer(texts, vocab_size=200)
    records = [
        Record(id=str(index), record_key=str(index), text=text, source_id="s")
        for index, text in enumerate(texts)
    ]
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
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


def test_retrieval_enforces_document_permissions(platform) -> None:

    class Embedder:
        name = "test"

        def embed(self, texts):
            return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    platform.store.add_dataset(
        __import__("internal_ai.domain", fromlist=["DatasetVersion"]).DatasetVersion(
            id="ds_1",
            name="c",
            version=1,
            content_hash="h",
            record_count=2,
            token_count=2,
            created_at="now",
        )
    )
    platform.store.add_dataset_records(
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
    service = RetrievalService(platform.store, platform.audit)
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


def test_model_router_rolls_back(platform) -> None:
    from internal_ai.domain import ModelVersion

    for index in (1, 2):
        platform.store.add_model(
            ModelVersion(
                id=f"mdl_{index}",
                name="embedding-v1",
                version=index,
                content_hash="h",
                created_at="now",
                status="promoted",
            )
        )
    router = ModelRouter(platform.store, platform.audit)
    router.set_active("embedding-v1", "mdl_2", actor="admin", reason="deploy")
    assert router.resolve("embedding-v1").id == "mdl_2"
    router.rollback("embedding-v1", "mdl_1", actor="admin")
    assert router.resolve("embedding-v1").id == "mdl_1"
    events = [
        event
        for event in platform.audit.recent(limit=20)
        if event.action == "serving.version_activated"
    ]
    assert len(events) == 2


def test_served_model_uses_a_promoted_model_when_the_sample_name_is_absent(platform) -> None:
    from internal_ai.domain import ModelVersion
    from internal_ai.serving.service import served_model

    platform.store.add_model(
        ModelVersion(
            id="mdl_named",
            name="new-volume-3",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            status="promoted",
        )
    )
    router = ModelRouter(platform.store, platform.audit)
    assert router.resolve("embedding-v1") is None
    assert served_model(platform.store, router).id == "mdl_named"


def test_batched_embeddings_match_one_forward(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from internal_ai.data.tokenizer import train_tokenizer
    from internal_ai.training.embedder import TorchEmbedder
    from internal_ai.training.model import TinyTransformer, TransformerConfig

    texts = ["alpha beta", "gamma", "delta epsilon zeta eta"] * 3
    tokenizer = train_tokenizer(texts, vocab_size=80)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=32,
    )
    path = tmp_path / "encoder.pt"
    torch.manual_seed(0)
    torch.save(
        {
            "model_state": TinyTransformer(config).state_dict(),
            "model_config": config.to_dict(),
        },
        path,
    )
    embedder = TorchEmbedder(path, tokenizer, max_seq_len=32)
    whole = embedder.embed(texts, batch_size=len(texts))
    batched = embedder.embed(texts, batch_size=2)
    assert whole.shape == batched.shape == (len(texts), config.d_model)
    assert np.allclose(whole, batched, atol=1e-5)
