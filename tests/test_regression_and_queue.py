"""A new candidate is evaluated by someone else, and a regression must be accepted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import enroll, paired_scores

from internal_ai.data.tokenizer import train_tokenizer
from internal_ai.domain import EvalReport, EvalSet, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.services import build
from internal_ai.services.jobs import JobRunner
from internal_ai.workflows import promote_model, recall_regression


def test_registering_a_model_queues_an_eval_when_a_frozen_set_exists(
    platform, tmp_path: Path
) -> None:
    tokenizer = train_tokenizer(["reset the vpn profile"], vocab_size=80)
    token_path = tmp_path / "tokenizer.json"
    tokenizer.save(token_path)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"not a checkpoint")
    dataset_id = "ds_queue"
    platform.store.add_eval_set(
        EvalSet(
            id="evs_queue",
            name="queue",
            version=1,
            frozen=True,
            created_at=now_iso(),
            items=[{"query": "vpn", "relevant_ids": ["1"], "dataset_id": dataset_id}],
        )
    )
    builder = enroll(platform, "Mo Lead", ["ml_lead"])
    enroll(platform, "Mia Lead", ["ml_lead"])
    result = {
        "run_id": "run_queue",
        "status": "completed",
        "name": "tiny",
        "model_path": str(model_path),
        "dataset_id": dataset_id,
        "tokenizer_path": str(token_path),
        "steps": 1,
        "tokens_seen": 8,
        "final_loss": 1.0,
        "cost_usd": 0.0,
        "checkpoint_path": "",
        "metrics_path": "",
        "alerts": [],
        "warnings": [],
        "estimate": {},
        "reproducibility": {},
        "token_budget": {},
        "resumed_from": 0,
    }
    platform.store.conn.execute(
        "INSERT INTO console_jobs (id, kind, title, requested_by, state, params, result,"
        " created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            "cjob_queue",
            "train_model",
            "train",
            builder.id,
            "succeeded",
            "{}",
            json.dumps(result),
            now_iso(),
        ),
    )

    build.register_model(platform, builder, JobRunner(platform, synchronous=True), "cjob_queue")

    kinds = [
        row["kind"]
        for row in platform.store.conn.execute("SELECT kind FROM console_jobs").fetchall()
    ]
    assert "evaluate_model" in kinds


def test_a_regressed_model_needs_the_word_accepted_from_someone_else(platform) -> None:
    platform.store.add_eval_set(
        EvalSet(
            id="evs_reg",
            name="reg",
            version=1,
            frozen=True,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    platform.store.add_model(
        ModelVersion(
            id="mdl_old",
            name="tiny",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            dataset_ids=["ds_reg"],
            card={"registered_by": "ppl_mo"},
        )
    )
    platform.store.add_model(
        ModelVersion(
            id="mdl_new",
            name="tiny",
            version=2,
            content_hash="h2",
            created_at="2026-01-02T00:00:00+00:00",
            dataset_ids=["ds_reg"],
            card={"registered_by": "ppl_mo"},
        )
    )
    for model_id, recall, created in (
        ("mdl_old", 0.5, "2026-01-01T00:00:00+00:00"),
        ("mdl_new", 0.2, "2026-01-02T00:00:00+00:00"),
    ):
        platform.store.add_eval_report(
            EvalReport(
                id=f"rep_{model_id}",
                model_id=model_id,
                eval_set_id="evs_reg",
                created_at=created,
                metrics={"recall_at_10": recall},
                baseline_metrics={"recall_at_10": 0.1},
                per_case=paired_scores("recall_at_10", int(recall * 100), 10, 100),
                passed=True,
                judged_by="ppl_mia",
            )
        )

    drop = recall_regression(platform, platform.store.get_model("mdl_new"))
    assert drop is not None and drop["drop"] > 0.05
    with pytest.raises(PermissionError, match="accepted") as refused:
        promote_model(platform, "mdl_new", gate_approved=True, approver="ppl_datta", note="")
    assert "recall_at_10" in str(refused.value)
    assert "0.50 -> 0.20" in str(refused.value)
    with pytest.raises(PermissionError, match="someone other than its builder"):
        promote_model(platform, "mdl_new", gate_approved=True, approver="ppl_mo", note="accepted")
    promoted = promote_model(
        platform, "mdl_new", gate_approved=True, approver="ppl_datta", note="accepted the drop"
    )
    assert promoted.status == "promoted"


def test_a_regression_names_the_question_the_candidate_lost(platform) -> None:
    queries = [
        "how do I reset my VPN",
        "where is the expense policy",
        "who approves a refund",
    ]
    platform.store.add_eval_set(
        EvalSet(
            id="evs_named",
            name="named",
            version=1,
            frozen=True,
            created_at="2026-01-01T00:00:00+00:00",
            content_hash="c",
            items=[
                {"query": query, "relevant_ids": [f"r{index}"]}
                for index, query in enumerate(queries)
            ],
        )
    )
    for model_id, found, created, version in (
        ("mdl_prev", [1.0, 1.0, 1.0], "2026-01-01T00:00:00+00:00", 1),
        ("mdl_lost", [1.0, 0.0, 1.0], "2026-01-02T00:00:00+00:00", 2),
    ):
        platform.store.add_model(
            ModelVersion(
                id=model_id,
                name="tiny",
                version=version,
                content_hash=model_id,
                created_at=created,
                card={"registered_by": "ppl_mo"},
            )
        )
        platform.store.add_eval_report(
            EvalReport(
                id=f"rep_{model_id}",
                model_id=model_id,
                eval_set_id="evs_named",
                created_at=created,
                metrics={"recall_at_10": sum(found) / len(found)},
                baseline_metrics={"recall_at_10": 0.1},
                per_case={"recall_at_10": {"candidate": found, "baseline": [0.0, 0.0, 0.0]}},
                passed=True,
                judged_by="ppl_mia",
            )
        )

    with pytest.raises(PermissionError) as refused:
        promote_model(platform, "mdl_lost", gate_approved=True, approver="ppl_datta", note="")

    message = str(refused.value)
    assert "recall_at_10" in message
    assert "1.00 -> 0.67" in message
    assert "where is the expense policy" in message
