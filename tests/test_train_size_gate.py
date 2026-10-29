"""Training starts only on a size the compute advice marks runnable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from programme import record_ready_programme

from internal_ai.compute.advise import advise_dataset
from internal_ai.data.tokenizer import Tokenizer
from internal_ai.domain import DatasetVersion, EvalSet
from internal_ai.ids import now_iso
from internal_ai.services.build import train_model
from internal_ai.services.jobs import JobRunner
from internal_ai.training.trainer import TrainConfig, TrainingOutcome
from internal_ai.workflows import register_model


def _jobs(platform) -> int:
    return int(
        platform.store.conn.execute("SELECT COUNT(*) AS n FROM console_jobs").fetchone()["n"]
    )


def _corpus(platform, token_count: int = 200_000_000, kind: str = "measured") -> str:
    dataset_id = "ds_size"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="tickets",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=token_count,
            created_at=now_iso(),
            released=True,
            stats={"token_count_kind": kind},
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": "rec-1",
                "payload": "the ledgerline feed failed during the warehouse audit",
                "acl": ["engineering"],
            }
        ],
    )
    return dataset_id


def _tokenizer(platform, dataset_id: str) -> Path:
    tokenizer = Tokenizer(vocab={"<pad>": 0, "<unk>": 1, "the": 2, "feed": 3}, merges=[])
    path = tokenizer.save(platform.settings.models_dir / f"tokenizer-{dataset_id}.json")
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest(), encoding="utf-8"
    )
    return path


def _frozen_eval(platform, dataset_id: str) -> None:
    platform.store.add_eval_set(
        EvalSet(
            id="evs_size",
            name="tickets-questions",
            version=1,
            frozen=True,
            created_at=now_iso(),
            items=[
                {
                    "dataset_id": dataset_id,
                    "filed_by": "ppl_expert",
                    "query": "why did the feed fail",
                    "relevant_ids": ["rec-1"],
                }
            ],
        )
    )


def test_no_advice_stored_refuses_before_any_step(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    dataset_id = _corpus(platform)
    with pytest.raises(PermissionError, match="record what this corpus needs"):
        train_model(
            platform,
            lead,
            JobRunner(platform, synchronous=True),
            dataset_id,
            size="small",
            vocab_rationale="600 tokens cover the ticket jargon",
        )
    assert platform.store.list_models() == []
    assert _jobs(platform) == 0


def test_a_waiting_size_names_its_gpu_hours_and_starts_nothing(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    dataset_id = _corpus(platform)
    advice = advise_dataset(platform, dataset_id)
    seven_b = next(row for row in advice["sizes"] if row["size"] == "7b")
    assert seven_b["runnable"] is False
    with pytest.raises(PermissionError, match="7b is waiting") as caught:
        train_model(
            platform,
            lead,
            JobRunner(platform, synchronous=True),
            dataset_id,
            size="7b",
            vocab_rationale="600 tokens cover the ticket jargon",
        )
    assert str(seven_b["planning_gpu_hours"]) in str(caught.value)
    assert platform.store.list_models() == []
    assert _jobs(platform) == 0


def test_a_runnable_small_run_completes_and_the_card_says_small(platform, enroll_person) -> None:
    from internal_ai.budget import BudgetScope

    record_ready_programme(platform, enroll_person(platform, "Ada Lead", ["program_lead"]))
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    dataset_id = _corpus(platform)
    _tokenizer(platform, dataset_id)
    _frozen_eval(platform, dataset_id)
    advise_dataset(platform, dataset_id)
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 50)

    runner = JobRunner(platform, synchronous=True)
    job = train_model(
        platform,
        lead,
        runner,
        dataset_id,
        steps=1,
        size="small",
        vocab_rationale="600 tokens cover the ticket jargon",
    )
    stored = runner.get(job["id"])
    assert stored is not None
    assert stored["state"] == "succeeded", stored
    assert stored["result"]["size"] == "small"

    from internal_ai.services.build import register_model as register

    model = register(platform, lead, runner, job["id"])
    assert model.card["size"] == "small"
    assert model.card["track"] == "from_scratch"
    assert model.card["planning_gpu_hours"] > 0
    assert model.card.get("track") != "alpha"


def test_the_shape_follows_the_size() -> None:
    small = TrainConfig.for_size("small")
    seven_b = TrainConfig.for_size("7b")
    assert small.d_model == 128 and small.n_layers == 4
    assert seven_b.d_model == 4096 and seven_b.n_layers == 32
    assert seven_b.size == "7b"


def test_a_card_records_the_size_from_the_run(platform) -> None:
    outcome = TrainingOutcome(
        run_id="run_size",
        status="completed",
        steps=1,
        tokens_seen=8,
        final_loss=1.0,
        cost_usd=0.0,
        checkpoint_path="",
        metrics_path="",
        model_path=str(platform.settings.artifacts_dir / "m.pt"),
        estimate={"planning_gpu_hours": 0.25},
        reproducibility={"config": {"size": "1b"}, "commit": "abc", "seed": 17},
    )
    dataset_id = _corpus(platform)
    model = register_model(
        platform, "sized", Path(outcome.model_path), outcome, dataset_id, actor="ml_engineer"
    )
    assert model.card["size"] == "1b"
    assert model.card["planning_gpu_hours"] == 0.25
    assert model.card["track"] == "from_scratch"
    assert json.dumps(model.card)  # the card stays JSON-serialisable for the store
