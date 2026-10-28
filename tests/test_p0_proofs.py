"""The remaining P0 proofs: overfit, resume, reproducibility, lineage.

Each one is a claim the acceptance checklist will not accept on inspection. The
model has to actually memorise a tiny batch, a killed run has to continue from
its own checkpoint, the same seed has to give the same curve, and a sample of
records has to reach the source and commit the dataset claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.context import Platform
from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.records import Record
from internal_ai.training.overfit import overfit_check
from internal_ai.training.trainer import TrainConfig, Trainer
from internal_ai.workflows import build_corpus, train_tokenizer_for


def records_for(platform: Any, dataset_id: str) -> list[Record]:
    rows = platform.store.dataset_records(dataset_id)
    return [
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


def tiny_config(steps: int) -> TrainConfig:
    return TrainConfig(
        steps=steps,
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


def trainer_for(platform: Any) -> Trainer:
    return Trainer(
        platform.store,
        platform.audit,
        platform.budget,
        platform.settings.artifacts_dir,
        root=platform.settings.root,
    )


def losses(outcome: Any) -> list[float]:
    return [
        row["loss"]
        for row in (
            json.loads(line) for line in Path(outcome.metrics_path).read_text().splitlines()
        )
        if "loss" in row
    ]


def test_the_model_can_overfit_a_tiny_corpus(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    report = overfit_check(platform, lead)
    assert report["passed"] is True, report
    assert report["final_loss"] < report["first_loss"] / 4
    events = [
        event
        for event in platform.store.list_audit(action="train.overfit_check")
        if event.actor == lead.id
    ]
    assert events and str(events[0].outcome) == "success"


def test_the_overfit_check_is_denied_without_the_training_role(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Dana Owner", ["data_owner"])
    with pytest.raises(PermissionError):
        overfit_check(platform, owner)
    denied = [
        event
        for event in platform.store.list_audit(action="console.denied")
        if event.actor == owner.id
    ]
    assert denied, "a refusal is audited"


def test_a_killed_run_resumes_from_its_checkpoint(platform, corpus_path: Path) -> None:
    built = build_corpus(platform, [corpus_path], name="resume")
    tokenizer, _coverage = train_tokenizer_for(platform, built.dataset_id, vocab_size=300)
    records = records_for(platform, built.dataset_id)
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    trainer = trainer_for(platform)

    first = trainer.train(
        tiny_config(4),
        records,
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        resume=False,
        run_id="run_resume_proof",
    )
    assert first.resumed_from == 0
    assert (Path(first.checkpoint_path)).exists()

    second = trainer.train(
        tiny_config(8),
        records,
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        run_id="run_resume_proof",
    )
    assert second.resumed_from == 4, second.resumed_from
    assert second.steps == 8
    resumed = [
        event
        for event in platform.store.list_audit(action="train.resumed")
        if event.subject == "run_resume_proof"
    ]
    assert resumed, "the resume is recorded"


def test_the_same_seed_gives_the_same_curve(platform, corpus_path: Path) -> None:
    built = build_corpus(platform, [corpus_path], name="repeat")
    tokenizer, _coverage = train_tokenizer_for(platform, built.dataset_id, vocab_size=300)
    records = records_for(platform, built.dataset_id)
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    trainer = trainer_for(platform)
    curves = []
    for name in ("run_repeat_a", "run_repeat_b"):
        outcome = trainer.train(
            tiny_config(4),
            records,
            tokenizer,
            dataset_id=built.dataset_id,
            dataset_hash=dataset.content_hash,
            resume=False,
            run_id=name,
        )
        curves.append(losses(outcome))
    assert curves[0] and curves[0] == curves[1]


def test_the_console_runs_the_overfit_check(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        lead = enroll(app.state.platform, "Mo Lead", ["ml_lead"])
        assert sign_in(client, lead.name).status_code == 303
        page = client.get("/console/models")
        assert 'action="/console/models/overfit"' in page.text

        posted = client.post(
            "/console/models/overfit",
            data={"csrf_token": csrf_from(page.text)},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        after = client.get("/console/models").text
        assert "overfit check passed" in after


def test_the_dataset_page_shows_the_lineage_sample(tmp_path: Path, corpus_path: Path) -> None:
    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    lead = enroll(setup, "Mo Lead", ["ml_lead"])
    built = build_corpus(setup, [corpus_path], name="sample-page")
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        assert sign_in(client, lead.name).status_code == 303
        page = client.get(f"/console/datasets/{built.dataset_id}")
        assert page.status_code == 200, page.text
        assert "Sampled lineage:" in page.text
        assert "records trace to their source" in page.text


def test_a_sampled_trace_reaches_every_source(platform, corpus_path: Path) -> None:
    built = build_corpus(platform, [corpus_path], name="trace")
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    report = pipeline.trace_sample(built.dataset_id, limit=10)
    assert report["passed"] is True, report
    assert report["coverage"] == 1.0
    assert report["code_commit"] != "unknown"
    assert platform.store.list_audit(action="dataset.lineage_sampled")

    victim = platform.store.dataset_records(built.dataset_id)[0]
    platform.store.conn.execute(
        "UPDATE dataset_records SET source_id = 'src_ghost' WHERE dataset_id = ? AND record_key = ?",
        (built.dataset_id, victim["record_key"]),
    )
    broken = pipeline.trace_sample(built.dataset_id, limit=10)
    assert broken["passed"] is False
    assert broken["coverage"] < 1.0
    assert any(failure["record"] == victim["record_key"] for failure in broken["failures"])


def test_every_metric_line_names_the_batch_it_read(platform, corpus_path: Path) -> None:
    built = build_corpus(platform, [corpus_path], name="data-position")
    tokenizer, _coverage = train_tokenizer_for(platform, built.dataset_id, vocab_size=300)
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    outcome = trainer_for(platform).train(
        tiny_config(4),
        records_for(platform, built.dataset_id),
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        resume=False,
        run_id="run_data_position",
    )

    lines = [
        json.loads(line)
        for line in Path(outcome.metrics_path).read_text(encoding="utf-8").splitlines()
    ]

    assert [line["step"] for line in lines] == [0, 1, 2, 3]
    assert all(line["batch_hash"] for line in lines)
    paired = {(line["batch_index"], line["batch_hash"]) for line in lines}
    assert len(paired) == len({line["batch_index"] for line in lines}), "a batch keeps one hash"


def test_a_resumed_run_continues_on_the_batch_the_checkpoint_left(
    platform, corpus_path: Path
) -> None:
    built = build_corpus(platform, [corpus_path], name="resume-position")
    tokenizer, _coverage = train_tokenizer_for(platform, built.dataset_id, vocab_size=300)
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    records = records_for(platform, built.dataset_id)
    trainer = trainer_for(platform)
    first = trainer.train(
        tiny_config(4),
        records,
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        resume=False,
        run_id="run_same_batch",
    )

    torch = pytest.importorskip("torch")
    checkpoint = torch.load(first.checkpoint_path, weights_only=True)
    left_off = checkpoint["data_position"]
    first_lines = Path(first.metrics_path).read_text(encoding="utf-8").splitlines()

    second = trainer.train(
        tiny_config(8),
        records,
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        run_id="run_same_batch",
    )

    resumed_lines = Path(second.metrics_path).read_text(encoding="utf-8").splitlines()
    new_lines = [json.loads(line) for line in resumed_lines[len(first_lines) :]]
    assert second.resumed_from == 4
    assert new_lines[0]["step"] == 4
    assert new_lines[0]["batch_index"] == left_off["batch_index"]
    assert new_lines[0]["batch_hash"] == left_off["batch_hash"]
    resumed = platform.store.list_audit(action="train.resumed")[0]
    assert resumed.details["batch_hash"] == left_off["batch_hash"]


def test_a_checkpoint_whose_data_position_moved_is_refused(platform, corpus_path: Path) -> None:
    built = build_corpus(platform, [corpus_path], name="resume-moved")
    tokenizer, _coverage = train_tokenizer_for(platform, built.dataset_id, vocab_size=300)
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    records = records_for(platform, built.dataset_id)
    trainer = trainer_for(platform)
    first = trainer.train(
        tiny_config(4),
        records,
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        resume=False,
        run_id="run_moved_data",
    )

    torch = pytest.importorskip("torch")
    payload = torch.load(first.checkpoint_path, weights_only=True)
    payload["data_position"] = {"batch_index": 0, "batch_hash": "deadbeef"}
    torch.save(payload, first.checkpoint_path)

    refusal = trainer.train(
        tiny_config(8),
        records,
        tokenizer,
        dataset_id=built.dataset_id,
        dataset_hash=dataset.content_hash,
        run_id="run_moved_data",
    )

    assert refusal.status == "blocked_by_data_position"
    assert refusal.steps == 0
    assert "deadbeef" in refusal.alerts[0]
    event = platform.store.list_audit(action="train.resume_position_mismatch")[0]
    assert event.details["recorded"]["batch_hash"] == "deadbeef"
