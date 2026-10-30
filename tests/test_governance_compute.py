"""Alerts, sub-agents, governance, compute plane and classifier paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from internal_ai.alerts import AlertEngine
from internal_ai.compute.scheduler import JobScheduler, JobSpec, JobState
from internal_ai.data.governance import RetentionManager, pii_spot_check
from internal_ai.data.labels import BILLING_LABEL_SET, build_labeled_dataset
from internal_ai.domain import BudgetScope, TaskKind
from internal_ai.roles import all_roles, get_role
from internal_ai.sandbox import Sandbox
from internal_ai.subagents import subagents_for, total_subagents
from internal_ai.workflows import build_corpus


def test_every_role_declares_subagents() -> None:
    for role in all_roles():
        assert subagents_for(role.name), f"{role.name} has no sub-agents"
    assert total_subagents() >= 40


def test_subagents_run_and_write_artifacts(platform) -> None:
    from internal_ai.domain import Task

    task = Task(
        id="tsk_1",
        title="Build the corpus",
        description="",
        kind=TaskKind.DATA,
        owner_role="data_engineer",
        acceptance_criteria=["corpus is clean", "lineage is complete"],
    )
    platform.store.add_task(task)
    result = platform.runtime.execute(get_role("data_engineer"), task)
    sub_agents = result.metadata["sub_agents"]
    assert len(sub_agents) == len(subagents_for("data_engineer"))
    assert all(item["status"] == "ok" for item in sub_agents)
    artifact_names = {Path(path).name for path in result.artifacts}
    assert any(name.startswith("subagent-tsk_1-") for name in artifact_names)
    events = [
        event for event in platform.audit.recent(limit=200) if event.action == "agent.subagent"
    ]
    assert len(events) == len(sub_agents)


def test_subagents_do_not_spawn_further_agents(platform) -> None:
    from internal_ai.domain import Task

    task = Task(
        id="tsk_2",
        title="Plan",
        description="",
        kind=TaskKind.PLAN,
        owner_role="orchestrator",
        acceptance_criteria=["planned"],
    )
    platform.store.add_task(task)
    platform.runtime.execute(get_role("orchestrator"), task, depth=1)
    events = [
        event for event in platform.audit.recent(limit=200) if event.action == "agent.subagent"
    ]
    assert events == []


def test_alerts_fire_on_spend_pressure(platform) -> None:
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 10.0)
    platform.budget.record_spend(9.0, agent="ml_engineer")
    engine = AlertEngine(platform.store, platform.audit, platform.budget)
    alerts = engine.check_spend()
    assert any(alert.kind == "spend_approaching_cap" for alert in alerts)
    platform.budget.record_spend(2.0, agent="ml_engineer")
    alerts = engine.check_spend()
    assert any(alert.kind == "budget_exhausted" for alert in alerts)


def test_alerts_fire_on_repeated_task_failures(platform) -> None:
    from internal_ai.domain import Task

    task = Task(
        id="tsk_3",
        title="Flaky",
        description="",
        kind=TaskKind.OPS,
        owner_role="orchestrator",
        attempts=3,
    )
    platform.store.add_task(task)
    engine = AlertEngine(platform.store, platform.audit, platform.budget)
    alerts = engine.check_tasks(attempt_threshold=3)
    assert any(alert.kind == "repeated_task_failures" for alert in alerts)


def test_alerts_detect_anomalous_data_access(platform) -> None:
    for index in range(30):
        platform.audit.record(
            actor="bulk_reader",
            action="serving.search",
            subject="ds_1",
            inputs={"query_hash": f"h{index}"},
        )
    for index in range(5):
        platform.audit.record(
            actor="normal_reader",
            action="serving.search",
            subject="ds_1",
            inputs={"query_hash": f"n{index}"},
        )
    engine = AlertEngine(platform.store, platform.audit, platform.budget)
    alerts = engine.check_data_access()
    assert any(alert.kind == "anomalous_data_access" for alert in alerts)
    assert any(alert.subject == "bulk_reader" for alert in alerts)


def test_alerts_can_be_acknowledged(platform) -> None:
    engine = AlertEngine(platform.store, platform.audit, platform.budget)
    alert = engine.raise_alert("test", "low", "subject", "message")
    assert alert.id in {item.id for item in engine.list_alerts()}
    engine.acknowledge(alert.id, "program_lead")
    assert alert.id not in {item.id for item in engine.list_alerts(unacknowledged_only=True)}


def test_pii_audit_passes_on_scrubbed_records() -> None:
    from internal_ai.data.records import Record

    records = [
        Record(
            id="1",
            record_key="a",
            text="the ledgerline feed failed for account acct-1234 in the emea region",
            source_id="s",
        )
    ]
    result = pii_spot_check(records, sample_size=10)
    assert result.passed
    assert result.critical_findings == 0


def test_a_zero_sample_still_reads_every_record() -> None:
    from internal_ai.data.records import Record

    records = [
        Record(id=str(index), record_key=f"r{index}", text="clean ledger note", source_id="s")
        for index in range(5)
    ]
    records.append(
        Record(
            id="dirty",
            record_key="dirty",
            text="contact person@example.com about the invoice",
            source_id="s",
        )
    )
    result = pii_spot_check(records, sample_size=0)
    assert not result.passed
    assert result.sampled == len(records)


def test_pii_audit_catches_surviving_pii() -> None:
    from internal_ai.data.records import Record

    records = [
        Record(
            id="1",
            record_key="a",
            text="contact person@example.com about the invoice reconciliation drift",
            source_id="s",
        )
    ]
    result = pii_spot_check(records, sample_size=10)
    assert not result.passed
    assert result.critical_findings >= 1
    assert result.findings.get("email", 0) >= 1


def test_release_is_blocked_when_pii_survives(platform, corpus_path, monkeypatch) -> None:
    from internal_ai.data.pipeline import DataPipeline

    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    result = build_corpus(platform, [corpus_path], name="corpus-test")
    rows = platform.store.dataset_records(result.dataset_id)
    dirty = rows[0]
    platform.store.conn.execute(
        "UPDATE dataset_records SET payload = ? WHERE dataset_id = ? AND record_key = ?",
        (
            "reach me at person@example.com about the invoice reconciliation",
            result.dataset_id,
            dirty["record_key"],
        ),
    )
    with pytest.raises(PermissionError, match="release audit"):
        pipeline.release(result.dataset_id, "ml_lead", gate_approved=True)


def test_a_released_dataset_cannot_be_split_again(platform) -> None:
    from internal_ai.domain import DatasetVersion
    from internal_ai.workflows import split_corpus

    dataset = platform.store.add_dataset(
        DatasetVersion(
            id="ds_released_split",
            name="released-split",
            version=1,
            content_hash="h",
            record_count=1,
            token_count=1,
            created_at="now",
            released=True,
        )
    )
    platform.store.add_dataset_records(
        dataset.id, [{"record_key": "a", "payload": "clean ledger note", "split": "test"}]
    )
    with pytest.raises(PermissionError, match="immutable"):
        split_corpus(platform, dataset.id, seed="other-seed")
    assert platform.store.dataset_records(dataset.id)[0]["split"] == "test"


def test_release_reads_a_record_a_one_row_sample_would_miss(platform) -> None:
    from internal_ai.data.pipeline import DataPipeline
    from internal_ai.domain import DatasetVersion

    dataset = platform.store.add_dataset(
        DatasetVersion(
            id="ds_hidden_pii",
            name="hidden-pii",
            version=1,
            content_hash="h",
            record_count=3,
            token_count=3,
            created_at="now",
        )
    )
    platform.store.add_dataset_records(
        dataset.id,
        [
            {"record_key": "a", "payload": "clean ledger note"},
            {"record_key": "b", "payload": "another clean ledger note"},
            {"record_key": "c", "payload": "contact person@example.com about the invoice"},
        ],
    )
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    with pytest.raises(PermissionError, match="release audit"):
        pipeline.release(dataset.id, "ml_lead", gate_approved=True)


def test_release_requires_the_gate(platform, corpus_path) -> None:
    from internal_ai.data.pipeline import DataPipeline

    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    result = build_corpus(platform, [corpus_path], name="corpus-test")
    from internal_ai.data.language import accept_language_sample

    with pytest.raises(PermissionError, match="dataset_release gate"):
        pipeline.release(result.dataset_id, "ml_lead", gate_approved=False)
    accept_language_sample(platform.store, result.dataset_id)
    counted = platform.store.get_dataset(result.dataset_id)
    stats = dict(counted.stats)
    stats["token_count_kind"] = "measured"
    platform.store.update_dataset_tokens(result.dataset_id, counted.token_count, stats)
    dataset = pipeline.release(result.dataset_id, "ml_lead", gate_approved=True)
    assert dataset.released


def test_withdrawn_source_is_excluded_from_new_versions(platform, corpus_path) -> None:
    manager = RetentionManager(platform.store, platform.audit)
    manager.withdraw_source(corpus_path.stem, "data owner request", "data_owner")
    result = build_corpus(platform, [corpus_path], name="corpus-after-withdrawal")
    assert result.records == 0
    events = [
        event
        for event in platform.audit.recent(limit=50)
        if event.action == "data.source_withdrawn_skipped"
    ]
    assert events


def test_purge_removes_records_from_datasets(platform, corpus_path) -> None:
    result = build_corpus(platform, [corpus_path], name="corpus-test")
    before = len(platform.store.dataset_records(result.dataset_id))
    manager = RetentionManager(platform.store, platform.audit)
    removed = manager.purge_source_records(corpus_path.stem, "data_owner", "right to erasure")
    assert removed == before
    assert platform.store.dataset_records(result.dataset_id) == []


def test_job_scheduler_rejects_over_cap_jobs(platform) -> None:
    scheduler = JobScheduler(
        platform.audit, platform.budget, Sandbox(platform.settings.sandbox_dir)
    )
    job = scheduler.submit(
        JobSpec(
            name="big",
            command=["echo", "hello"],
            agent="ml_engineer",
            tokens=10_000_000_000,
            parameters=1_000_000_000,
            cost_cap_usd=0.0001,
        )
    )
    assert job.state == JobState.REJECTED
    assert "cap" in job.rejection_reason
    denied = [event for event in platform.audit.recent(limit=20) if event.action == "job.rejected"]
    assert denied


def test_job_scheduler_runs_a_job(platform) -> None:
    scheduler = JobScheduler(
        platform.audit, platform.budget, Sandbox(platform.settings.sandbox_dir)
    )
    job = scheduler.submit(
        JobSpec(
            name="smoke",
            command=[sys.executable, "-c", "print('hello')"],
            agent="infra_engineer",
            tokens=1000,
            parameters=1000,
            cost_cap_usd=1.0,
        )
    )
    job = scheduler.run(job)
    assert job.state == JobState.SUCCEEDED
    assert "hello" in job.stdout


def test_job_queue_is_priority_ordered(platform) -> None:
    scheduler = JobScheduler(
        platform.audit, platform.budget, Sandbox(platform.settings.sandbox_dir)
    )
    low = scheduler.submit(
        JobSpec(name="low", command=["echo"], agent="a", priority=1, tokens=1, parameters=1)
    )
    high = scheduler.submit(
        JobSpec(name="high", command=["echo"], agent="a", priority=9, tokens=1, parameters=1)
    )
    assert next(job.id for job in scheduler.queue()) == high.id
    assert low.id in {job.id for job in scheduler.queue()}


def test_job_quota_limits_concurrency(platform) -> None:
    scheduler = JobScheduler(
        platform.audit, platform.budget, Sandbox(platform.settings.sandbox_dir)
    )
    scheduler.set_quota("gpu", 1)
    first = scheduler.submit(
        JobSpec(name="a", command=["echo"], agent="a", quota="gpu", tokens=1, parameters=1)
    )
    scheduler.jobs[first.id].state = JobState.RUNNING
    second = scheduler.submit(
        JobSpec(name="b", command=["echo"], agent="a", quota="gpu", tokens=1, parameters=1)
    )
    assert scheduler.run_next() is None
    assert second.state == JobState.QUEUED


def test_label_set_scopes_to_the_subject() -> None:
    from internal_ai.data.records import Record

    records = [
        Record(
            id="1",
            record_key="a",
            text=(
                "Invoice reconciliation failed for account acct-1. "
                "Later the warehouse audit mentioned shipment tracking."
            ),
            source_id="s",
        )
    ]
    labelled = build_labeled_dataset(records)
    assert labelled[0]["label"] == "billing"
    assert BILLING_LABEL_SET.label_for("shipment tracking was delayed") == "fulfilment"


def test_classifier_trains_and_reports_per_class(platform, corpus_path, tmp_path) -> None:
    from internal_ai.data.tokenizer import train_tokenizer
    from internal_ai.training.classifier import ClassifierConfig, ClassifierTrainer
    from internal_ai.training.model import TinyTransformer, TransformerConfig

    texts = [
        "Invoice reconciliation failed for account acct-1 in the emea region.",
        "Shipment tracking stalled for account acct-2 in the amer region.",
        "Customer onboarding stalled at the licence renewal step.",
        "Warehouse audit completed and the backorderqueue was cleared.",
    ] * 10
    tokenizer = train_tokenizer(texts, vocab_size=200)
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, d_ff=64, max_seq_len=64
    )
    encoder_path = tmp_path / "encoder.pt"
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "model_state": TinyTransformer(model_config).state_dict(),
            "model_config": model_config.to_dict(),
        },
        encoder_path,
    )
    labelled = build_labeled_dataset(
        [
            type("R", (), {"record_key": str(i), "text": text, "source_id": "s", "meta": {}})()
            for i, text in enumerate(texts)
        ]
    )
    trainer = ClassifierTrainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
    outcome = trainer.train(
        ClassifierConfig(epochs=30),
        encoder_path,
        tokenizer,
        labelled,
        [item["label"] for item in labelled],
    )
    assert outcome.status == "completed"
    assert outcome.accuracy > 0.5
    assert set(outcome.class_report) == set(outcome.labels)
    assert all("f1" in scores for scores in outcome.class_report.values())
