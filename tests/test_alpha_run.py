"""An adapter run is a governed job: capability, licence, record, audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.services import alpha as alpha_service
from internal_ai.services.alpha import train_alpha
from internal_ai.services.jobs import JobRunner
from internal_ai.services.permissions import PermissionDenied
from internal_ai.training.alpha.pin import PRIMARY, BaseRefused
from internal_ai.training.alpha.promote import register_alpha
from internal_ai.training.provenance import OPEN_WEIGHT_ALLOWED

ADAPTER_BYTES = b"not a real adapter, but a real file"


def enable_open_weights(platform) -> None:
    """An install that has decided it may adapt a base (see tests/test_provenance.py)."""
    decision = platform.policy.get("P6")
    decision.current_value = OPEN_WEIGHT_ALLOWED
    decision.status = "decided"
    platform.store.upsert_policy(decision)


def released_dataset(platform, *, released: bool = True) -> str:
    dataset_id = "ds_alpha"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="company",
            version=1,
            content_hash="hash-alpha",
            record_count=1,
            token_count=1,
            created_at=now_iso(),
            released=released,
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": "train-1",
                "payload": "the nightly reconciliation failed on the ledger feed",
                "source_id": "s",
                "acl": [],
            }
        ],
    )
    return dataset_id


def stub_training(base, root, texts, out, layers=2, sequences=4, seed=0) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ADAPTER_BYTES)
    return {
        "base_repo_id": base.repo_id,
        "adapter": str(out),
        "sequences": len(texts),
        "layers": layers,
        "loss_first": 1.0,
        "loss_last": 0.5,
    }


@pytest.fixture()
def governed(platform, enroll_person, monkeypatch):
    actor = enroll_person(platform, "Ada Lovelace", ["ml_lead"])
    enable_open_weights(platform)
    monkeypatch.setattr(alpha_service, "available_base", lambda root: PRIMARY)
    monkeypatch.setattr(alpha_service, "train_prefix", stub_training)
    return platform, actor


def test_an_adaptation_run_leaves_a_record(governed) -> None:
    platform, actor = governed
    dataset_id = released_dataset(platform)
    job = train_alpha(platform, actor, JobRunner(platform, synchronous=True), dataset_id)
    assert job["state"] == "succeeded", job.get("error")
    result = job["result"]
    record = json.loads(Path(result["reproducibility"]).read_text(encoding="utf-8"))
    assert record["track"] == "alpha"
    assert record["base"]["repo_id"] == PRIMARY.repo_id
    assert record["base"]["revision"] == PRIMARY.revision
    assert record["base"]["license_class"] == "permissive"
    assert record["dataset"]["content_hash"] == "hash-alpha"
    assert record["seed"] == 0
    assert record["adapter_sha256"]
    assert record["recipe"]["rank"] == 8
    runs = platform.store.list_runs()
    assert [run.task_id for run in runs] == [f"alpha:{dataset_id}"]
    assert runs[0].status == "completed"
    assert runs[0].metadata["run_id"] == result["run_id"]
    events = platform.store.conn.execute(
        "SELECT action, subject FROM audit_events WHERE action = 'alpha.train_finished'"
    ).fetchall()
    assert [row["subject"] for row in events] == [result["run_id"]]


def test_an_unreleased_dataset_cannot_be_adapted(governed) -> None:
    platform, actor = governed
    dataset_id = released_dataset(platform, released=False)
    with pytest.raises(PermissionError, match="released dataset"):
        train_alpha(platform, actor, JobRunner(platform, synchronous=True), dataset_id)
    assert platform.store.list_runs() == []


def test_adaptation_is_denied_to_a_role_that_may_not_train(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Grace Hopper", ["data_owner"])
    dataset_id = released_dataset(platform)
    with pytest.raises(PermissionDenied):
        train_alpha(platform, actor, JobRunner(platform, synchronous=True), dataset_id)


def test_no_allowed_base_runs_nothing(platform, enroll_person, monkeypatch) -> None:
    actor = enroll_person(platform, "Katherine Johnson", ["ml_lead"])
    enable_open_weights(platform)
    dataset_id = released_dataset(platform)

    def refuse(root):
        raise BaseRefused("no allowed pinned base is on disk; this install allows permissive")

    monkeypatch.setattr(alpha_service, "available_base", refuse)
    with pytest.raises(BaseRefused):
        train_alpha(platform, actor, JobRunner(platform, synchronous=True), dataset_id)
    assert platform.store.list_runs() == []


def test_the_card_carries_the_record_beside_the_adapter(governed, tmp_path) -> None:
    platform, actor = governed
    dataset_id = released_dataset(platform)
    job = train_alpha(platform, actor, JobRunner(platform, synchronous=True), dataset_id)
    adapter = Path(job["result"]["adapter"])
    model = register_alpha(platform, PRIMARY, adapter, dataset_id, actor)
    assert model.card["seed"] == 0
    assert model.card["commit"]
    assert model.card["run_id"] == job["result"]["run_id"]
    assert model.card["reproducibility"]["base"]["revision"] == PRIMARY.revision
