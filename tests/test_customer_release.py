"""Contact details stay on the source version. Release waits for a measured count."""

from __future__ import annotations

import pytest
from conftest import enroll

from internal_ai.data.language import accept_language_sample
from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.samples import generate_corpus
from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.services import build
from internal_ai.workflows import build_corpus


def _notes(platform) -> str:
    dataset_id = "ds_notes"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="notes",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=8,
            created_at=now_iso(),
            stats={"splits": {"counts": {"train": 1}}},
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": "notes/contact.md",
                "payload": "write to ada@example.com about the role",
                "split": "train",
            }
        ],
    )
    return dataset_id


def test_redaction_builds_a_new_version_and_leaves_the_source(platform) -> None:
    owner = enroll(platform, "Dana Owner", ["data_owner"])
    stranger = enroll(platform, "Finn Finance", ["finance"])
    dataset_id = _notes(platform)
    with pytest.raises(PermissionError):
        build.redact_personal_data(platform, stranger, dataset_id)
    created = build.redact_personal_data(platform, owner, dataset_id)
    parent = platform.store.dataset_records(dataset_id)[0]
    child = platform.store.dataset_records(created.id)[0]
    assert "ada@example.com" in parent["payload"]
    assert "ada@example.com" not in child["payload"]
    assert "[REDACTED:email]" in child["payload"]
    assert created.version == 2
    assert created.released is False
    assert platform.store.get_dataset(dataset_id).released is False
    with pytest.raises(ValueError, match="no personal data"):
        build.redact_personal_data(platform, owner, created.id)


def test_the_release_gate_waits_for_a_measured_count(platform) -> None:
    owner = enroll(platform, "Dana Owner", ["data_owner"])
    dataset_id = _notes(platform)
    with pytest.raises(ValueError, match="measure the tokens"):
        build.request_release(platform, owner, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    stats = dict(dataset.stats)
    stats["token_count_kind"] = "measured"
    platform.store.update_dataset_tokens(dataset_id, dataset.token_count, stats)
    created = build.redact_personal_data(platform, owner, dataset_id)
    child = platform.store.get_dataset(created.id)
    stats = dict(child.stats)
    stats["splits"] = {"counts": {"train": 1}}
    platform.store.update_dataset_tokens(created.id, child.token_count, stats)
    with pytest.raises(ValueError, match="measure the tokens"):
        build.request_release(platform, owner, created.id)


def test_release_refuses_a_character_estimate(platform) -> None:
    path = platform.settings.root / "data" / "plain.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_corpus(path, records=6, duplicate_ratio=0, pii_ratio=0, seed=3)
    dataset_id = build_corpus(platform, [path], name="plain").dataset_id
    accept_language_sample(platform.store, dataset_id)
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    with pytest.raises(PermissionError, match="measure the tokens"):
        pipeline.release(dataset_id, "ml_lead", gate_approved=True)
