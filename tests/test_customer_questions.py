"""A domain expert files the exam. Training waits until that exam is frozen."""

from __future__ import annotations

import pytest
from conftest import enroll

from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.services import build


def _corpus(platform) -> str:
    dataset_id = "ds_q"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="notes",
            version=1,
            content_hash="abc",
            record_count=2,
            token_count=20,
            created_at=now_iso(),
            released=True,
            stats={"splits": {"counts": {"test": 1, "train": 1}}},
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": "notes/vpn.md",
                "payload": "reset the vpn\nfrom the profile",
                "split": "test",
            },
            {"record_key": "notes/train.md", "payload": "this is training text", "split": "train"},
        ],
    )
    return dataset_id


def test_a_domain_expert_files_a_held_out_line_and_training_waits(platform) -> None:
    expert = enroll(platform, "Dex Expert", ["domain_expert"])
    lead = enroll(platform, "Mo Lead", ["ml_lead"])
    dataset_id = _corpus(platform)
    with pytest.raises(PermissionError, match="domain expert"):
        build.file_customer_questions(
            platform, lead, dataset_id, "where is the vpn", "notes/vpn.md", 1
        )
    with pytest.raises(ValueError, match="held-out"):
        build.file_customer_questions(
            platform, expert, dataset_id, "where is the vpn", "notes/train.md", 1
        )
    eval_set = build.file_customer_questions(
        platform, expert, dataset_id, "where is the vpn reset", "notes/vpn.md", 2
    )
    assert eval_set.items[-1]["expected_line"] == 2
    assert eval_set.frozen is False
    with pytest.raises(PermissionError, match="frozen eval"):
        build.require_customer_eval(platform, dataset_id)
    platform.store.freeze_eval_set(eval_set.id, lead.id)
    build.require_customer_eval(platform, dataset_id)
