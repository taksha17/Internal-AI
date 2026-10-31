"""A supervision set cannot be model output, and a leaked secret fails the safety probe."""

from __future__ import annotations

import hashlib

import pytest

from internal_ai.data.supervision import (
    PLANTED_SECRET,
    ingest_supervision,
    safety_probe,
    train_supervisor,
)
from internal_ai.data.tokenizer import train_tokenizer


def test_authored_by_model_is_rejected(platform) -> None:
    with pytest.raises(ValueError, match="authored_by=model"):
        ingest_supervision(
            platform,
            [
                {
                    "instruction": "reset the vpn",
                    "response": "open the profile",
                    "source_id": "src_doc",
                    "authored_by": "model",
                }
            ],
        )


def test_a_supervised_run_writes_a_report(platform, tmp_path) -> None:
    dataset = ingest_supervision(
        platform,
        [
            {
                "instruction": "how do I reset the vpn",
                "response": "open the profile and sign in again",
                "source_id": "src_doc",
                "authored_by": "internal_document",
            },
            {
                "instruction": "when does billing export",
                "response": "the export runs every friday",
                "source_id": "src_doc",
                "authored_by": "ppl_dana",
            },
        ],
    )
    tokenizer = train_tokenizer(
        ["how do I reset the vpn\nopen the profile and sign in again"], vocab_size=80
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    result = train_supervisor(platform, dataset.id, path)
    assert result["metrics_path"]
    report = platform.store.list_eval_reports(result["run_id"])[-1]
    assert report.checks["safety"]["passed"] is True


def test_lineage_names_each_source_in_the_shape_readers_expect(platform) -> None:
    dataset = ingest_supervision(
        platform,
        [
            {
                "instruction": "how do I reset the vpn",
                "response": "open the profile and sign in again",
                "source_id": "src_doc",
                "authored_by": "internal_document",
            }
        ],
    )

    source = dataset.lineage["sources"][0]
    assert source["source_id"] == "src_doc"
    assert source["content_hash"]
    assert dataset.lineage["config_hash"]


def test_a_completion_that_emits_the_planted_secret_fails(platform) -> None:
    probe = safety_probe(lambda _prompt: f"the value is {PLANTED_SECRET}")
    assert probe["safety"]["passed"] is False
