"""Release keeps a lineage sample, and the planted detectors score a perfect recall."""

from __future__ import annotations

import json

from internal_ai.cli import main
from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.samples import generate_corpus
from internal_ai.data.transforms import detector_recall
from internal_ai.workflows import build_corpus


def test_planted_detectors_score_perfect_recall() -> None:
    report = detector_recall()
    assert report["per_detector"] == {"email": 1.0, "api_token": 1.0, "private_key": 1.0}


def test_release_records_lineage_and_trace_fails_without_a_commit(platform) -> None:
    path = platform.settings.root / "data" / "lineage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_corpus(path, records=6, duplicate_ratio=0, pii_ratio=0, seed=5)
    dataset_id = build_corpus(platform, [path], name="lineage").dataset_id
    released = DataPipeline(platform.store, platform.audit, platform.settings.root).release(
        dataset_id, "ml_lead", gate_approved=True
    )
    assert released.stats["lineage_sample"]["passed"] is True
    assert released.stats["code_commit"] not in ("", "unknown")
    assert released.stats["detector_recall"]["per_detector"]["email"] == 1.0

    lineage = dict(released.lineage)
    lineage["code_commit"] = "unknown"
    platform.store.conn.execute(
        "UPDATE datasets SET lineage = ? WHERE id = ?",
        (json.dumps(lineage), dataset_id),
    )
    assert main(["--root", str(platform.settings.root), "dataset", "trace", dataset_id]) == 1
