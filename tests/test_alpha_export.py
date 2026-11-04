"""The export lane is a demo lane, and it refuses anything but public data.

Staging writes raw released text to a box outside the boundary, so the lane is
governed like any other data path: a corpus holding a record that is not
public is refused with the counts named, the run happens only under an
explicit offline override, and a staged export leaves an audit event that says
what left and under which commit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion, EvalSet
from internal_ai.ids import now_iso

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO


def load_stage():
    spec = importlib.util.spec_from_file_location(
        "alpha_y15_kaggle", ROOT / "scripts" / "alpha_y15_kaggle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATASET = "ds_export"
Q1 = "How is the warehouse ledger reconciled at night?"
P1 = "The nightly reconciliation job reads the warehouse ledgerline feed for every open account."
Q2 = "Where does the EMEA batch job write its receipts?"
P2 = "The EMEA batch job writes reconciliation receipts into the export warehouse folder."


def arrange(platform: Platform, sensitivity: str | None = None) -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id=DATASET,
            name="public corpus",
            version=1,
            content_hash="hash-ds_export",
            record_count=2,
            token_count=100,
            created_at=now_iso(),
            released=True,
        )
    )
    rows: list[dict[str, Any]] = [
        {
            "record_key": "runbook.txt:1",
            "payload": P1,
            "source_id": "s",
            "acl": [],
            "split": "train",
        },
        {
            "record_key": "runbook.txt:2",
            "payload": P2,
            "source_id": "s",
            "acl": [],
            "split": "train",
        },
    ]
    if sensitivity is not None:
        for row in rows:
            row["sensitivity"] = sensitivity
    platform.store.add_dataset_records(DATASET, rows)
    platform.store.add_eval_set(
        EvalSet(
            id="evs_export",
            name="held-out",
            version=1,
            frozen=True,
            created_at=now_iso(),
            content_hash="hash-evs",
            items=[
                {"id": "q0", "query": Q1},
                {"id": "q1", "query": Q2},
            ],
        )
    )


def test_a_dataset_with_a_non_public_record_is_refused(tmp_path: Path, platform) -> None:
    script = load_stage()
    arrange(platform, sensitivity="internal")

    with pytest.raises(SystemExit, match="not public"):
        script.stage(platform, DATASET, "evs_export", tmp_path / "out")


def test_the_offline_default_refuses_the_outbound_lane(tmp_path: Path, platform) -> None:
    script = load_stage()
    arrange(platform, sensitivity="public")

    with pytest.raises(SystemExit, match="INTERNAL_AI_OFFLINE=0"):
        script.stage(platform, DATASET, "evs_export", tmp_path / "out")


def test_an_staged_export_records_what_left_and_under_which_commit(
    tmp_path: Path, platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_stage()
    arrange(platform, sensitivity="public")
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setattr(script, "username", lambda: "test-owner")

    staged = script.stage(platform, DATASET, "evs_export", tmp_path / "out")

    assert staged["train_records"] == 2
    events = platform.audit.recent(limit=50)
    export = [event for event in events if event.action == "remote_training_export"]
    assert export, [event.action for event in events]
    details = export[-1].details
    assert details["dataset_id"] == DATASET
    assert details["train_records"] == 2
    assert details["offline_override"] is True
    assert details["commit"] == staged["commit"]
    assert details["eval_set_id"] == "evs_export"
