"""Failed citations can be listed and filed. They stay out of training until a release."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from conftest import enroll

from internal_ai.cli import main
from internal_ai.domain import AuditEvent, AuditOutcome
from internal_ai.feedback import (
    check_quality_drift,
    export_feedback,
    failed_citations,
    file_feedback,
    latest_gpu_busy,
)
from internal_ai.ids import new_id
from internal_ai.training.trainer import gpu_busy


def _request(platform, when: str, passed: bool) -> None:
    platform.store.append_audit(
        AuditEvent(
            id=new_id("aud"),
            seq=0,
            actor="ppl_ada",
            actor_kind="human",
            action="serving.request",
            subject="/v1/answer",
            outcome=AuditOutcome.SUCCESS,
            details={
                "citation_passed": passed,
                "model": "embedding-v1",
                "hits": 3,
                "latency_ms": 18,
                "question": "where is the razorpay host configured",
            },
            created_at=when,
        )
    )


def test_feedback_list_prints_the_miss_and_not_the_question(platform, capsys) -> None:
    _request(platform, datetime.now(UTC).isoformat(timespec="seconds"), False)
    assert main(["--root", str(platform.settings.root), "feedback", "list"]) == 0
    printed = capsys.readouterr().out
    assert "embedding-v1" in printed
    assert "hits 3" in printed
    assert "18" in printed
    assert "razorpay" not in printed
    assert "question" not in failed_citations(platform)[0]


def test_a_data_owner_files_a_row_that_export_does_not_release(platform) -> None:
    owner = enroll(platform, "Dana Owner", ["data_owner"])
    lead = enroll(platform, "Mo Lead", ["ml_lead"])
    with pytest.raises(PermissionError, match="data_owner"):
        file_feedback(platform, lead, "where is the host", "next.config.ts", 36, "ds_1")
    row = file_feedback(platform, owner, "where is the host", "next.config.ts", 36, "ds_1")
    dataset = export_feedback(platform, owner)
    assert dataset.released is False
    assert row["id"] in dataset.lineage["feedback_ids"]
    assert dataset.stats["training_data"] is False


def test_a_doubled_failure_rate_alerts_and_a_first_week_does_not(platform) -> None:
    now = datetime.now(UTC)
    recent = (now - timedelta(days=1)).isoformat(timespec="seconds")
    previous = (now - timedelta(days=10)).isoformat(timespec="seconds")
    _request(platform, recent, False)
    _request(platform, recent, False)
    assert check_quality_drift(platform) is False
    for _ in range(4):
        _request(platform, previous, True)
    _request(platform, previous, False)
    assert check_quality_drift(platform) is True
    kinds = [alert["kind"] for alert in platform.store.conn.execute("SELECT kind FROM alerts")]
    assert "quality_drift" in kinds


def test_gpu_busy_is_zero_without_cuda_and_the_jobs_page_can_read_it(tmp_path) -> None:
    assert gpu_busy() == 0
    metrics = tmp_path / "runs" / "run_1" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"step": 1, "gpu_busy": 0}\n', encoding="utf-8")
    assert latest_gpu_busy(tmp_path) == 0
