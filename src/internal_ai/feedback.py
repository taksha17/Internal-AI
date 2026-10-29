"""Ask failures a person files by hand. They are not training data until a release."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from internal_ai.alerts import AlertEngine
from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion
from internal_ai.ids import content_hash, new_id, now_iso
from internal_ai.services.actor import Actor

WINDOW = timedelta(days=7)


def failed_citations(platform: Platform) -> list[dict[str, Any]]:
    """Recent answers whose citation check failed. The question is not in this row."""
    events = platform.store.list_audit(action="serving.request", limit=100_000, descending=True)
    rows = []
    for event in events:
        if event.details.get("citation_passed") is not False:
            continue
        rows.append(
            {
                "time": event.created_at,
                "model": event.details.get("model", ""),
                "hits": event.details.get("hits", 0),
                "latency_ms": event.details.get("latency_ms", 0),
            }
        )
    return rows


def file_feedback(
    platform: Platform,
    actor: Actor,
    question: str,
    expected_file: str,
    expected_line: int,
    dataset_id: str,
) -> dict[str, Any]:
    if not actor.holds("data_owner"):
        raise PermissionError("filing feedback requires data_owner")
    row = {
        "id": new_id("fbk"),
        "question": question,
        "expected_file": expected_file,
        "expected_line": int(expected_line),
        "dataset_id": dataset_id,
        "filed_by": actor.id,
        "filed_at": now_iso(),
    }
    platform.store.conn.execute(
        """
        INSERT INTO feedback
            (id, question, expected_file, expected_line, dataset_id, filed_by, filed_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            row["id"],
            question,
            expected_file,
            row["expected_line"],
            dataset_id,
            actor.id,
            row["filed_at"],
        ),
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind="human",
        action="feedback.filed",
        subject=row["id"],
        details={"dataset_id": dataset_id, "expected_file": expected_file},
    )
    return row


def list_feedback(platform: Platform) -> list[dict[str, Any]]:
    rows = platform.store.conn.execute("SELECT * FROM feedback ORDER BY filed_at").fetchall()
    return [dict(row) for row in rows]


def export_feedback(platform: Platform, actor: Actor) -> DatasetVersion:
    """A new unreleased dataset. Release still needs its gates; nothing here trains."""
    if not actor.holds("data_owner"):
        raise PermissionError("exporting feedback requires data_owner")
    rows = list_feedback(platform)
    dataset_id = new_id("ds")
    records = []
    for row in rows:
        text = f"{row['question']}\nexpected {row['expected_file']}:{row['expected_line']}"
        records.append(
            {
                "record_key": row["id"],
                "payload": text,
                "source_id": row["dataset_id"],
                "meta": {"kind": "feedback", "feedback_id": row["id"]},
            }
        )
    dataset = DatasetVersion(
        id=dataset_id,
        name="ask-feedback",
        version=1,
        content_hash=content_hash({"rows": [item["record_key"] for item in records]}),
        record_count=len(records),
        token_count=sum(max(1, len(item["payload"]) // 4) for item in records),
        created_at=now_iso(),
        lineage={"kind": "feedback", "feedback_ids": [row["id"] for row in rows]},
        stats={"kind": "feedback", "training_data": False},
        released=False,
    )
    platform.store.add_dataset(dataset)
    platform.store.add_dataset_records(dataset_id, records)
    return platform.store.get_dataset(dataset_id) or dataset


def check_quality_drift(platform: Platform) -> bool:
    """Alert when this week's failed-citation rate is more than twice last week's.

    A first week, with nothing in the previous window, does not alert.
    """
    now = datetime.now(UTC)
    events = platform.store.list_audit(action="serving.request", limit=100_000, descending=False)
    recent_total, recent_failed = _window(events, now - WINDOW, now)
    previous_total, previous_failed = _window(events, now - 2 * WINDOW, now - WINDOW)
    if previous_total == 0:
        return False
    recent_rate = recent_failed / recent_total if recent_total else 0.0
    previous_rate = previous_failed / previous_total
    if recent_rate <= previous_rate * 2:
        return False
    AlertEngine(platform.store, platform.audit, platform.budget).raise_alert(
        "quality_drift",
        "high",
        "serving.request",
        "failed citations this week are more than twice the previous week",
        {"recent_rate": recent_rate, "previous_rate": previous_rate},
    )
    return True


def _window(events: list[Any], start: datetime, end: datetime) -> tuple[int, int]:
    total = 0
    failed = 0
    for event in events:
        if "citation_passed" not in event.details:
            continue
        stamp = datetime.fromisoformat(event.created_at)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        if not (start <= stamp < end):
            continue
        total += 1
        if event.details.get("citation_passed") is False:
            failed += 1
    return total, failed


def latest_gpu_busy(artifacts: Path) -> int | None:
    files = list(artifacts.glob("**/metrics.jsonl"))
    if not files:
        return None
    newest = max(files, key=lambda item: item.stat().st_mtime)
    lines = [line for line in newest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return int(json.loads(lines[-1]).get("gpu_busy", 0))
