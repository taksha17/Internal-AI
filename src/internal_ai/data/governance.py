"""Pre-release PII spot check and source retention (PRD section 11).

Two obligations live here:

- Before a dataset is released, a sample of the scrubbed records is checked for
  surviving PII and secrets. A critical finding blocks the release.
- When a source is withdrawn, it is tombstoned so future dataset versions
  exclude it, while released versions keep their history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.data.records import Record
from internal_ai.data.transforms import PII_PATTERNS
from internal_ai.domain import AuditOutcome
from internal_ai.ids import new_id, now_iso
from internal_ai.store import Store

CRITICAL_DETECTORS = (
    "email",
    "phone",
    "ssn",
    "credit_card",
    "aws_access_key",
    "private_key",
    "api_token",
    "password_assignment",
    "connection_string",
    "jwt",
)


@dataclass(slots=True)
class PiiAuditResult:
    sampled: int
    findings: dict[str, int] = field(default_factory=dict)
    critical_findings: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampled": self.sampled,
            "findings": self.findings,
            "critical_findings": self.critical_findings,
            "examples": self.examples[:5],
            "passed": self.passed,
        }


def pii_spot_check(
    records: list[Record],
    sample_size: int = 50,
    seed: str = "pii-audit",
    detectors: list[str] | None = None,
) -> PiiAuditResult:
    """Sample released records and look for surviving PII."""
    selected = list(records)
    if sample_size < len(selected):
        generator = random.Random(seed)
        selected = generator.sample(selected, sample_size)
    findings: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    critical = 0
    for record in selected:
        for kind in detectors or list(PII_PATTERNS):
            match = PII_PATTERNS[kind].search(record.text)
            if not match:
                continue
            findings[kind] = findings.get(kind, 0) + 1
            if kind in CRITICAL_DETECTORS:
                critical += 1
            if len(examples) < 10:
                examples.append(
                    {"record_key": record.record_key, "detector": kind, "match": match.group(0)}
                )
    return PiiAuditResult(
        sampled=len(selected),
        findings=findings,
        critical_findings=critical,
        examples=examples,
        passed=critical == 0,
    )


class RetentionManager:
    """Withdraws sources so future dataset versions exclude them."""

    def __init__(self, store: Store, audit: AuditLog) -> None:
        self.store = store
        self.audit = audit

    def tombstoned_sources(self) -> set[str]:
        rows = self.store.conn.execute("SELECT source_id FROM source_tombstones").fetchall()
        return {row["source_id"] for row in rows}

    def withdraw_source(self, source_id: str, reason: str, requested_by: str) -> dict[str, Any]:
        if source_id in self.tombstoned_sources():
            return {"source_id": source_id, "already_withdrawn": True}
        self.store.conn.execute(
            "INSERT INTO source_tombstones (source_id, reason, requested_by, created_at)"
            " VALUES (?,?,?,?)",
            (source_id, reason, requested_by, now_iso()),
        )
        self.audit.record(
            actor=requested_by,
            actor_kind="human",
            action="data.source_withdrawn",
            subject=source_id,
            outcome=AuditOutcome.SUCCESS,
            details={"reason": reason},
        )
        return {"source_id": source_id, "already_withdrawn": False, "reason": reason}

    def remove_records_from_dataset(self, dataset_id: str, record_keys: list[str]) -> int:
        removed = 0
        for key in record_keys:
            cursor = self.store.conn.execute(
                "DELETE FROM dataset_records WHERE dataset_id = ? AND record_key = ?",
                (dataset_id, key),
            )
            removed += cursor.rowcount
        dataset = self.store.get_dataset(dataset_id)
        if dataset is not None:
            remaining = self.store.dataset_records(dataset_id)
            dataset.record_count = len(remaining)
            dataset.token_count = sum(int(row["tokens"]) for row in remaining)
            self.store.conn.execute(
                "UPDATE datasets SET record_count = ?, token_count = ? WHERE id = ?",
                (dataset.record_count, dataset.token_count, dataset_id),
            )
        return removed

    def purge_source_records(self, source_id: str, requested_by: str, reason: str) -> int:
        removed = 0
        for dataset in self.store.list_datasets():
            keys = [
                row["record_key"]
                for row in self.store.dataset_records(dataset.id)
                if row["source_id"] == source_id
            ]
            removed += self.remove_records_from_dataset(dataset.id, keys)
        self.audit.record(
            actor=requested_by,
            actor_kind="human",
            action="data.records_purged",
            subject=source_id,
            outcome=AuditOutcome.SUCCESS,
            details={"records_removed": removed, "reason": reason},
        )
        return removed


def audit_id() -> str:
    return new_id("pii")
