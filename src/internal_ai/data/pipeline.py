"""Data pipeline: ingest to versioned, scrubbed, lineage-tracked corpus (FR-5 to FR-9)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.data.records import EXCLUDED_SENSITIVITY, Record, Source, ingest_source
from internal_ai.data.transforms import (
    deduplicate,
    filter_quality,
    normalize,
    scrub_pii,
)
from internal_ai.domain import AuditOutcome, DatasetVersion
from internal_ai.ids import content_hash, new_id, now_iso
from internal_ai.store import Store


@dataclass(slots=True)
class PipelineConfig:
    name: str
    version: int = 1
    min_chars: int = 40
    max_chars: int = 200_000
    dedup_threshold: int = 12
    dedup_jaccard: float = 0.8
    allowed_languages: tuple[str, ...] = ("en",)
    excluded_sensitivity: tuple[str, ...] = EXCLUDED_SENSITIVITY
    min_alpha_ratio: float = 0.55
    max_boilerplate_ratio: float = 0.25
    detectors: tuple[str, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "dedup_threshold": self.dedup_threshold,
            "dedup_jaccard": self.dedup_jaccard,
            "allowed_languages": list(self.allowed_languages),
            "excluded_sensitivity": list(self.excluded_sensitivity),
            "min_alpha_ratio": self.min_alpha_ratio,
            "max_boilerplate_ratio": self.max_boilerplate_ratio,
            "detectors": list(self.detectors) if self.detectors else None,
            "extra": dict(self.extra),
        }

    @property
    def hash(self) -> str:
        return content_hash(self.as_dict())


@dataclass(slots=True)
class PipelineResult:
    dataset: DatasetVersion
    records: list[Record]
    stats: dict[str, Any]
    lineage: dict[str, Any]


def _git_commit(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


class DataPipeline:
    def __init__(self, store: Store, audit: AuditLog, root: Path | None = None) -> None:
        self.store = store
        self.audit = audit
        self.root = root or Path.cwd()

    def run(
        self,
        sources: list[Source],
        config: PipelineConfig,
        actor: str = "data_engineer",
    ) -> PipelineResult:
        steps: list[dict[str, Any]] = []
        lineage_sources: list[dict[str, Any]] = []
        records: list[Record] = []
        withdrawn = {
            row["source_id"]
            for row in self.store.conn.execute("SELECT source_id FROM source_tombstones").fetchall()
        }
        for source in sources:
            if source.source_id in withdrawn:
                self.audit.record(
                    actor=actor,
                    action="data.source_withdrawn_skipped",
                    subject=source.source_id,
                    outcome=AuditOutcome.DENIED,
                    details={"reason": "source was withdrawn and is excluded from new versions"},
                )
                continue
            if not source.approved:
                self.audit.record(
                    actor=actor,
                    action="data.source_rejected",
                    subject=source.source_id,
                    outcome=AuditOutcome.DENIED,
                    details={"reason": "source is not approved for use"},
                )
                continue
            ingested = ingest_source(source)
            source_hash = source.content_hash()
            lineage_sources.append(
                {
                    "source_id": source.source_id,
                    "path": str(source.path),
                    "content_hash": source_hash,
                    "owner": source.owner,
                    "sensitivity": source.sensitivity,
                    "records_ingested": len(ingested),
                    "approved": True,
                }
            )
            records.extend(ingested)
        steps.append({"step": "ingest", "input": 0, "output": len(records)})

        excluded = set(config.excluded_sensitivity)
        before = len(records)
        records = [record for record in records if record.sensitivity not in excluded]
        steps.append(
            {
                "step": "sensitivity_exclusion",
                "input": before,
                "output": len(records),
                "excluded_classes": sorted(excluded),
            }
        )

        records, stats = normalize(records)
        steps.append({"step": "normalize", **stats})

        dedup = deduplicate(
            records,
            threshold=config.dedup_threshold,
            jaccard_threshold=config.dedup_jaccard,
        )
        records = dedup[0]
        steps.append({"step": "deduplicate", **dedup[1]})

        records, stats = scrub_pii(
            records, detectors=list(config.detectors) if config.detectors else None
        )
        steps.append({"step": "pii_scrub", **stats})

        records, stats = filter_quality(
            records,
            min_chars=config.min_chars,
            max_chars=config.max_chars,
            allowed_languages=config.allowed_languages,
            min_alpha_ratio=config.min_alpha_ratio,
            max_boilerplate_ratio=config.max_boilerplate_ratio,
        )
        steps.append({"step": "quality_filter", **stats})

        for record in records:
            record.tokens = max(1, len(record.text) // 4)
        token_count = sum(record.tokens for record in records)
        record_hashes = sorted(record.fingerprint() for record in records)
        dataset_hash = content_hash({"records": record_hashes, "config": config.hash})

        lineage = {
            "sources": lineage_sources,
            "steps": steps,
            "config": config.as_dict(),
            "config_hash": config.hash,
            "code_commit": _git_commit(self.root),
            "created_at": now_iso(),
        }
        dataset = DatasetVersion(
            id=new_id("ds"),
            name=config.name,
            version=config.version,
            content_hash=dataset_hash,
            record_count=len(records),
            token_count=token_count,
            created_at=now_iso(),
            lineage=lineage,
            stats={"steps": steps},
            released=False,
            parent_version=None,
        )
        self.store.add_dataset(dataset)
        self.store.add_dataset_records(dataset.id, [record.to_row() for record in records])
        self.audit.record(
            actor=actor,
            action="dataset.version_created",
            subject=dataset.id,
            outcome=AuditOutcome.SUCCESS,
            details={
                "name": dataset.name,
                "version": dataset.version,
                "content_hash": dataset.content_hash,
                "records": dataset.record_count,
                "tokens": dataset.token_count,
                "lineage_coverage": "100%",
            },
        )
        return PipelineResult(
            dataset=dataset, records=records, stats={"steps": steps}, lineage=lineage
        )

    def run_pii_audit(self, dataset_id: str, sample_size: int = 50) -> dict[str, Any]:
        """Sampled check that no PII survived scrubbing (NFR: Privacy)."""
        from internal_ai.data.governance import pii_spot_check

        rows = self.store.dataset_records(dataset_id)
        records = [
            Record(
                id=row["id"],
                record_key=row["record_key"],
                text=row["payload"],
                source_id=row["source_id"],
                sensitivity=row["sensitivity"],
                acl=row["acl"],
                tokens=row["tokens"],
                meta=row["meta"],
            )
            for row in rows
        ]
        result = pii_spot_check(records, sample_size=sample_size)
        dataset = self.store.get_dataset(dataset_id)
        if dataset is not None:
            stats = dict(dataset.stats)
            stats["pii_audit"] = result.to_dict()
            dataset.stats = stats
            self.store.conn.execute(
                "UPDATE datasets SET stats = ? WHERE id = ?",
                (json.dumps(stats, sort_keys=True, default=str), dataset_id),
            )
        self.audit.record(
            actor="data_engineer",
            action="dataset.pii_audit",
            subject=dataset_id,
            outcome=AuditOutcome.SUCCESS if result.passed else AuditOutcome.FAILURE,
            details=result.to_dict(),
        )
        return result.to_dict()

    def release(
        self, dataset_id: str, actor: str, gate_approved: bool, sample_size: int = 50
    ) -> DatasetVersion:
        if not gate_approved:
            raise PermissionError(
                "dataset release requires an approved dataset_release gate (FR-15, PRD section 14)"
            )
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        audit_result = self.run_pii_audit(dataset_id, sample_size=sample_size)
        if not audit_result["passed"]:
            raise PermissionError(
                f"PII audit found {audit_result['critical_findings']} critical findings; "
                "the dataset cannot be released"
            )
        self.store.release_dataset(dataset_id)
        released = self.store.get_dataset(dataset_id)
        self.audit.record(
            actor=actor,
            actor_kind="human",
            action="dataset.released",
            subject=dataset_id,
            details={
                "content_hash": dataset.content_hash,
                "records": dataset.record_count,
                "pii_audit": audit_result,
            },
        )
        return released or dataset

    def lineage(self, dataset_id: str) -> dict[str, Any]:
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return dataset.lineage
