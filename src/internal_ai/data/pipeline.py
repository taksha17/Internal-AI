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
    duplicate_report,
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
    dedup_residual_max: float = 0.01
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
            "dedup_residual_max": self.dedup_residual_max,
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
    """The commit that built this dataset.

    The data root is usually outside the checkout, so the package directory is
    tried too: a build run against a temporary data directory still records the
    code that produced it.
    """
    for candidate in (root, Path(__file__).resolve().parents[3]):
        try:
            proc = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return "unknown"


def sample_lineage(dataset: DatasetVersion, rows: list[Any], limit: int = 20) -> dict[str, Any]:
    """Trace a sample of records back to their source, config and commit.

    Lineage at dataset level can be present while a record points at a source the
    lineage never saw; only reading the rows shows that. The sample is sorted by
    record key so two runs of the check inspect the same rows.
    """
    sources = {entry.get("source_id"): entry for entry in dataset.lineage.get("sources", [])}
    ordered = sorted(rows, key=lambda row: row["record_key"])[:limit]
    failures: list[dict[str, str]] = []
    for row in ordered:
        source = sources.get(row["source_id"])
        if source is None:
            failures.append({"record": row["record_key"], "problem": "source missing from lineage"})
        elif not source.get("content_hash"):
            failures.append({"record": row["record_key"], "problem": "source has no content hash"})
    checked = len(ordered)
    traced = checked - len(failures)
    config_hash = str(dataset.lineage.get("config_hash", ""))
    commit = str(dataset.lineage.get("code_commit", "unknown"))
    return {
        "sampled": checked,
        "traced": traced,
        "coverage": round(traced / checked, 4) if checked else 0.0,
        "failures": failures[:5],
        "config_hash": config_hash,
        "code_commit": commit,
        "passed": bool(checked) and not failures and bool(config_hash) and commit != "unknown",
    }


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
        label_field: str = "",
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
        # No tokenizer exists yet at build time. The count is a planning estimate
        # until ``measure_token_count`` replaces it after the tokenizer is trained.
        token_count_kind = "estimate"
        record_hashes = sorted(record.fingerprint() for record in records)
        # Labels define the artifact as much as the text does, but only when the
        # data brought them: a dataset built without any keeps the hash it had.
        given_labels = sorted(
            str(record.meta["label"]) for record in records if record.meta.get("label")
        )
        hash_input: dict[str, Any] = {"records": record_hashes, "config": config.hash}
        if given_labels:
            hash_input["labels"] = given_labels
        dataset_hash = content_hash(hash_input)

        lineage = {
            "sources": lineage_sources,
            "steps": steps,
            "config": config.as_dict(),
            "config_hash": config.hash,
            "code_commit": _git_commit(self.root),
            "created_at": now_iso(),
        }
        if given_labels:
            lineage["labels"] = {
                "column": label_field,
                "count": len(given_labels),
                "distribution": {
                    label: given_labels.count(label) for label in sorted(set(given_labels))
                },
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
            stats={"steps": steps, "token_count_kind": token_count_kind},
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
                "token_count_kind": token_count_kind,
                "lineage_coverage": "100%",
            },
        )
        return PipelineResult(
            dataset=dataset, records=records, stats={"steps": steps}, lineage=lineage
        )

    def measure_token_count(
        self, dataset_id: str, tokenizer: Any, actor: str = "data_engineer"
    ) -> dict[str, Any]:
        """Replace the character-based estimate with the tokenizer's own count.

        The estimate at build time is ``len(text) // 4``, which is close enough to
        plan with and wrong enough to plan a run on. Once the tokenizer is trained
        the exact count can be measured, and the dataset says which one it holds.
        """
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        rows = self.store.dataset_records(dataset_id)
        counted = sum(len(tokenizer.encode(row["payload"])) for row in rows)
        stats = dict(dataset.stats)
        stats["token_count_kind"] = "measured"
        stats["token_count_estimate"] = dataset.token_count
        stats["tokenizer_hash"] = content_hash(tokenizer.vocab)
        self.store.update_dataset_tokens(dataset_id, counted, stats)
        result = {
            "dataset_id": dataset_id,
            "tokens": counted,
            "estimate": dataset.token_count,
            "tokenizer_hash": stats["tokenizer_hash"],
        }
        self.audit.record(
            actor=actor,
            action="dataset.tokens_measured",
            subject=dataset_id,
            details=result,
        )
        return result

    def trace_sample(
        self, dataset_id: str, limit: int = 20, actor: str = "data_engineer"
    ) -> dict[str, Any]:
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        report = sample_lineage(dataset, self.store.dataset_records(dataset_id), limit=limit)
        self.audit.record(
            actor=actor,
            action="dataset.lineage_sampled",
            subject=dataset_id,
            outcome=AuditOutcome.SUCCESS if report["passed"] else AuditOutcome.FAILURE,
            details=report,
        )
        return report

    def run_pii_audit(self, dataset_id: str, sample_size: int | None = 50) -> dict[str, Any]:
        """Sampled check that no PII survived scrubbing (NFR: Privacy).

        ``sample_size=None`` reads every record. Release uses that. A caller-chosen
        sample stays available for the informational audit command.
        """
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
        result = pii_spot_check(
            records, sample_size=len(records) if sample_size is None else sample_size
        )
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

    def _dataset_records(self, dataset_id: str) -> list[Record]:
        return [
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
            for row in self.store.dataset_records(dataset_id)
        ]

    def _store_stats(self, dataset_id: str, key: str, value: dict[str, Any]) -> None:
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            return
        stats = dict(dataset.stats)
        stats[key] = value
        for name in (
            "exact_duplicate_groups",
            "near_duplicate_pairs",
            "residual_rate",
            "code_commit",
        ):
            if name in value:
                stats[name] = value[name]
        self.store.conn.execute(
            "UPDATE datasets SET stats = ? WHERE id = ?",
            (json.dumps(stats, sort_keys=True, default=str), dataset_id),
        )

    def _duplicate_gate(self, dataset: DatasetVersion) -> dict[str, Any]:
        config = dataset.lineage.get("config", {})
        report = duplicate_report(
            self._dataset_records(dataset.id),
            threshold=int(config.get("dedup_threshold", 12)),
            jaccard_threshold=float(config.get("dedup_jaccard", 0.8)),
            residual_max=float(config.get("dedup_residual_max", 0.01)),
        )
        self._store_stats(dataset.id, "duplicate_gate", report)
        return report

    def drop_duplicate_copies(self, dataset_id: str, actor: str) -> DatasetVersion:
        """A new version with exact and near copies removed. The source version stays."""
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        records = self._dataset_records(dataset_id)
        config = dataset.lineage.get("config", {})
        kept, stats = deduplicate(
            records,
            threshold=int(config.get("dedup_threshold", 12)),
            jaccard_threshold=float(config.get("dedup_jaccard", 0.8)),
        )
        if len(kept) == len(records):
            raise ValueError("no duplicate copies to drop")
        fresh = [
            Record(
                id=new_id("rec"),
                record_key=record.record_key,
                text=record.text,
                source_id=record.source_id,
                sensitivity=record.sensitivity,
                acl=list(record.acl),
                tokens=record.tokens,
                meta=dict(record.meta),
            )
            for record in kept
        ]
        lineage = dict(dataset.lineage)
        steps = list(lineage.get("steps", []))
        steps.append({"step": "deduplicate", **stats})
        lineage["steps"] = steps
        lineage["parent"] = dataset.id
        created = DatasetVersion(
            id=new_id("ds"),
            name=dataset.name,
            version=dataset.version + 1,
            content_hash=content_hash(
                {"parent": dataset.id, "records": sorted(item.fingerprint() for item in fresh)}
            ),
            record_count=len(fresh),
            token_count=sum(item.tokens for item in fresh),
            created_at=now_iso(),
            lineage=lineage,
            stats={
                "steps": steps,
                "token_count_kind": dataset.stats.get("token_count_kind", "estimate"),
                "exact_duplicate_groups": stats["exact_duplicates_removed"],
                "near_duplicate_pairs": stats["near_duplicates_removed"],
                "residual_rate": stats["duplicate_rate"],
            },
            released=False,
            parent_version=dataset.id,
        )
        before = self.store.get_dataset(dataset_id)
        self.store.add_dataset(created)
        self.store.add_dataset_records(created.id, [item.to_row() for item in fresh])
        after = self.store.get_dataset(dataset_id)
        if before is None or after is None or before.released != after.released:
            raise RuntimeError("dropping copies changed the source version")
        if after.record_count != before.record_count:
            raise RuntimeError("dropping copies changed the source version")
        self.audit.record(
            actor=actor,
            actor_kind="human",
            action="dataset.duplicates_dropped",
            subject=created.id,
            details={"parent": dataset.id, "records": created.record_count, **stats},
        )
        return created

    def _exclusion_audit(self, dataset_id: str, actor: str) -> dict[str, Any]:
        from internal_ai.data.governance import exclusion_audit

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
        legal = {
            row["id"]: row["legal_status"]
            for row in self.store.conn.execute("SELECT id, legal_status FROM data_sources")
        }
        result = exclusion_audit(records, legal)
        dataset = self.store.get_dataset(dataset_id)
        if dataset is not None:
            stats = dict(dataset.stats)
            stats["exclusion_audit"] = result
            self.store.conn.execute(
                "UPDATE datasets SET stats = ? WHERE id = ?",
                (json.dumps(stats, sort_keys=True, default=str), dataset_id),
            )
        classes = sorted(
            {hit.get("sensitivity") or hit.get("legal_status", "") for hit in result["hits"]}
        )
        self.audit.record(
            actor=actor,
            actor_kind="human",
            action="dataset.exclusion_audit",
            subject=dataset_id,
            outcome=AuditOutcome.SUCCESS if result["passed"] else AuditOutcome.DENIED,
            details={"classes": [name for name in classes if name], **result},
        )
        return result

    def release(self, dataset_id: str, actor: str, gate_approved: bool) -> DatasetVersion:
        if not gate_approved:
            raise PermissionError(
                "dataset release requires an approved dataset_release gate (FR-15, PRD section 14)"
            )
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        duplicates = self._duplicate_gate(dataset)
        if duplicates["exact_duplicate_groups"]:
            shown = "; ".join(f"{left} = {right}" for left, right in duplicates["pairs"])
            raise PermissionError(
                f"{duplicates['exact_duplicate_groups']} exact duplicate groups; {shown}"
            )
        if duplicates["residual_rate"] > duplicates["residual_max"]:
            raise PermissionError(
                "near-duplicate residual "
                f"{duplicates['residual_rate']} is above {duplicates['residual_max']}"
            )
        lineage = self.trace_sample(dataset_id, limit=20, actor=actor)
        self._store_stats(dataset_id, "lineage_sample", lineage)
        if not lineage["passed"]:
            raise PermissionError(
                "lineage sample failed"
                + (f": {lineage['failures'][0]['problem']}" if lineage["failures"] else "")
                + ("" if lineage["code_commit"] != "unknown" else ": code commit is unknown")
            )
        from internal_ai.data.transforms import detector_recall

        self._store_stats(dataset_id, "detector_recall", detector_recall())
        exclusion = self._exclusion_audit(dataset_id, actor)
        if not exclusion["passed"]:
            named = sorted(
                {hit.get("sensitivity") or hit.get("legal_status", "") for hit in exclusion["hits"]}
            )
            raise PermissionError(
                "release sample contains excluded data: "
                + ", ".join(name for name in named if name)
            )
        audit_result = self.run_pii_audit(dataset_id, sample_size=None)
        if not audit_result["passed"]:
            raise PermissionError(
                f"release audit found {audit_result['critical_findings']} critical finding(s) "
                "(personal data or a secret); the dataset cannot be released"
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
