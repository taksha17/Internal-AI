"""Record model and source ingestion for the data plane."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXCLUDED_SENSITIVITY = ("regulated", "customer_confidential", "pii_source")


@dataclass(slots=True)
class Record:
    id: str
    record_key: str
    text: str
    source_id: str
    sensitivity: str = "internal"
    acl: list[str] = field(default_factory=list)
    tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.text.strip().encode("utf-8"))
        return digest.hexdigest()

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "record_key": self.record_key,
            "payload": self.text,
            "source_id": self.source_id,
            "sensitivity": self.sensitivity,
            "acl": list(self.acl),
            "tokens": self.tokens,
            "meta": dict(self.meta),
        }


@dataclass(slots=True)
class Source:
    source_id: str
    path: Path
    sensitivity: str = "internal"
    acl: list[str] = field(default_factory=list)
    approved: bool = False
    owner: str = ""
    text_field: str | None = None

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        with open(self.path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> list[str]:
    return [
        chunk.strip() for chunk in path.read_text(encoding="utf-8").split("\n\n") if chunk.strip()
    ]


def ingest_source(source: Source) -> list[Record]:
    """Read a source into records. Unapproved sources yield nothing."""
    if not source.approved:
        return []
    suffix = source.path.suffix.lower()
    records: list[Record] = []
    if suffix in (".jsonl", ".ndjson"):
        rows = _read_jsonl(source.path)
        for index, row in enumerate(rows):
            text = str(row.get(source.text_field or "text", "")).strip()
            if not text:
                continue
            records.append(
                Record(
                    id=f"{source.source_id}:{index}",
                    record_key=str(row.get("id") or row.get("record_key") or index),
                    text=text,
                    source_id=source.source_id,
                    sensitivity=str(row.get("sensitivity", source.sensitivity)),
                    acl=list(row.get("acl", source.acl)),
                    meta={"format": "jsonl", "extra_keys": sorted(k for k in row if k != "text")},
                )
            )
    elif suffix == ".csv":
        rows = _read_csv(source.path)
        for index, row in enumerate(rows):
            text = str(row.get(source.text_field or "text", "")).strip()
            if not text:
                continue
            records.append(
                Record(
                    id=f"{source.source_id}:{index}",
                    record_key=str(row.get("id") or index),
                    text=text,
                    source_id=source.source_id,
                    sensitivity=str(row.get("sensitivity", source.sensitivity)),
                    acl=list(row.get("acl", source.acl)),
                    meta={"format": "csv"},
                )
            )
    else:
        chunks = _read_text(source.path)
        for index, chunk in enumerate(chunks):
            records.append(
                Record(
                    id=f"{source.source_id}:{index}",
                    record_key=str(index),
                    text=chunk,
                    source_id=source.source_id,
                    sensitivity=source.sensitivity,
                    acl=list(source.acl),
                    meta={"format": "txt"},
                )
            )
    return records
