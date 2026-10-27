"""Record model and source ingestion for the data plane."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from internal_ai.data.extract import DOCUMENT_SUFFIXES, document_text

# Row-oriented files: one record per row, so a row can carry its own answer.
ROW_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".ndjson"})

# Which column holds the writing, in order of preference. A real export rarely
# has one called "text", so what it did pick is reported back to the person.
TEXT_HINTS = (
    "text",
    "body",
    "content",
    "message",
    "description",
    "detail",
    "summary",
    "subject",
    "note",
    "question",
    "answer",
    "title",
)

SAMPLE_ROWS = 50

EXCLUDED_SENSITIVITY = (
    "regulated",
    "customer_confidential",
    "confidential",
    "personal",
    "pii_source",
    "pretrained_weights",
    "system",
    "recovered",
)


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
    label_field: str | None = None

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


def _paragraphs(text: str) -> list[str]:
    """A document or a note as the chunks a person would call paragraphs."""
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def _read_text(path: Path) -> list[str]:
    return _paragraphs(path.read_text(encoding="utf-8"))


def _sample_rows(suffix: str, content: str) -> list[dict[str, Any]]:
    """The first rows of a table, enough to see its columns without reading it all."""
    if suffix in (".jsonl", ".ndjson"):
        rows: list[dict[str, Any]] = []
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                rows.append(item)
            if len(rows) >= SAMPLE_ROWS:
                break
        return rows
    if suffix in (".csv", ".tsv"):
        reader = csv.DictReader(io.StringIO(content), delimiter="\t" if suffix == ".tsv" else ",")
        return [row for _, row in zip(range(SAMPLE_ROWS), reader, strict=False)]
    return []


def _longest_column(rows: list[dict[str, Any]], exclude: str) -> str:
    """The column carrying the most writing, when no name says which one it is."""
    names = [name for name in rows[0] if name and name != exclude]
    if not names:
        return ""
    totals = {name: sum(len(str(row.get(name) or "")) for row in rows) for name in names}
    best = max(sorted(names), key=lambda name: totals[name])
    return best if totals[best] else ""


def choose_columns(suffix: str, content: str) -> tuple[str, str]:
    """Which column is the text and which one holds the answer.

    Guessing wrong either drops every row or teaches the model the wrong field,
    so the choice is returned for the caller to show rather than kept quiet. The
    answer column is only claimed when its name says it is one.
    """
    from internal_ai.data.suggest import LABEL_HINTS

    rows = _sample_rows(suffix, content)
    if not rows:
        return "", ""
    actual = {str(name).lower(): str(name) for name in rows[0] if name}
    label = next((actual[hint] for hint in sorted(LABEL_HINTS) if hint in actual), "")
    text = next(
        (actual[hint] for hint in TEXT_HINTS if hint in actual and actual[hint] != label), ""
    )
    return (text or _longest_column(rows, exclude=label), label)


def _bounded_acl(row: dict[str, Any], source: Source) -> list[str]:
    """A row may narrow the source ACL. It may not grant a group the source did not."""
    source_acl = [str(item) for item in source.acl]
    raw = row.get("acl", source_acl)
    if not isinstance(raw, list):
        return source_acl
    requested = [str(item) for item in raw if str(item).strip()]
    if not source_acl:
        return requested
    allowed = [item for item in requested if item in set(source_acl)]
    return allowed or source_acl


def _label_meta(row: dict[str, Any], source: Source) -> dict[str, str]:
    """The answer this row carries, when the source names the column holding it."""
    if not source.label_field:
        return {}
    value = str(row.get(source.label_field) or "").strip()
    return {"label": value} if value else {}


def ingest_source(source: Source) -> list[Record]:
    """Read a source into records. Unapproved sources yield nothing."""
    if not source.approved:
        return []
    suffix = source.path.suffix.lower()
    text_field = source.text_field or "text"
    records: list[Record] = []
    if suffix in (".jsonl", ".ndjson"):
        rows = _read_jsonl(source.path)
        for index, row in enumerate(rows):
            text = str(row.get(text_field, "")).strip()
            if not text:
                continue
            records.append(
                Record(
                    id=f"{source.source_id}:{index}",
                    record_key=str(row.get("id") or row.get("record_key") or index),
                    text=text,
                    source_id=source.source_id,
                    sensitivity=str(row.get("sensitivity", source.sensitivity)),
                    acl=_bounded_acl(row, source),
                    meta={
                        "format": "jsonl",
                        "extra_keys": sorted(
                            k for k in row if k not in (text_field, source.label_field)
                        ),
                        **_label_meta(row, source),
                    },
                )
            )
    elif suffix == ".csv":
        rows = _read_csv(source.path)
        for index, row in enumerate(rows):
            text = str(row.get(text_field, "")).strip()
            if not text:
                continue
            records.append(
                Record(
                    id=f"{source.source_id}:{index}",
                    record_key=str(row.get("id") or index),
                    text=text,
                    source_id=source.source_id,
                    sensitivity=str(row.get("sensitivity", source.sensitivity)),
                    acl=_bounded_acl(row, source),
                    meta={"format": "csv", **_label_meta(row, source)},
                )
            )
    elif suffix in DOCUMENT_SUFFIXES:
        extracted = document_text(source.path)
        if not extracted.usable:
            return []
        for index, chunk in enumerate(_paragraphs(extracted.text)):
            records.append(
                Record(
                    id=f"{source.source_id}:{index}",
                    record_key=str(index),
                    text=chunk,
                    source_id=source.source_id,
                    sensitivity=source.sensitivity,
                    acl=list(source.acl),
                    meta={"format": suffix.lstrip(".")},
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
