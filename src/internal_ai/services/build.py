"""The model build path: data source to promoted model, one guarded step at a time.

Every step checks the person's permission, refuses to run ahead of the gate it
depends on, and calls the existing data, training and evaluation code. Nothing
here trains differently from the CLI; it only adds the checks a browser user
must not be able to skip.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.data.extract import DOCUMENT_SUFFIXES, document_text
from internal_ai.data.inventory import (
    SECRET_PATTERNS,
    classify_path,
    disguised_binary,
    leave_out,
    target_escapes,
)
from internal_ai.data.records import ROW_SUFFIXES, choose_columns
from internal_ai.data.suggest import label_columns
from internal_ai.data.tree import is_text_file
from internal_ai.domain import GateRequest, GateStatus, GateType
from internal_ai.ids import new_id, now_iso
from internal_ai.services import governance
from internal_ai.services.actor import Actor
from internal_ai.services.jobs import JobRunner
from internal_ai.services.permissions import Action, PermissionDenied, deny, require

WEIGHT_SUFFIXES = frozenset(
    {".pt", ".pth", ".bin", ".safetensors", ".gguf", ".onnx", ".ckpt", ".h5", ".pb", ".tflite"}
)
SOURCE_FILE_SUFFIXES = frozenset({".jsonl", ".ndjson", ".csv", ".txt", ".md"}) | DOCUMENT_SUFFIXES


def _secret_hits(text: str) -> dict[str, int]:
    """Which secrets appear in a piece of text, and how many times."""
    return {
        name: len(pattern.findall(text))
        for name, pattern in SECRET_PATTERNS.items()
        if pattern.findall(text)
    }


TRAINING_COST_GATE_USD = 10.0
DATA_ROOTS_ENV = "INTERNAL_AI_DATA_ROOTS"
TRAINING_AGENT = "ml_engineer"
DATA_AGENT = "data_engineer"
EVAL_AGENT = "evaluation"


# ----------------------------------------------------------------- helpers


def data_roots(platform: Platform) -> list[Path]:
    """Directories a console user may point a data source at."""
    raw = os.environ.get(DATA_ROOTS_ENV, "")
    if raw.strip():
        return [Path(item).expanduser().resolve() for item in raw.split(os.pathsep) if item]
    return [(platform.settings.root / "data").resolve(), samples_dir(platform).resolve()]


def samples_dir(platform: Platform) -> Path:
    return platform.settings.home / "samples"


def gate_for(platform: Platform, gate_type: GateType, subject: str) -> GateRequest | None:
    """The most recent gate of a type for a subject, whatever its status."""
    gates = [
        gate
        for gate in platform.store.list_gates()
        if str(gate.gate_type) == str(gate_type) and gate.subject == subject
    ]
    gates.sort(key=lambda gate: (gate.created_at, gate.id))
    return gates[-1] if gates else None


def _approved(platform: Platform, gate_type: GateType, subject: str) -> GateRequest | None:
    gate = gate_for(platform, gate_type, subject)
    if gate is not None and gate.status == GateStatus.APPROVED:
        return gate
    return None


def _require_gate(
    platform: Platform, actor: Actor, gate_type: GateType, subject: str, what: str
) -> GateRequest:
    gate = _approved(platform, gate_type, subject)
    if gate is None:
        raise deny(
            platform.audit,
            actor,
            what,
            subject,
            f"{what.replace('_', ' ')} needs an approved {gate_type} gate for {subject}",
        )
    return gate


def _no_open_gate(platform: Platform, gate_type: GateType, subject: str) -> None:
    """Refuse a second gate while the first is still live.

    Release, freeze and promotion are one-way transitions: once the gate is
    approved the subject is consumed (released, frozen, promoted), and while
    it is pending there is already something to decide. A rejection closes
    the gate, so asking again after a rejection is allowed.
    """
    gate = gate_for(platform, gate_type, subject)
    if gate is not None and gate.status in (GateStatus.PENDING, GateStatus.APPROVED):
        raise ValueError(f"a {gate_type} gate for {subject} is already {gate.status} ({gate.id})")


# -------------------------------------------------------------- data sources


@dataclass(slots=True)
class SourceInspection:
    path: str
    kind: str
    sensitivity: str
    reason: str
    files: int
    text_files: int
    weight_files: list[str]
    secret_hits: dict[str, int]
    problems: list[str]
    formats: dict[str, int] = field(default_factory=dict)
    label_columns: list[str] = field(default_factory=list)
    text_field: str = ""
    label_field: str = ""
    unreadable: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sensitivity": self.sensitivity,
            "reason": self.reason,
            "files": self.files,
            "text_files": self.text_files,
            "weight_files": self.weight_files,
            "secret_hits": self.secret_hits,
            "problems": self.problems,
            "formats": dict(sorted(self.formats.items())),
            "label_columns": self.label_columns,
            "text_field": self.text_field,
            "label_field": self.label_field,
            "unreadable": self.unreadable,
        }


def inspect_source(
    platform: Platform,
    raw_path: str,
    progress: Callable[[int, str], None] | None = None,
) -> SourceInspection:
    """What the platform sees in a candidate source, and every reason to refuse it.

    A caller that shows progress gets a percentage and a short line describing
    the work: the file walk happens first, then the files are read one by one,
    so the number matches files actually examined.
    """

    def report(percent: int, detail: str) -> None:
        if progress is not None:
            progress(percent, detail)

    problems: list[str] = []
    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return SourceInspection(
            raw_path, "missing", "unknown", "", 0, 0, [], {}, ["the path does not exist"]
        )
    roots = data_roots(platform)
    containing = [root for root in roots if resolved == root or root in resolved.parents]
    if not containing:
        report(100, "done")
        return SourceInspection(
            str(resolved),
            "tree" if resolved.is_dir() else "file",
            *classify_path(resolved),
            0,
            0,
            [],
            {},
            [
                "the path is outside the approved data folders: "
                + ", ".join(str(root) for root in roots)
            ],
        )
    root = max(containing, key=lambda item: len(item.parts))
    below = resolved.relative_to(root)
    sensitivity, reason = classify_path(below if below.parts else Path(resolved.name))
    if sensitivity != "internal":
        report(100, "done")
        return SourceInspection(
            str(resolved),
            "tree" if resolved.is_dir() else "file",
            sensitivity,
            reason,
            0,
            0,
            [],
            {},
            [f"the path is classified as {sensitivity} ({reason}) and is excluded"],
        )
    weight_files: list[str] = []
    secret_hits: dict[str, int] = {}
    escaped: list[str] = []
    formats: dict[str, int] = {}
    labels: list[str] = []
    unreadable: list[str] = []
    text_field = ""
    label_field = ""
    if resolved.is_dir():
        kind = "tree"
        report(0, "listing files")
        candidates: list[Path] = []
        for directory, dirnames, names in os.walk(resolved, topdown=True):
            dirnames[:] = [
                name
                for name in dirnames
                if leave_out(Path(directory, name).relative_to(resolved)) is None
            ]
            candidates.extend(Path(directory, name) for name in names)
        total = len(candidates)

        def scanned(done: int, total_files: int) -> None:
            percent = int(90 * done / total_files) if total_files else 90
            report(percent, f"{done:,} of {total_files:,} files read")

        scanned(0, total)
        files = 0
        text_files = 0
        for done, file_path in enumerate(candidates, start=1):
            files += 1
            relative = str(file_path.relative_to(resolved))
            if leave_out(Path(relative)) or target_escapes(file_path, resolved):
                if target_escapes(file_path, resolved):
                    escaped.append(relative)
                scanned(done, total)
                continue
            suffix = file_path.suffix.lower()
            if suffix in WEIGHT_SUFFIXES or (
                is_text_file(file_path) and disguised_binary(file_path)
            ):
                weight_files.append(relative)
            elif suffix in DOCUMENT_SUFFIXES:
                extracted = document_text(file_path)
                if not extracted.usable:
                    unreadable.append(f"{relative}: {extracted.reason}")
                else:
                    hits = _secret_hits(extracted.text)
                    if hits:
                        for name, count in hits.items():
                            secret_hits[name] = secret_hits.get(name, 0) + count
                    else:
                        text_files += 1
                        formats[suffix] = formats.get(suffix, 0) + 1
            elif is_text_file(file_path):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    content = ""
                hits = _secret_hits(content)
                if hits:
                    for name, count in hits.items():
                        secret_hits[name] = secret_hits.get(name, 0) + count
                elif content.strip():
                    text_files += 1
                    suffix = file_path.suffix.lower() or "<none>"
                    formats[suffix] = formats.get(suffix, 0) + 1
                    for name in label_columns(file_path.suffix, content):
                        if name not in labels:
                            labels.append(name)
            scanned(done, total)
    else:
        kind = "file"
        report(30, "reading file")
        files, text_files = 1, 0
        suffix = resolved.suffix.lower()
        if suffix in WEIGHT_SUFFIXES:
            weight_files.append(resolved.name)
        elif suffix not in SOURCE_FILE_SUFFIXES:
            problems.append(
                f"{suffix or 'files without an extension'} is not a supported source format; "
                f"use one of {', '.join(sorted(SOURCE_FILE_SUFFIXES))}"
            )
        elif suffix in DOCUMENT_SUFFIXES:
            extracted = document_text(resolved)
            if not extracted.usable:
                problems.append(f"the file could not be read: {extracted.reason}")
            else:
                text_files = 1
                formats[suffix] = 1
                for name, count in _secret_hits(extracted.text).items():
                    secret_hits[name] = count
        elif disguised_binary(resolved):
            weight_files.append(resolved.name)
        else:
            text_files = 1
            formats[suffix or "<none>"] = 1
            content = resolved.read_text(encoding="utf-8", errors="ignore")
            labels = label_columns(suffix, content)
            if suffix in ROW_SUFFIXES:
                text_field, label_field = choose_columns(suffix, content)
            for name, pattern in SECRET_PATTERNS.items():
                hits = len(pattern.findall(content))
                if hits:
                    secret_hits[name] = hits
    nothing_left = kind == "tree" and text_files == 0
    if weight_files and (kind == "file" or nothing_left):
        problems.append(
            "the source contains model weight files, and pretrained weights are never used: "
            + ", ".join(sorted(weight_files)[:5])
        )
    if escaped and nothing_left:
        problems.append("a link points outside the source: " + ", ".join(sorted(escaped)[:5]))
    if secret_hits and nothing_left:
        problems.append(
            "the folder contains what look like secrets ("
            + ", ".join(f"{name}: {count}" for name, count in sorted(secret_hits.items()))
            + "); those files were left out, and nothing usable remains"
        )
    if nothing_left and unreadable:
        problems.append(
            f"{len(unreadable)} documents hold no readable text: " + "; ".join(unreadable[:3])
        )
    if nothing_left:
        problems.append("the folder has no text files to learn from")
    report(100, "done")
    return SourceInspection(
        str(resolved),
        kind,
        sensitivity,
        reason,
        files,
        text_files,
        sorted(weight_files),
        secret_hits,
        problems,
        dict(sorted(formats.items())),
        sorted(labels),
        text_field,
        label_field,
        unreadable,
    )


# What a guided run is aiming at, in the words shown on the form.
GUIDED_INTENTS: tuple[tuple[str, str], ...] = (
    ("search", "Search over it"),
    ("decide", "Sort it into labels"),
    ("generate", "Train a model on it"),
)

# The same three, in the words an approver reads on the gate evidence.
GUIDED_PURPOSES: dict[str, str] = {
    "search": "guided search over this folder",
    "decide": "guided classifier over this folder, sorting its rows by the answers it carries",
    "generate": "guided model trained from scratch on this folder",
}


def guided_intent(raw: str) -> str:
    """The kind of model a guided run aims at, defaulting to search."""
    return raw if raw in GUIDED_PURPOSES else "search"


def guided_register(
    platform: Platform, actor: Actor, raw_path: str, intent: str = "search"
) -> dict[str, Any]:
    """Register a pasted folder on the guided path with safe defaults.

    The novice supplies a path and, at most, which kind of model they are aiming
    at. The name comes from the folder, readers stay on the default group, and
    the purpose names the intent so the gate evidence says what is really being
    built. Permission and refusal rules are unchanged: this calls
    register_source and inherits its checks.
    """
    chosen = guided_intent(intent)
    stem = Path(raw_path).expanduser().stem.strip() or "guided"
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in stem)[:60]
    return register_source(
        platform,
        actor,
        raw_path,
        clean or "guided",
        GUIDED_PURPOSES[chosen],
        ["engineering"],
    )


def generate_sample(platform: Platform, actor: Actor, records: int = 120) -> str:
    """Write a synthetic support-ticket corpus into the samples folder."""
    from internal_ai.data.samples import generate_corpus

    require(platform.audit, actor, Action.REGISTER_SOURCE, "samples")
    target = samples_dir(platform) / f"synthetic-tickets-{records}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    generate_corpus(target, records=records)
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="data.sample_generated",
        subject=str(target.name),
        details={"records": records},
    )
    return str(target)


def register_source(
    platform: Platform,
    actor: Actor,
    raw_path: str,
    name: str,
    purpose: str,
    acl: list[str],
) -> dict[str, Any]:
    """Record a candidate source and open its new_data_source_access gate."""
    require(platform.audit, actor, Action.REGISTER_SOURCE, raw_path)
    inspection = inspect_source(platform, raw_path)
    if not inspection.acceptable:
        raise deny(
            platform.audit,
            actor,
            "register_source",
            raw_path,
            "the source was refused: " + "; ".join(inspection.problems),
        )
    if not purpose.strip():
        raise ValueError("say what the source will be used for")
    clean_name = (
        "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip()) or "source"
    )
    source_id = new_id("src")
    groups = sorted({item.strip() for item in acl if item.strip()}) or ["engineering"]
    gate = governance.request_gate(
        platform,
        actor,
        GateType.NEW_DATA_SOURCE,
        title=f"Use {clean_name} as training data",
        subject=source_id,
        evidence={
            "source_description": (
                f"{inspection.kind} at {inspection.path}: {inspection.files} files, "
                f"{inspection.text_files} text"
            ),
            "sensitivity_class": inspection.sensitivity,
            "purpose": purpose.strip(),
            "readers": ", ".join(groups),
            "scan": json.dumps(inspection.to_dict(), sort_keys=True),
        },
    )
    platform.store.conn.execute(
        "INSERT INTO data_sources (id, name, path, kind, acl, sensitivity, scan, purpose,"
        " requested_by, gate_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            source_id,
            clean_name,
            inspection.path,
            inspection.kind,
            json.dumps(groups),
            inspection.sensitivity,
            json.dumps(inspection.to_dict(), sort_keys=True),
            purpose.strip(),
            actor.id,
            gate.id,
            now_iso(),
        ),
    )
    return get_source(platform, source_id) or {}


def get_source(platform: Platform, source_id: str) -> dict[str, Any] | None:
    row = platform.store.conn.execute(
        "SELECT * FROM data_sources WHERE id = ?", (source_id,)
    ).fetchone()
    return _source_dict(platform, row) if row else None


def list_sources(platform: Platform) -> list[dict[str, Any]]:
    rows = platform.store.conn.execute(
        "SELECT * FROM data_sources ORDER BY created_at, id"
    ).fetchall()
    return [_source_dict(platform, row) for row in rows]


def _source_dict(platform: Platform, row: Any) -> dict[str, Any]:
    item = dict(row)
    item["acl"] = json.loads(item["acl"])
    item["scan"] = json.loads(item["scan"])
    gate = platform.store.get_gate(item["gate_id"]) if item["gate_id"] else None
    item["gate_status"] = str(gate.status) if gate else "none"
    withdrawn = platform.store.conn.execute(
        "SELECT reason FROM source_tombstones WHERE source_id = ?", (item["id"],)
    ).fetchone()
    item["withdrawn"] = withdrawn["reason"] if withdrawn else None
    item["purge_requested_by"] = platform.store.kv_get(f"purge_request:{item['id']}", "") or None
    return item


def build_dataset(platform: Platform, actor: Actor, runner: JobRunner, source_id: str) -> dict:
    """Ingest, deduplicate and scrub an approved source into a new dataset version."""
    require(platform.audit, actor, Action.BUILD_DATASET, source_id)
    source = get_source(platform, source_id)
    if source is None:
        raise KeyError(f"unknown source: {source_id}")
    if source["withdrawn"]:
        raise deny(platform.audit, actor, "build_dataset", source_id, "the source was withdrawn")
    _require_gate(platform, actor, GateType.NEW_DATA_SOURCE, source_id, "build_dataset")
    inspection = inspect_source(platform, source["path"])
    if not inspection.acceptable:
        raise deny(
            platform.audit,
            actor,
            "build_dataset",
            source_id,
            "the source changed since it was approved and is now refused: "
            + "; ".join(inspection.problems),
        )
    existing = platform.store.list_datasets(name=source["name"])
    version = max((item.version for item in existing), default=0) + 1

    def work() -> dict[str, Any]:
        if source["kind"] == "tree":
            from internal_ai.data.tree import build_tree_index
            from internal_ai.workflows import build_tree_corpus

            index = build_tree_index(Path(source["path"]), source_id=source_id)
            built = build_tree_corpus(
                platform, index, name=source["name"], version=version, acl=source["acl"]
            )
        else:
            from internal_ai.data.pipeline import DataPipeline, PipelineConfig
            from internal_ai.data.records import Source

            pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
            result = pipeline.run(
                [
                    Source(
                        source_id=source_id,
                        path=Path(source["path"]),
                        sensitivity="internal",
                        acl=source["acl"],
                        approved=True,
                        owner=source["requested_by"],
                        text_field=inspection.text_field or None,
                        label_field=inspection.label_field or None,
                    )
                ],
                PipelineConfig(name=source["name"], version=version),
                actor=DATA_AGENT,
                label_field=inspection.label_field,
            )
            built = result.dataset
        dataset_id = built.dataset_id if hasattr(built, "dataset_id") else built.id
        platform.store.conn.execute(
            "UPDATE data_sources SET dataset_id = ? WHERE id = ?", (dataset_id, source_id)
        )
        dataset = platform.store.get_dataset(dataset_id)
        return {
            "dataset_id": dataset_id,
            "records": dataset.record_count if dataset else 0,
            "tokens": dataset.token_count if dataset else 0,
        }

    return runner.submit(
        actor,
        "build_dataset",
        f"Build dataset {source['name']} v{version}",
        {"source_id": source_id, "version": version},
        work,
        agent=DATA_AGENT,
    )


# ------------------------------------------------------------------ datasets


def dataset_view(platform: Platform, actor: Actor, dataset_id: str) -> dict[str, Any]:
    """Statistics for everyone; a few scrubbed snippets only for allowed readers."""
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    principal = actor.principal()
    rows = platform.store.dataset_records(dataset_id)
    visible = [row for row in rows if principal.can_read(row["acl"], row["sensitivity"])]
    splits: dict[str, int] = {}
    for row in rows:
        splits[row.get("split") or "unassigned"] = (
            splits.get(row.get("split") or "unassigned", 0) + 1
        )
    snippets = [
        {
            "record_key": row["record_key"],
            "text": " ".join(row["payload"].split()[:40]),
        }
        for row in sorted(visible, key=lambda item: item["record_key"])[:3]
    ]
    return {
        "dataset": dataset,
        "records": len(rows),
        "visible_records": len(visible),
        "splits": splits,
        "has_splits": "train" in splits,
        "snippets": snippets,
        "steps": dataset.stats.get("steps", []),
        "pii_audit": dataset.stats.get("pii_audit"),
        "release_gate": gate_for(platform, GateType.DATASET_RELEASE, dataset_id),
    }


def split_dataset(platform: Platform, actor: Actor, dataset_id: str) -> dict[str, Any]:
    from internal_ai.workflows import split_corpus

    require(platform.audit, actor, Action.BUILD_DATASET, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if dataset.released:
        raise ValueError("a released dataset is immutable; build a new version to re-split")
    result = split_corpus(platform, dataset_id)
    return result.stats


def request_release(platform: Platform, actor: Actor, dataset_id: str) -> GateRequest:
    from internal_ai.data.pipeline import DataPipeline

    require(platform.audit, actor, Action.RELEASE_DATASET, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if dataset.released:
        raise ValueError("this dataset is already released")
    if "splits" not in dataset.stats:
        raise ValueError("split the dataset before asking for its release")
    _no_open_gate(platform, GateType.DATASET_RELEASE, dataset_id)
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    audit_result = pipeline.run_pii_audit(dataset_id, sample_size=None)
    if not audit_result["passed"]:
        raise ValueError(
            f"the PII audit found {audit_result['critical_findings']} critical findings; "
            "this dataset cannot be released"
        )
    dataset = platform.store.get_dataset(dataset_id) or dataset
    return governance.request_gate(
        platform,
        actor,
        GateType.DATASET_RELEASE,
        title=f"Release dataset {dataset.name} v{dataset.version}",
        subject=dataset_id,
        evidence={
            "quality_report": json.dumps(dataset.stats.get("steps", []), sort_keys=True),
            "pii_audit_sample": json.dumps(audit_result, sort_keys=True),
            "lineage": json.dumps(
                {
                    "content_hash": dataset.content_hash,
                    "sources": [
                        item.get("source_id") for item in dataset.lineage.get("sources", [])
                    ],
                    "splits": dataset.stats.get("splits", {}).get("counts"),
                },
                sort_keys=True,
            ),
        },
    )


def release_dataset(platform: Platform, actor: Actor, dataset_id: str) -> Any:
    from internal_ai.data.pipeline import DataPipeline

    require(platform.audit, actor, Action.RELEASE_DATASET, dataset_id)
    _require_gate(platform, actor, GateType.DATASET_RELEASE, dataset_id, "release_dataset")
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    return pipeline.release(dataset_id, actor.id, gate_approved=True)


def withdraw_source(platform: Platform, actor: Actor, source_id: str, reason: str) -> dict:
    """Stop using a source in new dataset versions. Reduces risk, so it is immediate."""
    from internal_ai.data.governance import RetentionManager

    require(platform.audit, actor, Action.WITHDRAW_SOURCE, source_id)
    if not reason.strip():
        raise ValueError("withdrawing a source needs a typed reason")
    return RetentionManager(platform.store, platform.audit).withdraw_source(
        source_id, reason.strip(), actor.id
    )


def request_purge(platform: Platform, actor: Actor, source_id: str, reason: str) -> None:
    """First half of an irreversible erasure; a second person confirms it."""
    require(platform.audit, actor, Action.WITHDRAW_SOURCE, source_id)
    if not reason.strip():
        raise ValueError("a purge needs a typed reason")
    platform.store.kv_set(f"purge_request:{source_id}", actor.id)
    platform.store.kv_set(f"purge_reason:{source_id}", reason.strip())
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="data.purge_requested",
        subject=source_id,
        details={"reason": reason.strip()},
    )


def confirm_purge(platform: Platform, actor: Actor, source_id: str) -> int:
    from internal_ai.data.governance import RetentionManager

    require(platform.audit, actor, Action.WITHDRAW_SOURCE, source_id)
    requested_by = platform.store.kv_get(f"purge_request:{source_id}", "") or ""
    if not requested_by:
        raise ValueError("no purge was requested for this source")
    if requested_by == actor.id:
        raise deny(
            platform.audit,
            actor,
            "confirm_purge",
            source_id,
            "the person who requested a purge cannot confirm it; ask a second person",
        )
    reason = platform.store.kv_get(f"purge_reason:{source_id}", "") or "erasure"
    manager = RetentionManager(platform.store, platform.audit)
    manager.withdraw_source(source_id, reason, actor.id)
    removed = manager.purge_source_records(source_id, actor.id, reason)
    platform.store.kv_set(f"purge_request:{source_id}", "")
    return removed


# ------------------------------------------------------------- tokenizer


def train_tokenizer(
    platform: Platform, actor: Actor, runner: JobRunner, dataset_id: str, vocab_size: int = 4000
) -> dict[str, Any]:
    from internal_ai.workflows import train_tokenizer_for

    require(platform.audit, actor, Action.BUILD_DATASET, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if "splits" not in dataset.stats:
        raise ValueError("split the dataset first, so the tokenizer only sees the training split")
    if not 256 <= vocab_size <= 64_000:
        raise ValueError("choose a vocabulary between 256 and 64000 tokens")

    def work() -> dict[str, Any]:
        _tokenizer, coverage = train_tokenizer_for(
            platform, dataset_id, vocab_size=vocab_size, actor=DATA_AGENT
        )
        return coverage

    return runner.submit(
        actor,
        "train_tokenizer",
        f"Train tokenizer on {dataset.name} v{dataset.version}",
        {"dataset_id": dataset_id, "vocab_size": vocab_size},
        work,
        agent=DATA_AGENT,
    )


def tokenizer_path(platform: Platform, dataset_id: str) -> Path | None:
    path = platform.settings.models_dir / f"tokenizer-{dataset_id}.json"
    return path if path.exists() else None


# --------------------------------------------------------------- eval sets


def create_eval_set(
    platform: Platform,
    actor: Actor,
    dataset_id: str,
    name: str,
    mode: str = "paraphrase",
    count: int = 24,
) -> Any:
    from internal_ai.evals.harness import EvalHarness
    from internal_ai.workflows import derive_eval_items

    require(platform.audit, actor, Action.CREATE_EVAL_SET, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if "splits" not in dataset.stats:
        raise ValueError("split the dataset first, so eval questions come from held-out records")
    if mode not in ("paraphrase", "title"):
        raise ValueError("mode is paraphrase or title")
    items = derive_eval_items(platform, dataset_id, count=count, mode=mode)
    if not items:
        raise ValueError("no eval questions could be derived from this dataset; try the title mode")
    for item in items:
        item["dataset_id"] = dataset_id
    harness = EvalHarness(platform.store, platform.audit)
    existing = [item for item in platform.store.list_eval_sets() if item.name == name]
    version = max((item.version for item in existing), default=0) + 1
    return harness.create_eval_set(name, items, version=version, actor=actor.id)


def eval_set_leakage(platform: Platform, eval_set_id: str) -> dict[str, Any]:
    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise KeyError(f"unknown eval set: {eval_set_id}")
    dataset_ids = {item.get("dataset_id") for item in eval_set.items if item.get("dataset_id")}
    held_out: set[str] = set()
    for dataset_id in dataset_ids:
        held_out |= {
            row["record_key"] for row in platform.store.dataset_records_by_split(dataset_id, "test")
        }
    relevant = {doc for item in eval_set.items for doc in item.get("relevant_ids", [])}
    leaked = sorted(relevant - held_out)
    return {
        "items": len(eval_set.items),
        "relevant_documents": len(relevant),
        "outside_test_split": leaked[:20],
        "passed": not leaked and bool(relevant),
        "datasets": sorted(item for item in dataset_ids if item),
    }


def request_freeze(platform: Platform, actor: Actor, eval_set_id: str) -> GateRequest:
    require(platform.audit, actor, Action.CREATE_EVAL_SET, eval_set_id)
    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise KeyError(f"unknown eval set: {eval_set_id}")
    if eval_set.frozen:
        raise ValueError("this eval set is already frozen")
    _no_open_gate(platform, GateType.EVAL_SET_FREEZE, eval_set_id)
    leakage = eval_set_leakage(platform, eval_set_id)
    if not leakage["passed"]:
        raise ValueError(
            "some eval answers are not in the held-out test split, so the set would leak "
            "training data: " + ", ".join(leakage["outside_test_split"])
        )
    sources = sorted({str(item.get("source_id")) for item in eval_set.items})
    return governance.request_gate(
        platform,
        actor,
        GateType.EVAL_SET_FREEZE,
        title=f"Freeze eval set {eval_set.name} v{eval_set.version}",
        subject=eval_set_id,
        evidence={
            "coverage_analysis": (
                f"{leakage['items']} questions over {leakage['relevant_documents']} held-out "
                f"documents from sources {', '.join(sources)}"
            ),
            "leakage_check": json.dumps(leakage, sort_keys=True),
        },
    )


def freeze_eval_set(platform: Platform, actor: Actor, eval_set_id: str) -> Any:
    from internal_ai.evals.harness import EvalHarness

    require(platform.audit, actor, Action.FREEZE_EVAL_SET, eval_set_id)
    _require_gate(platform, actor, GateType.EVAL_SET_FREEZE, eval_set_id, "freeze_eval_set")
    harness = EvalHarness(platform.store, platform.audit)
    return harness.freeze_eval_set(eval_set_id, actor.id, gate_approved=True)


def edit_eval_set(platform: Platform, actor: Actor, eval_set_id: str) -> None:
    """There is no edit; a frozen set refuses and an open set must be re-created."""
    from internal_ai.evals.harness import EvalHarness

    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise KeyError(f"unknown eval set: {eval_set_id}")
    if eval_set.frozen:
        EvalHarness(platform.store, platform.audit).write_to_frozen_set(eval_set_id, actor.id)
    raise ValueError("eval sets are not edited; create a new version instead")


# ----------------------------------------------------------------- training


def train_model(
    platform: Platform,
    actor: Actor,
    runner: JobRunner,
    dataset_id: str,
    name: str = "embedding-v1",
    steps: int = 60,
    cost_cap_usd: float = 2.0,
    pair_mode: str = "paraphrase",
    learning_rate: float = 3e-4,
    seed: int = 17,
) -> dict[str, Any]:
    """Train a from-scratch embedding model on the training split of a released dataset."""
    from internal_ai.data.tokenizer import Tokenizer
    from internal_ai.training.trainer import TrainConfig
    from internal_ai.workflows import train_embedding_model

    require(platform.audit, actor, Action.TRAIN_MODEL, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if not dataset.released:
        raise deny(
            platform.audit,
            actor,
            "train_model",
            dataset_id,
            "training uses released datasets only; release this dataset first",
        )
    token_file = tokenizer_path(platform, dataset_id)
    if token_file is None:
        raise ValueError("train a tokenizer for this dataset first")
    if not 1 <= steps <= 5000:
        raise ValueError("choose between 1 and 5000 training steps")
    if cost_cap_usd <= 0:
        raise ValueError("a training run needs a positive cost cap")
    if pair_mode not in ("paraphrase", "title"):
        raise ValueError("pair mode is paraphrase or title")
    subject = f"{dataset_id}:{name}"
    if cost_cap_usd > TRAINING_COST_GATE_USD:
        _require_gate(platform, actor, GateType.TRAINING_RUN_COST, subject, "train_model")
    config = TrainConfig(
        name=name,
        steps=steps,
        cost_cap_usd=cost_cap_usd,
        learning_rate=learning_rate,
        seed=seed,
    )

    def work() -> dict[str, Any]:
        tokenizer = Tokenizer.load(token_file)
        outcome = train_embedding_model(
            platform, dataset_id, tokenizer, config, actor=TRAINING_AGENT, pair_mode=pair_mode
        )
        result = outcome.to_dict()
        result["dataset_id"] = dataset_id
        result["tokenizer_path"] = str(token_file)
        result["name"] = name
        return result

    return runner.submit(
        actor,
        "train_model",
        f"Train {name} on {dataset.name} v{dataset.version}",
        {
            "dataset_id": dataset_id,
            "name": name,
            "steps": steps,
            "pair_mode": pair_mode,
            "learning_rate": learning_rate,
        },
        work,
        agent=TRAINING_AGENT,
        cost_cap_usd=cost_cap_usd,
    )


def register_model(platform: Platform, actor: Actor, runner: JobRunner, job_id: str) -> Any:
    from internal_ai.training.trainer import TrainingOutcome
    from internal_ai.workflows import register_model as register

    require(platform.audit, actor, Action.REGISTER_MODEL, job_id)
    job = runner.get(job_id)
    if job is None or job["kind"] != "train_model":
        raise KeyError(f"unknown training job: {job_id}")
    result = job["result"]
    if job["state"] != "succeeded" or result.get("status") != "completed":
        raise ValueError(
            "only a completed training run can be registered; this one ended as "
            f"{result.get('status') or job['state']}"
        )
    already = [
        model
        for model in platform.store.list_models()
        if model.card.get("run_id") == result["run_id"]
    ]
    if already:
        raise ValueError(f"run {result['run_id']} is already registered as {already[0].id}")
    known = set(TrainingOutcome.__slots__)
    outcome = TrainingOutcome(**{key: value for key, value in result.items() if key in known})
    model = register(
        platform,
        result["name"],
        Path(result["model_path"]),
        outcome,
        result["dataset_id"],
        actor=actor.id,
    )
    card = dict(model.card)
    card["run_id"] = result["run_id"]
    card["tokenizer_path"] = result["tokenizer_path"]
    card["registered_by"] = actor.id
    platform.store.update_model_card(model.id, card)
    return platform.store.get_model(model.id)


def evaluate_model(
    platform: Platform,
    actor: Actor,
    runner: JobRunner,
    model_id: str,
    eval_set_id: str,
    margin: float = 0.0,
) -> dict[str, Any]:
    from internal_ai.data.tokenizer import Tokenizer
    from internal_ai.workflows import evaluate_embedding_model

    require(platform.audit, actor, Action.EVALUATE_MODEL, model_id)
    model = platform.store.get_model(model_id)
    if model is None:
        raise KeyError(f"unknown model: {model_id}")
    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise KeyError(f"unknown eval set: {eval_set_id}")
    if not eval_set.frozen:
        raise deny(
            platform.audit,
            actor,
            "evaluate_model",
            eval_set_id,
            "evaluation runs against frozen eval sets only; freeze this set first",
        )
    dataset_id = model.dataset_ids[0]
    token_file = Path(model.card.get("tokenizer_path") or "")
    if not token_file.exists():
        raise ValueError("the tokenizer this model was trained with is missing")

    def work() -> dict[str, Any]:
        return evaluate_embedding_model(
            platform,
            Path(model.path),
            eval_set_id,
            Tokenizer.load(token_file),
            dataset_id,
            margin=margin,
            actor=EVAL_AGENT,
            model_id=model_id,
        )

    return runner.submit(
        actor,
        "evaluate_model",
        f"Evaluate {model.name} v{model.version} on {eval_set.name} v{eval_set.version}",
        {"model_id": model_id, "eval_set_id": eval_set_id, "margin": margin},
        work,
        agent=EVAL_AGENT,
    )


def classifier_encoders(platform: Platform) -> list[Any]:
    """Registered models that can be the frozen encoder a classifier sits on."""
    ready = []
    for model in platform.store.list_models():
        token_file = Path(str(model.card.get("tokenizer_path") or ""))
        if Path(model.path).exists() and token_file.name and token_file.exists():
            ready.append(model)
    return ready


def label_summary(platform: Platform, dataset_id: str) -> dict[str, int]:
    """Where the answers came from: the data's own column, or the shipped rules."""
    rows = platform.store.dataset_records(dataset_id)
    from_data = sum(1 for row in rows if (row.get("meta") or {}).get("label"))
    return {"rows": len(rows), "from_data": from_data, "from_rules": len(rows) - from_data}


def train_classifier(
    platform: Platform,
    actor: Actor,
    runner: JobRunner,
    dataset_id: str,
    encoder_model_id: str,
    name: str = "router-v1",
    epochs: int = 40,
    cost_cap_usd: float = 2.0,
) -> dict[str, Any]:
    """Train a task classifier: a linear head over a frozen from-scratch encoder.

    The answers come from the data when the source had an answer column, and
    from the shipped keyword rules otherwise. The run is deliberately not
    promotable: promotion looks for an eval report on a frozen retrieval set,
    and a classifier's held-out split is not one.
    """
    from internal_ai.training.classifier import ClassifierConfig

    require(platform.audit, actor, Action.TRAIN_MODEL, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if not dataset.released:
        raise deny(
            platform.audit,
            actor,
            "train_classifier",
            dataset_id,
            "training uses released datasets only; release this dataset first",
        )
    token_file = tokenizer_path(platform, dataset_id)
    if token_file is None:
        raise ValueError("train a tokenizer for this dataset first")
    encoder = platform.store.get_model(encoder_model_id)
    if encoder is None:
        raise KeyError(f"unknown model: {encoder_model_id}")
    encoder_path = Path(encoder.path)
    if not encoder_path.exists():
        raise ValueError("the encoder this classifier sits on is missing")
    if not 1 <= epochs <= 500:
        raise ValueError("choose between 1 and 500 training epochs")
    if cost_cap_usd <= 0:
        raise ValueError("a training run needs a positive cost cap")
    subject = f"{dataset_id}:{name}"
    if cost_cap_usd > TRAINING_COST_GATE_USD:
        _require_gate(platform, actor, GateType.TRAINING_RUN_COST, subject, "train_classifier")
    config = ClassifierConfig(name=name, epochs=epochs, cost_cap_usd=cost_cap_usd)

    def work() -> dict[str, Any]:
        from internal_ai.data.tokenizer import Tokenizer
        from internal_ai.workflows import train_classifier_model

        outcome = train_classifier_model(
            platform,
            dataset_id,
            encoder_path,
            Tokenizer.load(token_file),
            config,
            actor=TRAINING_AGENT,
        )
        result = outcome.to_dict()
        result.update(
            {
                "dataset_id": dataset_id,
                "name": name,
                "encoder_model_id": encoder_model_id,
                "tokenizer_path": str(token_file),
                "label_summary": label_summary(platform, dataset_id),
            }
        )
        return result

    return runner.submit(
        actor,
        "train_classifier",
        f"Train classifier {name} on {dataset.name} v{dataset.version}",
        {
            "dataset_id": dataset_id,
            "encoder_model_id": encoder_model_id,
            "name": name,
            "epochs": epochs,
        },
        work,
        agent=TRAINING_AGENT,
        cost_cap_usd=cost_cap_usd,
    )


def evaluate_classifier(
    platform: Platform, actor: Actor, runner: JobRunner, job_id: str, split: str = "test"
) -> dict[str, Any]:
    """Score a classifier on a held-out split against the majority class."""
    require(platform.audit, actor, Action.EVALUATE_MODEL, job_id)
    job = runner.get(job_id)
    if job is None or job["kind"] != "train_classifier":
        raise KeyError(f"unknown classifier run: {job_id}")
    if job["state"] != "succeeded":
        raise ValueError("only a completed classifier run can be evaluated")
    if split not in ("train", "validation", "test"):
        raise ValueError("the split is train, validation or test")
    result = job["result"]
    model_path = Path(result["model_path"])
    token_file = Path(result["tokenizer_path"])
    dataset_id = result["dataset_id"]

    def work() -> dict[str, Any]:
        from internal_ai.data.tokenizer import Tokenizer
        from internal_ai.workflows import evaluate_classifier_model

        report = evaluate_classifier_model(
            platform,
            model_path,
            Tokenizer.load(token_file),
            dataset_id,
            split=split,
            actor=EVAL_AGENT,
        )
        return {"run_id": result["run_id"], "name": result["name"], **report}

    return runner.submit(
        actor,
        "evaluate_classifier",
        f"Evaluate classifier {result['name']} on the {split} split",
        {
            "job_id": job_id,
            "run_id": result["run_id"],
            "dataset_id": dataset_id,
            "split": split,
        },
        work,
        agent=EVAL_AGENT,
    )


def latest_report(platform: Platform, model_id: str) -> Any:
    reports = platform.store.list_eval_reports(model_id)
    return reports[-1] if reports else None


def request_promotion(
    platform: Platform, actor: Actor, model_id: str, red_team_results: str
) -> GateRequest:
    from internal_ai.workflows import latest_frozen_report, model_card_markdown

    require(platform.audit, actor, Action.PROMOTE_MODEL, model_id)
    model = platform.store.get_model(model_id)
    if model is None:
        raise KeyError(f"unknown model: {model_id}")
    report = latest_frozen_report(platform, model_id)
    if report is None:
        raise ValueError("evaluate the model against a frozen eval set first")
    if not red_team_results.strip():
        raise ValueError("describe the red-team results: memorisation, PII and permission tests")
    _no_open_gate(platform, GateType.MODEL_PROMOTION, model_id)
    return governance.request_gate(
        platform,
        actor,
        GateType.MODEL_PROMOTION,
        title=f"Promote {model.name} v{model.version}",
        subject=model_id,
        evidence={
            "eval_results_vs_baseline": json.dumps(
                {
                    "report_id": report.id,
                    "passed": report.passed,
                    "metrics": report.metrics,
                    "baseline": report.baseline_metrics,
                    "notes": report.notes,
                },
                sort_keys=True,
            ),
            "model_card": model_card_markdown(model),
            "red_team_results": red_team_results.strip(),
        },
    )


def promote(platform: Platform, actor: Actor, model_id: str) -> Any:
    from internal_ai.workflows import promote_model

    require(platform.audit, actor, Action.PROMOTE_MODEL, model_id)
    _require_gate(platform, actor, GateType.MODEL_PROMOTION, model_id, "promote_model")
    try:
        return promote_model(platform, model_id, gate_approved=True, approver=actor.id)
    except PermissionError as exc:
        raise PermissionDenied(str(exc)) from exc


# ---------------------------------------------------------------------- ask


def ask(
    platform: Platform, actor: Actor, dataset_id: str, question: str, max_steps: int = 4
) -> Any:
    from internal_ai.agentic import AgenticSearcher

    require(platform.audit, actor, Action.ASK, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if not dataset.released:
        raise deny(
            platform.audit,
            actor,
            "ask",
            dataset_id,
            "questions are answered from released datasets only",
        )
    if not question.strip():
        raise ValueError("type a question")
    searcher = AgenticSearcher(platform.store, platform.audit, platform.budget, embedder=None)
    return searcher.search(
        question.strip()[:500],
        dataset_id,
        actor.principal(),
        max_steps=max(1, min(max_steps, 8)),
        actor=actor.id,
    )
