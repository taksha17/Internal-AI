"""Data inventory: catalogue candidate sources before anything is ingested.

The PRD requires inventory first (section 11): catalogue sources, owners,
volume, formats, sensitivity class and legal status before a source is
approved. This scanner produces that catalogue without reading record content
into the corpus. It walks a tree, counts what is there, classifies each
directory by sensitivity, and flags likely secrets and personal data.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".md",
        ".rst",
        ".txt",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
        ".tsv",
        ".html",
        ".css",
        ".scss",
        ".sql",
        ".sh",
        ".bash",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".rb",
        ".php",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".swift",
        ".ipynb",
        ".gradle",
        ".dockerfile",
        ".lock",
    }
)

SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".cache",
        ".idea",
        ".vscode",
        "site-packages",
        ".tox",
        "target",
        "System Volume Information",
        "$RECYCLE.BIN",
    }
)

BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".mp3",
        ".wav",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".db",
        ".sqlite",
        ".sqlite3",
    }
)

SENSITIVITY_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("regulated", "emr", ("emr", "medical", "health", "patient", "clinical", "hipaa", "phi")),
    (
        "personal",
        "personal",
        ("iphone", "photo", "pics", "trip", "movies", "valorant", "resume", "career", "job search"),
    ),
    (
        "confidential",
        "confidential",
        ("customer", "billing", "invoice", "payroll", "hr", "legal", "kite-invoicews"),
    ),
    (
        "pretrained_weights",
        "pretrained weights",
        ("ollama", "models", "weights", "huggingface", "checkpoints"),
    ),
    (
        "recovered",
        "recovered files",
        ("found.000", "found.001", ".trash-1000", "$recycle.bin"),
    ),
    (
        "system",
        "system or downloaded content",
        (".caches", "software ( downloads )", "docker-data"),
    ),
    ("internal", "internal", ()),
)

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "api_token": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "password_assignment": re.compile(
        r"(?i)\b(password|passwd|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}"
    ),
    "connection_string": re.compile(r"(?i)\b(?:postgres|mysql|mongodb|redis)://[^\s:@]+:[^\s@]+@"),
}


@dataclass(slots=True)
class SourceEntry:
    path: str
    name: str
    file_count: int = 0
    text_file_count: int = 0
    binary_file_count: int = 0
    total_bytes: int = 0
    formats: dict[str, int] = field(default_factory=dict)
    sensitivity: str = "internal"
    sensitivity_reason: str = "no sensitive markers in the path"
    secret_hits: dict[str, int] = field(default_factory=dict)
    owner: str = ""
    status: str = "candidate"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "file_count": self.file_count,
            "text_file_count": self.text_file_count,
            "binary_file_count": self.binary_file_count,
            "total_bytes": self.total_bytes,
            "formats": dict(sorted(self.formats.items(), key=lambda item: -item[1])),
            "sensitivity": self.sensitivity,
            "sensitivity_reason": self.sensitivity_reason,
            "secret_hits": dict(sorted(self.secret_hits.items())),
            "owner": self.owner,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass(slots=True)
class Inventory:
    root: str
    entries: list[SourceEntry] = field(default_factory=list)
    scanned_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "scanned_at": self.scanned_at,
            "sources": [entry.to_dict() for entry in self.entries],
            "summary": {
                "sources": len(self.entries),
                "total_bytes": sum(entry.total_bytes for entry in self.entries),
                "by_sensitivity": {
                    sensitivity: sum(
                        1 for entry in self.entries if entry.sensitivity == sensitivity
                    )
                    for sensitivity in sorted({entry.sensitivity for entry in self.entries})
                },
                "sources_with_secrets": sum(1 for entry in self.entries if entry.secret_hits),
            },
        }

    def candidates(self) -> list[SourceEntry]:
        return [entry for entry in self.entries if entry.sensitivity == "internal"]

    def excluded(self) -> list[SourceEntry]:
        return [entry for entry in self.entries if entry.sensitivity != "internal"]


def classify_path(path: Path) -> tuple[str, str]:
    lowered = str(path).lower()
    for sensitivity, _reason, markers in SENSITIVITY_RULES:
        for marker in markers:
            if marker in lowered:
                return sensitivity, f"path contains the marker '{marker}'"
    return "internal", "no sensitive markers in the path"


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name.lower() in ("dockerfile", "makefile", "readme", "license")


def scan_source(path: Path, max_bytes_per_file: int = 2_000_000) -> SourceEntry:
    sensitivity, reason = classify_path(path)
    entry = SourceEntry(
        path=str(path), name=path.name, sensitivity=sensitivity, sensitivity_reason=reason
    )
    for root, directories, files in os.walk(path, topdown=True):
        directories[:] = [
            name
            for name in directories
            if name not in SKIP_DIRECTORIES and not name.startswith(".cache")
        ]
        for filename in files:
            file_path = Path(root) / filename
            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            entry.file_count += 1
            entry.total_bytes += size
            suffix = file_path.suffix.lower() or "<none>"
            entry.formats[suffix] = entry.formats.get(suffix, 0) + 1
            if suffix in BINARY_SUFFIXES:
                entry.binary_file_count += 1
                continue
            if not _is_text_file(file_path):
                entry.binary_file_count += 1
                continue
            entry.text_file_count += 1
            if size > max_bytes_per_file:
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for kind, pattern in SECRET_PATTERNS.items():
                hits = len(pattern.findall(content))
                if hits:
                    entry.secret_hits[kind] = entry.secret_hits.get(kind, 0) + hits
    return entry


def inventory(root: Path, max_depth: int = 2) -> Inventory:
    """Catalogue the immediate children of a root as candidate sources."""
    from llmforge.ids import now_iso

    entries: list[SourceEntry] = []
    root = root.resolve()
    children = sorted([child for child in root.iterdir() if child.is_dir()], key=lambda p: p.name)
    for child in children:
        if child.name in SKIP_DIRECTORIES:
            continue
        entries.append(scan_source(child))
    loose_files = [child for child in root.iterdir() if child.is_file()]
    if loose_files:
        entry = SourceEntry(
            path=str(root),
            name=root.name,
            file_count=len(loose_files),
            total_bytes=sum(item.stat().st_size for item in loose_files),
            sensitivity="internal",
            sensitivity_reason="loose files at the root of the tree",
        )
        for item in loose_files:
            suffix = item.suffix.lower() or "<none>"
            entry.formats[suffix] = entry.formats.get(suffix, 0) + 1
        entries.append(entry)
    return Inventory(root=str(root), entries=entries, scanned_at=now_iso())


def render_inventory(report: dict[str, Any]) -> str:
    lines = [
        f"Data inventory for {report['root']}",
        f"scanned {report['summary']['sources']} candidate sources, "
        f"{report['summary']['total_bytes'] / 1_000_000:.1f} MB",
        "",
    ]
    header = f"{'source':<46} {'files':>6} {'text':>6} {'MB':>8}  {'sensitivity':<12} secrets"
    lines.append(header)
    lines.append("-" * len(header))
    for entry in report["sources"]:
        lines.append(
            f"{entry['name'][:45]:<46} {entry['file_count']:>6} {entry['text_file_count']:>6} "
            f"{entry['total_bytes'] / 1_000_000:>8.2f}  {entry['sensitivity']:<12} "
            f"{sum(entry['secret_hits'].values())}"
        )
    lines.append("")
    for entry in report["sources"]:
        if entry["sensitivity"] != "internal":
            lines.append(
                f"  excluded {entry['name']}: {entry['sensitivity']} "
                f"({entry['sensitivity_reason']})"
            )
    return "\n".join(lines)
