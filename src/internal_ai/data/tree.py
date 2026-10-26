"""File-tree ingestion for code and documentation corpora.

A tree source turns a directory of files into dataset records with path
metadata, so retrieval can point at a file and a line rather than at an
opaque document id. Files that change between runs are re-ingested, so the
index can be refreshed cheaply as the code moves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from internal_ai.data.inventory import disguised_binary, target_escapes
from internal_ai.data.records import Record
from internal_ai.ids import file_hash, new_id, text_hash

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

MAX_FILE_BYTES = 2_000_000


@dataclass(slots=True)
class TreeFile:
    path: Path
    relative_path: str
    text: str
    content_hash: str
    suffix: str
    size: int
    line_count: int

    def to_record(self, source_id: str, acl: list[str]) -> Record:
        return Record(
            id=new_id("rec"),
            record_key=self.relative_path,
            text=self.text,
            source_id=source_id,
            sensitivity="internal",
            acl=list(acl),
            tokens=max(1, len(self.text) // 4),
            meta={
                "suffix": self.suffix,
                "bytes": self.size,
                "lines": self.line_count,
                "content_hash": self.content_hash,
                "path": self.relative_path,
            },
        )


@dataclass(slots=True)
class TreeIndex:
    root: Path
    source_id: str
    files: list[TreeFile] = field(default_factory=list)
    skipped: int = 0
    truncated: int = 0

    def records(self, acl: list[str] | None = None) -> list[Record]:
        access = acl if acl is not None else ["engineering", "support"]
        return [item.to_record(self.source_id, access) for item in self.files]

    def stats(self) -> dict[str, Any]:
        suffixes: dict[str, int] = {}
        for item in self.files:
            suffixes[item.suffix] = suffixes.get(item.suffix, 0) + 1
        return {
            "root": str(self.root),
            "files": len(self.files),
            "skipped": self.skipped,
            "truncated": self.truncated,
            "bytes": sum(item.size for item in self.files),
            "lines": sum(item.line_count for item in self.files),
            "formats": dict(sorted(suffixes.items(), key=lambda pair: -pair[1])),
        }


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name.lower() in ("dockerfile", "makefile", "readme", "license", "changelog")


def iter_tree_files(root: Path, follow_symlinks: bool = False) -> list[Path]:
    files: list[Path] = []
    for directory, directories, names in os.walk(root, topdown=True, followlinks=follow_symlinks):
        directories[:] = [
            name
            for name in directories
            if name not in SKIP_DIRECTORIES and not name.startswith(".")
        ]
        for name in names:
            if name.startswith("."):
                continue
            files.append(Path(directory) / name)
    return sorted(files)


def build_tree_index(
    root: Path,
    source_id: str | None = None,
    max_bytes: int = MAX_FILE_BYTES,
    previous_hashes: dict[str, str] | None = None,
) -> TreeIndex:
    """Index a tree of text files, skipping unchanged ones when hashes are known."""
    root = root.resolve()
    index = TreeIndex(root=root, source_id=source_id or root.name)
    for path in iter_tree_files(root):
        if not is_text_file(path) or target_escapes(path, root) or disguised_binary(path):
            index.skipped += 1
            continue
        try:
            size = path.stat().st_size
        except OSError:
            index.skipped += 1
            continue
        if size > max_bytes:
            index.truncated += 1
            continue
        relative = str(path.relative_to(root))
        digest = file_hash(path)
        if previous_hashes is not None and previous_hashes.get(relative) == digest:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            index.skipped += 1
            continue
        if not text.strip():
            index.skipped += 1
            continue
        index.files.append(
            TreeFile(
                path=path,
                relative_path=relative,
                text=text,
                content_hash=digest,
                suffix=path.suffix.lower() or "<none>",
                size=size,
                line_count=text.count("\n") + 1,
            )
        )
    return index


def index_hashes(index: TreeIndex) -> dict[str, str]:
    return {item.relative_path: item.content_hash for item in index.files}


def snippet_for(path_text: str, needle: str, context_lines: int = 2) -> str:
    lines = path_text.splitlines()
    lowered = needle.lower()
    for number, line in enumerate(lines):
        if lowered and lowered in line.lower():
            start = max(number - context_lines, 0)
            end = min(number + context_lines + 1, len(lines))
            return "\n".join(lines[start:end])
    return lines[0] if lines else ""


def stable_record_key(relative_path: str) -> str:
    return text_hash(relative_path)[:16]
