"""Identifier helpers.

Ids are deterministic and content-derived wherever possible so that runs are
reproducible from commit, config, dataset version and seed (NFR: Reproducibility).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def content_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path | str) -> str:
    """One hash over a directory's files, so an adapter directory has an identity.

    A saved PEFT adapter is a directory, not a file, and hashing only its name
    would let a different adapter take the place of the one an eval report named.
    """
    root = Path(path)
    digest = hashlib.sha256()
    for entry in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(entry.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with open(entry, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
