"""Where a source's records come from, and the ACL each one arrived with.

A path on disk is registered directly. A connector is the other door: a git
working tree, or a remote system this install cannot see into yet. Every
record comes back with the groups that may read it. A record that arrives
without that list is refused, because an unreported ACL is not an open
document and must not be ingested as one.

SharePoint, Confluence and an ERP are named so a caller cannot pick them and
get silence. Until an account can report their ACLs, they refuse.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = frozenset(
    {".md", ".txt", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".py", ".html"}
)
SKIP_DIRECTORIES = frozenset({".git", ".venv", "node_modules", "__pycache__"})
MAX_BYTES = 262_144
KINDS = ("git", "sharepoint", "confluence", "erp")
REMOTE_KINDS = frozenset({"sharepoint", "confluence", "erp"})


class AclUnreported(PermissionError):
    """A connector returned a record whose access list it did not look up."""


@dataclass(frozen=True, slots=True)
class Pulled:
    key: str
    text: str
    acl: tuple[str, ...] | None


def require_acl(rows: list[Pulled]) -> list[Pulled]:
    """Refuse the batch when any record did not bring an ACL."""
    missing = [row.key for row in rows if row.acl is None]
    if missing:
        shown = ", ".join(missing[:5])
        raise AclUnreported(
            f"{len(missing)} record(s) arrived without an ACL ({shown}); "
            "a connector that cannot report access is refused"
        )
    return rows


def pull(
    kind: str,
    path: str,
    acl: list[str] | None,
    cursor: str = "",
) -> tuple[list[Pulled], str]:
    """Records from this connector, and the cursor a later pull should send.

    The same cursor means nothing changed, so the second result is empty.
    """
    name = (kind or "").strip().lower()
    if name in REMOTE_KINDS:
        raise AclUnreported(
            f"{name} cannot report an ACL until an account is configured; "
            "refusing the source rather than ingesting it as open"
        )
    if name != "git":
        raise AclUnreported(f"unknown connector {kind!r}; expected one of {', '.join(KINDS)}")
    rows, digest = _git_tree(Path(path), None if acl is None else tuple(acl))
    require_acl(rows)
    if cursor and cursor == digest:
        return [], digest
    return rows, digest


def _git_tree(root: Path, acl: tuple[str, ...] | None) -> tuple[list[Pulled], str]:
    if not root.is_dir():
        raise AclUnreported(f"{root} is not a directory a git connector can read")
    base = root.resolve()
    rows: list[Pulled] = []
    parts: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRECTORIES)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            resolved = path.resolve()
            if not _inside(base, resolved):
                continue
            if resolved.stat().st_size > MAX_BYTES:
                continue
            text = resolved.read_text(encoding="utf-8", errors="replace")
            relative = resolved.relative_to(base).as_posix()
            rows.append(Pulled(key=relative, text=text, acl=acl))
            parts.append(f"{relative}\0{hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return rows, digest


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
