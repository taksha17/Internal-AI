"""What a model artifact was, kept somewhere it can be checked and rebuilt.

An adapter directory is the thing a promoted model actually consists of, and a
local path to it is a promise nobody verifies. A copy that lost a file, a
checkpoint overwritten in place, a directory that grew a file after the card was
written -- each is silent at load time and shows up as a model that answers
differently.

So an artifact is stored as two kinds of object: every file under a key that is
its own sha256, and a manifest that lists name, hash and size for the whole
directory. The manifest's own hash is the artifact's identity, which is what the
model card records. Materialising it somewhere else rebuilds the directory and
checks each file against the manifest, so a rebuilt copy is byte-identical or it
is refused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from internal_ai.storage import BlobStore, StorageDenied

MANIFEST_NAME = "manifest.jsonl"
DEFAULT_PREFIX = "artifacts"


class ArtifactDenied(RuntimeError):
    """The artifact was refused rather than stored or rebuilt incompletely."""


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class Artifact:
    manifest_key: str
    digest: str
    files: tuple[ArtifactFile, ...]
    size: int
    prefix: str = DEFAULT_PREFIX

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest_key,
            "digest": self.digest,
            "files": len(self.files),
            "size": self.size,
        }


def collect(path: Path) -> tuple[ArtifactFile, ...]:
    """Every file in the artifact, named relative to it and sorted by that name.

    Symlinks are refused: an artifact that pulls in a file from elsewhere cannot
    be rebuilt from its own manifest, which is the only reason to have one.
    """
    root = Path(path)
    if root.is_symlink():
        raise ArtifactDenied(f"{root} is a symlink; an artifact is a copy, not a pointer")
    if root.is_file():
        return (
            ArtifactFile(
                name=root.name,
                sha256=hashlib.sha256(root.read_bytes()).hexdigest(),
                size=root.stat().st_size,
            ),
        )
    if not root.is_dir():
        raise ArtifactDenied(f"{root} does not exist")
    entries: list[ArtifactFile] = []
    for child in sorted(root.rglob("*")):
        if child.is_symlink():
            raise ArtifactDenied(f"{child} is a symlink; an artifact is a copy, not a pointer")
        if not child.is_file():
            continue
        entries.append(
            ArtifactFile(
                name=child.relative_to(root).as_posix(),
                sha256=hashlib.sha256(child.read_bytes()).hexdigest(),
                size=child.stat().st_size,
            )
        )
    if not entries:
        raise ArtifactDenied(f"{root} holds no files, so there is no artifact to record")
    return tuple(sorted(entries, key=lambda item: item.name))


def manifest_bytes(files: tuple[ArtifactFile, ...]) -> bytes:
    """The canonical listing: one compact JSON object per file, sorted by name."""
    lines = [
        json.dumps(
            {"name": item.name, "sha256": item.sha256, "size": item.size},
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in sorted(files, key=lambda entry: entry.name)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def artifact_digest(files: tuple[ArtifactFile, ...]) -> str:
    """The artifact's identity: the hash of its manifest, not of one file."""
    return hashlib.sha256(manifest_bytes(files)).hexdigest()


def _file_prefix(prefix: str, digest: str, name: str) -> str:
    parent = PurePosixPath(name).parent
    stem = f"{prefix.strip('/')}/{digest}"
    return stem if str(parent) in {"", "."} else f"{stem}/{parent.as_posix()}"


def file_key(prefix: str, digest: str, name: str, sha256: str) -> str:
    """Where one file of an artifact lives, derived rather than remembered."""
    return f"{_file_prefix(prefix, digest, name)}/{sha256}"


def store_artifact(
    blobs: BlobStore,
    path: Path,
    *,
    prefix: str = DEFAULT_PREFIX,
) -> Artifact:
    files = collect(path)
    digest = artifact_digest(files)
    root = Path(path)
    for item in files:
        payload = (root / item.name).read_bytes() if root.is_dir() else root.read_bytes()
        blobs.put(payload, prefix=_file_prefix(prefix, digest, item.name))
    manifest = blobs.put(manifest_bytes(files), prefix=f"{prefix}/{digest}")
    return Artifact(
        manifest_key=manifest.key,
        digest=digest,
        files=files,
        size=sum(item.size for item in files),
        prefix=prefix,
    )


def read_manifest(blobs: BlobStore, key: str) -> tuple[ArtifactFile, ...]:
    payload = blobs.get(key)
    entries = [
        ArtifactFile(name=row["name"], sha256=row["sha256"], size=int(row["size"]))
        for row in (json.loads(line) for line in payload.decode("utf-8").splitlines() if line)
    ]
    if artifact_digest(tuple(entries)) != key.rsplit("/", 1)[-1]:
        raise StorageDenied(f"manifest {key} does not describe itself; the listing was altered")
    return tuple(entries)


def materialize(blobs: BlobStore, artifact: Artifact, target: Path) -> list[Path]:
    """Rebuild the artifact into ``target``, refusing any file that does not match.

    A rebuilt copy is the whole point of storing the bytes: an install that has
    the manifest can serve the model without the directory the training job ran
    in. Each file is fetched by content, so a mismatch is an error here rather
    than a model that quietly loads different weights.
    """
    destination = Path(target)
    written: list[Path] = []
    for item in artifact.files:
        key = file_key(artifact.prefix, artifact.digest, item.name, item.sha256)
        payload = blobs.get(key, expect_sha256=item.sha256)
        path = destination / item.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        written.append(path)
    return written


def verify(path: Path, files: tuple[ArtifactFile, ...]) -> list[str]:
    """Names whose bytes no longer match the manifest, in manifest order."""
    root = Path(path)
    mismatches: list[str] = []
    for item in files:
        candidate = root / item.name if root.is_dir() else root
        try:
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            mismatches.append(item.name)
            continue
        if actual != item.sha256:
            mismatches.append(item.name)
    return mismatches


def missing(path: Path, files: tuple[ArtifactFile, ...]) -> list[str]:
    """Names the manifest lists that are not on disk."""
    root = Path(path)
    return [item.name for item in files if not (root / item.name).exists()]


def has_manifest(card: dict[str, object]) -> bool:
    artifact = card.get("artifact")
    return isinstance(artifact, dict) and bool(artifact.get("manifest"))


def artifact_from_card(card: dict[str, object]) -> Artifact | None:
    """The artifact a model card records, or None for a card written before this existed."""
    entry = card.get("artifact")
    if not isinstance(entry, dict) or not entry.get("manifest"):
        return None
    manifest = str(entry["manifest"])
    digest = str(entry.get("digest") or manifest.rsplit("/", 1)[-1])
    return Artifact(
        manifest_key=manifest,
        digest=digest,
        files=(),
        size=int(entry.get("size") or 0),
    )


def read_card_artifact(blobs: BlobStore, card: dict[str, object]) -> Artifact | None:
    artifact = artifact_from_card(card)
    if artifact is None:
        return None
    files = read_manifest(blobs, artifact.manifest_key)
    return Artifact(
        manifest_key=artifact.manifest_key,
        digest=artifact.digest,
        files=files,
        size=sum(item.size for item in files),
    )
