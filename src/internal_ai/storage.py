"""Where released bytes live, and why the key is the digest of what it holds.

A dataset version, a frozen eval set and a model artifact are all "the bytes
somebody can read again later". Locally they sit under ``var/``; in an enterprise
install they sit in S3 or MinIO, and the rest of the platform must not care
which. Same key, same hash, same refusal when the two disagree.

Keys are content addresses. Writing the same bytes twice is a no-op instead of a
second version, and reading a key whose content no longer matches its name is an
error instead of a silent acceptance -- that check is the reason the name is a
hash rather than a timestamp.

A dataset shard may store those bytes as JSONL or as Parquet. The key is the
digest of the stored bytes, so the two encodings do not share a key. The
logical content hash, the one a version's lineage is about, is computed from
the canonical rows in ``data/shards.py`` and is the same for both.

This module decides nothing about *who* may read a bucket. Access policy stays
in ``services/permissions.py`` and ``gates.py``; a caller that has not been
checked must not reach this module at all.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ENV_BACKEND = "INTERNAL_AI_BLOB_BACKEND"
ENV_BUCKET = "INTERNAL_AI_BLOB_BUCKET"
ENV_ENDPOINT = "INTERNAL_AI_BLOB_ENDPOINT"
ENV_PREFIX = "INTERNAL_AI_BLOB_PREFIX"
ENV_RETENTION_DAYS = "INTERNAL_AI_BLOB_RETENTION_DAYS"

BACKENDS = ("local", "s3")


class StorageDenied(RuntimeError):
    """The store refused instead of writing or serving bytes it cannot vouch for."""


class BlobMissing(KeyError):
    """No object under that key, so nothing was read."""


@dataclass(frozen=True, slots=True)
class Blob:
    """What a write produced: the key, and what the bytes were."""

    key: str
    size: int
    sha256: str
    backend: str


def digest_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_key(payload: bytes, prefix: str = "") -> str:
    """The key these bytes live under: their sha256, optionally namespaced.

    The digest is always the last path segment, so a reader can check content
    against the name without knowing how the caller namespaced it.
    """
    digest = digest_of(payload)
    cleaned = prefix.strip("/")
    return f"{cleaned}/{digest}" if cleaned else digest


def key_digest(key: str) -> str:
    """The digest a key claims, taken from its last segment."""
    return key.rsplit("/", 1)[-1]


def check_key(key: str) -> str:
    """Refuse a key that could name something outside the store.

    A content address is a hex digest under slash-separated namespaces, so
    anything else is either a caller bug or an attempt to escape the root.
    """
    if not key or key != key.strip():
        raise StorageDenied("a key is non-empty and carries no surrounding space")
    if key.startswith("/") or "\\" in key or "\x00" in key:
        raise StorageDenied(f"key {key!r} is not a relative object key")
    for part in key.split("/"):
        if part in {"", ".", ".."}:
            raise StorageDenied(f"key {key!r} contains an empty, current or parent segment")
    return key


class BlobStore(Protocol):
    """The operations the platform needs, and nothing else."""

    name: str

    def put(
        self,
        payload: bytes,
        *,
        prefix: str = "",
        retention_days: int | None = None,
    ) -> Blob: ...

    def get(self, key: str, *, expect_sha256: str | None = None) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def verify(self, key: str) -> bool: ...

    def retention_until(self, key: str) -> str | None: ...


class LocalBlobStore:
    """Content-addressed files under one directory, read-only once written.

    The read-only bit is not security -- a local root can clear it -- it is a
    tripwire: a second write to a live key fails loudly instead of quietly
    changing what an existing version holds.
    """

    name = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        checked = check_key(key)
        return self.root / checked

    def put(
        self,
        payload: bytes,
        *,
        prefix: str = "",
        retention_days: int | None = None,
    ) -> Blob:
        key = content_key(payload, prefix)
        target = self._path(key)
        digest = digest_of(payload)
        if target.exists():
            self._require_same_bytes(target, digest, key)
            self._record_retention(key, retention_days)
            return Blob(key=key, size=len(payload), sha256=digest, backend=self.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".put-")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._record_retention(key, retention_days)
        return Blob(key=key, size=len(payload), sha256=digest, backend=self.name)

    def retention_until(self, key: str) -> str | None:
        """The compliance retain-until date recorded for this key, if one was."""
        path = self._lock_path(key)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def _lock_path(self, key: str) -> Path:
        return self.root / ".locks" / check_key(key)

    def _record_retention(self, key: str, retention_days: int | None) -> None:
        if not retention_days:
            return
        path = self._lock_path(key)
        if path.exists():
            return
        from datetime import UTC, datetime, timedelta

        until = (datetime.now(UTC) + timedelta(days=int(retention_days))).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(until + "\n", encoding="utf-8")
        os.chmod(path, 0o444)

    def get(self, key: str, *, expect_sha256: str | None = None) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise BlobMissing(key)
        payload = path.read_bytes()
        self._require_digest(payload, key, expect_sha256)
        return payload

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def verify(self, key: str) -> bool:
        try:
            self.get(key)
        except (BlobMissing, StorageDenied):
            return False
        return True

    def _require_same_bytes(self, path: Path, digest: str, key: str) -> None:
        if digest_of(path.read_bytes()) != digest:
            raise StorageDenied(
                f"{key} already holds different bytes; a content address is never reused"
            )

    @staticmethod
    def _require_digest(payload: bytes, key: str, expect_sha256: str | None) -> None:
        claimed = key_digest(key) if expect_sha256 is None else expect_sha256
        if digest_of(payload) != claimed:
            raise StorageDenied(
                f"bytes under {key} do not hash to {claimed}; "
                "the object was altered after it was written"
            )


class S3BlobStore:
    """S3 or MinIO. Refuses when boto3 is absent instead of degrading to local.

    A silent fallback to the local disk would put an "enterprise" install's
    released datasets on the machine that happened to run the job, which is the
    failure this backend exists to prevent. Optional object-lock retention makes
    a written version immutable on the bucket side as well as in code.
    """

    name = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        prefix: str = "",
        retention_days: int | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise StorageDenied("an S3 backend needs a bucket")
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.prefix = prefix.strip("/")
        self.retention_days = retention_days
        self._client = client
        self._locked: dict[str, str] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            import boto3  # optional: only an install that chose s3 pays for it
        except ImportError as exc:
            raise StorageDenied(
                "the S3 backend needs boto3; install it rather than falling back to local disk"
            ) from exc
        return boto3.client("s3", endpoint_url=self.endpoint_url)

    def _object_key(self, key: str) -> str:
        checked = check_key(key)
        return f"{self.prefix}/{checked}" if self.prefix else checked

    def put(
        self,
        payload: bytes,
        *,
        prefix: str = "",
        retention_days: int | None = None,
    ) -> Blob:
        namespaced = content_key(payload, prefix)
        object_key = self._object_key(namespaced)
        digest = digest_of(payload)
        if self.exists(namespaced):
            existing = self.get(namespaced)
            if digest_of(existing) != digest:
                raise StorageDenied(
                    f"{namespaced} already holds different bytes; a content address is never reused"
                )
            return Blob(key=namespaced, size=len(payload), sha256=digest, backend=self.name)
        extra: dict[str, Any] = {}
        days = self.retention_days if retention_days is None else retention_days
        if days:
            from datetime import UTC, datetime, timedelta

            until = datetime.now(UTC) + timedelta(days=days)
            extra["ObjectLockMode"] = "COMPLIANCE"
            extra["ObjectLockRetainUntilDate"] = until
            self._locked.setdefault(namespaced, until.isoformat())
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=payload, **extra)
        return Blob(key=namespaced, size=len(payload), sha256=digest, backend=self.name)

    def retention_until(self, key: str) -> str | None:
        return self._locked.get(key)

    def get(self, key: str, *, expect_sha256: str | None = None) -> bytes:
        object_key = self._object_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:
            # A 404 is a miss; every other failure (auth, network, permissions)
            # is re-raised, because calling it absent would hide a real fault.
            if self._is_missing(exc):
                raise BlobMissing(key) from exc
            raise
        payload = response["Body"].read()
        LocalBlobStore._require_digest(payload, key, expect_sha256)
        return payload

    def exists(self, key: str) -> bool:
        object_key = self._object_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:
            if self._is_missing(exc):
                return False
            raise
        return True

    def verify(self, key: str) -> bool:
        try:
            self.get(key)
        except (BlobMissing, StorageDenied):
            return False
        return True

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        response = getattr(exc, "response", None) or {}
        code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        return code in {"404", "NoSuchKey", "NotFound"}


def blob_store(root: Path) -> BlobStore:
    """The store this install asked for; a bad setting is a refusal, not a default.

    ``INTERNAL_AI_BLOB_BACKEND=local`` keeps every byte on this machine, which is
    the from-scratch install and the air-gapped one. ``s3`` needs a bucket, and
    an unrecognised value stops the install rather than quietly writing locally.
    """
    backend = os.environ.get(ENV_BACKEND, "local").strip().lower() or "local"
    if backend == "local":
        return LocalBlobStore(root)
    if backend != "s3":
        raise StorageDenied(f"unknown blob backend {backend!r}; expected one of {list(BACKENDS)}")
    bucket = os.environ.get(ENV_BUCKET, "").strip()
    if not bucket:
        raise StorageDenied(f"{ENV_BUCKET} is required when the backend is s3")
    days = os.environ.get(ENV_RETENTION_DAYS, "").strip()
    return S3BlobStore(
        bucket,
        endpoint_url=os.environ.get(ENV_ENDPOINT) or None,
        prefix=os.environ.get(ENV_PREFIX, ""),
        retention_days=int(days) if days else None,
    )


def disk_free(root: Path) -> int:
    """Bytes available where a local store writes; a caller decides whether that is enough."""
    return shutil.disk_usage(root).free
