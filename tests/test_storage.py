"""Released bytes are addressed by their own hash, and an altered object is caught.

The interesting failures are the quiet ones: a key that names nothing, a key that
climbs out of the store, an object whose bytes changed after it was written, and
an enterprise backend that silently degrades to local disk. Each has a test here.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from internal_ai import storage
from internal_ai.storage import (
    BlobMissing,
    LocalBlobStore,
    S3BlobStore,
    StorageDenied,
    blob_store,
    content_key,
    key_digest,
)


class NotFound(Exception):
    """What a fake S3 client raises for an absent object."""

    response: ClassVar[dict[str, dict[str, str]]] = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.puts: list[dict[str, object]] = []

    def put_object(self, Bucket: str, Key: str, Body: bytes, **extra: object) -> None:
        self.objects[(Bucket, Key)] = Body
        self.puts.append(extra)

    def get_object(self, Bucket: str, Key: str) -> dict[str, object]:
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError:
            raise NotFound() from None
        return {"Body": io.BytesIO(body)}

    def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
        if (Bucket, Key) not in self.objects:
            raise NotFound()
        return {}


def test_the_key_is_the_digest_of_the_bytes() -> None:
    payload = b"one game per line\n"
    key = content_key(payload)
    assert len(key) == 64
    assert content_key(payload) == key
    assert content_key(payload, "datasets/nba") == f"datasets/nba/{key}"
    assert key_digest(f"datasets/nba/{key}") == key


def test_a_round_trip_returns_the_same_bytes(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    blob = store.put(b"payload", prefix="eval")
    assert blob.key.startswith("eval/")
    assert blob.sha256 == key_digest(blob.key)
    assert store.get(blob.key) == b"payload"
    assert store.exists(blob.key)
    assert store.verify(blob.key)


def test_writing_the_same_bytes_twice_is_one_object(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    first = store.put(b"same", prefix="releases")
    second = store.put(b"same", prefix="releases")
    assert first.key == second.key
    assert len(list((tmp_path / "releases").iterdir())) == 1


def test_an_altered_object_is_refused_rather_than_served(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    blob = store.put(b"truth")
    path = tmp_path / blob.key
    path.chmod(0o644)
    path.write_bytes(b"lie")
    with pytest.raises(StorageDenied, match="altered"):
        store.get(blob.key)
    assert store.verify(blob.key) is False


def test_a_caller_expectation_that_does_not_match_is_refused(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    blob = store.put(b"payload")
    with pytest.raises(StorageDenied):
        store.get(blob.key, expect_sha256="0" * 64)


def test_a_key_that_climbs_out_of_the_root_is_refused(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    for key in ("../escape", "/absolute", "a//b", "a/./b", "a\\b", ""):
        with pytest.raises(StorageDenied):
            store.get(key)


def test_a_missing_key_is_a_miss_not_an_empty_object(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    with pytest.raises(BlobMissing):
        store.get("0" * 64)
    assert store.exists("0" * 64) is False


def test_the_factory_refuses_an_unknown_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(storage.ENV_BACKEND, "ftp")
    with pytest.raises(StorageDenied, match="unknown blob backend"):
        blob_store(tmp_path)


def test_the_factory_refuses_s3_without_a_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(storage.ENV_BACKEND, "s3")
    monkeypatch.delenv(storage.ENV_BUCKET, raising=False)
    with pytest.raises(StorageDenied, match="BUCKET"):
        blob_store(tmp_path)


def test_the_s3_backend_refuses_when_boto3_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    store = S3BlobStore("released")
    with pytest.raises(StorageDenied, match="boto3"):
        _ = store.client


def test_the_s3_backend_writes_content_addressed_objects_and_retains_them() -> None:
    fake = FakeS3()
    store = S3BlobStore("released", prefix="releases", retention_days=30, client=fake)
    blob = store.put(b"adapter", prefix="models")
    assert blob.key.startswith("models/")
    assert fake.objects[("released", f"releases/{blob.key}")] == b"adapter"
    assert fake.puts[0]["ObjectLockMode"] == "COMPLIANCE"
    assert "ObjectLockRetainUntilDate" in fake.puts[0]
    assert store.get(blob.key) == b"adapter"
    assert store.exists(blob.key)
    assert store.verify(blob.key)


def test_the_s3_backend_reports_a_missing_object_as_a_miss() -> None:
    store = S3BlobStore("released", client=FakeS3())
    with pytest.raises(BlobMissing):
        store.get("0" * 64)
    assert store.exists("0" * 64) is False


def test_the_s3_backend_refuses_to_reuse_a_key_for_different_bytes() -> None:
    fake = FakeS3()
    store = S3BlobStore("released", client=fake)
    blob = store.put(b"first")
    fake.objects[("released", blob.key)] = b"second"
    with pytest.raises(StorageDenied):
        store.put(b"first")
