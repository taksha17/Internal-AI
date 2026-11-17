"""A released version leaves the database as one immutable object.

Rows in the control-plane database are this install's index. The shard is what
another machine reads: canonical JSONL, sorted by record key, with a fixed field
set, so the same records always hash the same no matter how they were inserted
or in what order they were read.

The key is that hash. A reader therefore checks the shard it got against the
name it asked for, which is the only way a released version can be shown to still
be the version that was released.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from internal_ai.storage import BlobStore, StorageDenied

if TYPE_CHECKING:
    from internal_ai.store import Store

CANONICAL_FIELDS = (
    "record_key",
    "payload",
    "source_id",
    "sensitivity",
    "acl",
    "tokens",
    "split",
    "meta",
)

SHARD_PREFIX = "datasets"


@dataclass(frozen=True, slots=True)
class Shard:
    """What a release wrote: where it lives, what it hashes to, what it holds."""

    key: str
    sha256: str
    records: int
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "records": self.records,
            "size": self.size,
        }


def canonical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """One row in the shape a shard carries: no ids that belong to this install.

    ``id`` and ``dataset_id`` are database-local; two installs holding the same
    corpus would otherwise hash differently for reasons that are not the corpus.
    """
    row = {field: record.get(field) for field in CANONICAL_FIELDS}
    row["acl"] = list(row["acl"] or [])
    row["meta"] = dict(row["meta"] or {})
    return row


def canonical_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    """The bytes of a shard: one compact JSON object per line, sorted by key."""
    lines = [
        json.dumps(
            canonical_record(record),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in sorted(records, key=lambda row: str(row["record_key"]))
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def write_shard(
    blobs: BlobStore,
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    prefix: str = SHARD_PREFIX,
) -> Shard:
    rows = list(records)
    payload = canonical_bytes(rows)
    blob = blobs.put(payload, prefix=f"{prefix}/{dataset_id}")
    return Shard(key=blob.key, sha256=blob.sha256, records=len(rows), size=blob.size)


def read_shard(
    blobs: BlobStore,
    key: str,
    *,
    expect_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Read a shard back, refusing bytes that no longer match the key they came from."""
    payload = blobs.get(key, expect_sha256=expect_sha256)
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def released_rows(
    store: Store,
    blobs: BlobStore | None,
    dataset_id: str,
) -> list[dict[str, Any]]:
    """The rows of a released version, read from the object that names them.

    A version released before shards existed has none recorded and still reads
    from the database. A version that has one is read from it, because the
    database can be edited afterwards and the object cannot: rows in the table
    are this install's index, not the released artifact. A recorded shard with no
    object store configured is a refusal, not a quiet fall back to whatever the
    table says now.
    """
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        # A deployment can name a version this install never built (an imported
        # model card, a second serving replica). That reads as no documents, the
        # same answer the table gave before shards existed, so the caller's
        # "nothing to cite" path is unchanged.
        return store.dataset_records(dataset_id)
    shard = (dataset.stats or {}).get("shard") or {}
    key = shard.get("key")
    if not key:
        return store.dataset_records(dataset_id)
    if blobs is None:
        raise StorageDenied(
            f"dataset {dataset_id} was released as shard {key} and no object store "
            "is configured to read it back"
        )
    return read_shard(blobs, str(key), expect_sha256=shard.get("sha256"))
