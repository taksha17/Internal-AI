"""A released version leaves the database as one immutable object.

Rows in the control-plane database are this install's index. The shard is what
another machine reads: the same canonical rows, sorted by record key, with one
fixed schema, so the same records always hash the same no matter how they were
inserted or which encoding the install stored them in.

JSONL and Parquet are two encodings of that one row set. The object key is the
digest of the stored bytes, which differs between encodings. The content hash
is the digest of the canonical JSONL, and both encodings record it. A version
whose two encodings hashed differently would be two versions.

The key is the object digest. A reader checks the bytes against the name, then
checks the decoded rows against the content hash, which is the only way a
released version can be shown to still be the version that was released.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from internal_ai.services.tenancy import DEFAULT_TENANT
from internal_ai.storage import BlobStore, StorageDenied, digest_of

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
    "tenant",
)

SHARD_PREFIX = "datasets"
ENV_SHARD_ENCODING = "INTERNAL_AI_SHARD_ENCODING"
SHARD_ENCODINGS = ("jsonl", "parquet")

# One schema for every version written by this code. It is copied onto the
# version at release, so a later change here does not reinterpret an object
# that was released under the previous fields.
SHARD_SCHEMA: dict[str, Any] = {
    "id": "internal-ai.dataset-shard",
    "version": 1,
    "fields": [
        {"name": "record_key", "type": "string"},
        {"name": "payload", "type": "string"},
        {"name": "source_id", "type": "string"},
        {"name": "sensitivity", "type": "string"},
        {"name": "acl", "type": "list<string>"},
        {"name": "tokens", "type": "int64"},
        {"name": "split", "type": "string"},
        {"name": "meta", "type": "json"},
        {"name": "tenant", "type": "string"},
    ],
}

_ARROW = {
    "string": "string",
    "int64": "int64",
    "list<string>": "list<string>",
    "json": "string",
}


def schema_digest(schema: Mapping[str, Any] = SHARD_SCHEMA) -> str:
    """The schema a version claims, as a hash of the document it stored."""
    payload = json.dumps(dict(schema), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return digest_of(payload.encode("utf-8"))


def configured_encoding() -> str:
    """The encoding this install writes, before it is checked.

    An unreadable value is returned as written so a page can show it. Writing
    is what refuses it.
    """
    return os.environ.get(ENV_SHARD_ENCODING, "jsonl").strip().lower() or "jsonl"


def require_encoding(encoding: str) -> str:
    if encoding not in SHARD_ENCODINGS:
        raise StorageDenied(
            f"shard encoding {encoding!r} is not one of {', '.join(SHARD_ENCODINGS)}"
        )
    return encoding


@dataclass(frozen=True, slots=True)
class Shard:
    """What a release wrote: where it lives, what it hashes to, what it holds."""

    key: str
    sha256: str
    records: int
    size: int
    encoding: str = "jsonl"
    content_sha256: str = ""
    schema_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "records": self.records,
            "size": self.size,
            "encoding": self.encoding,
            "content_sha256": self.content_sha256,
            "schema_sha256": self.schema_sha256,
            "schema": SHARD_SCHEMA,
        }


def canonical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """One row in the shape a shard carries: no ids that belong to this install.

    ``id`` and ``dataset_id`` are database-local; two installs holding the same
    corpus would otherwise hash differently for reasons that are not the corpus.
    """
    row = {field: record.get(field) for field in CANONICAL_FIELDS}
    row["acl"] = list(row["acl"] or [])
    row["meta"] = dict(row["meta"] or {})
    # A shard written before tenancy carries no tenant; it is the install's own.
    row["tenant"] = str(row["tenant"] or DEFAULT_TENANT)
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
    encoding: str | None = None,
) -> Shard:
    chosen = require_encoding(configured_encoding() if encoding is None else encoding)
    rows = list(records)
    content = canonical_bytes(rows)
    payload = content if chosen == "jsonl" else _parquet_bytes(rows)
    blob = blobs.put(payload, prefix=f"{prefix}/{dataset_id}")
    return Shard(
        key=blob.key,
        sha256=blob.sha256,
        records=len(rows),
        size=blob.size,
        encoding=chosen,
        content_sha256=digest_of(content),
        schema_sha256=schema_digest(),
    )


def read_shard(
    blobs: BlobStore,
    key: str,
    *,
    expect_sha256: str | None = None,
    expect_content_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Read a shard back, refusing bytes that are not the version that was released.

    The object digest catches a file that was replaced. The content hash catches
    an encoding that does not decode to the canonical rows, including a Parquet
    file written under some other schema.
    """
    payload = blobs.get(key, expect_sha256=expect_sha256)
    rows = _decode(payload)
    if expect_content_sha256 and digest_of(canonical_bytes(rows)) != expect_content_sha256:
        raise StorageDenied(
            "shard content hash does not match the canonical rows; the encoding is not this version"
        )
    return rows


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
    return read_shard(
        blobs,
        str(key),
        expect_sha256=shard.get("sha256"),
        expect_content_sha256=shard.get("content_sha256") or None,
    )


def _decode(payload: bytes) -> list[dict[str, Any]]:
    if payload.startswith(b"PAR1"):
        return _rows_from_parquet(payload)
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise StorageDenied(
            "Parquet shards need the parquet extra: pip install 'internal-ai[parquet]'"
        ) from exc
    return pa, pq


def _arrow_type(pa: Any, kind: str) -> Any:
    if kind == "string":
        return pa.string()
    if kind == "int64":
        return pa.int64()
    if kind == "list<string>":
        return pa.list_(pa.string())
    raise StorageDenied(f"shard schema type {kind!r} is not one this version writes")


def _arrow_schema(pa: Any) -> Any:
    return pa.schema(
        [
            (field["name"], _arrow_type(pa, _ARROW[field["type"]]))
            for field in SHARD_SCHEMA["fields"]
        ]
    )


def _same_schema(found: Any, expected: Any) -> bool:
    if list(found.names) != list(expected.names):
        return False
    return all(left.type == right.type for left, right in zip(found, expected, strict=True))


def _parquet_column(name: str, kind: str, rows: list[dict[str, Any]]) -> list[Any]:
    if kind == "json":
        return [
            json.dumps(row["meta"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        ]
    if kind == "list<string>":
        values: list[Any] = []
        for row in rows:
            items = list(row["acl"])
            if any(not isinstance(item, str) for item in items):
                raise StorageDenied(
                    f"shard field {name} holds a non-string, which this schema refuses"
                )
            values.append(items)
        return values
    if kind == "int64":
        values = []
        for row in rows:
            value = row[name]
            if value is None:
                values.append(None)
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise StorageDenied(f"shard field {name} is not an integer")
            values.append(value)
        return values
    values = []
    for row in rows:
        value = row[name]
        if value is not None and not isinstance(value, str):
            raise StorageDenied(f"shard field {name} is not a string")
        values.append(value)
    return values


def _parquet_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    pa, pq = _pyarrow()
    rows = [
        canonical_record(record)
        for record in sorted(records, key=lambda row: str(row["record_key"]))
    ]
    columns = {
        field["name"]: _parquet_column(field["name"], field["type"], rows)
        for field in SHARD_SCHEMA["fields"]
    }
    table = pa.table(columns, schema=_arrow_schema(pa))
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="NONE",
        version="2.6",
        data_page_version="1.0",
        use_dictionary=False,
        write_statistics=False,
        write_page_index=False,
        row_group_size=max(len(rows), 1),
    )
    return sink.getvalue().to_pybytes()


def _rows_from_parquet(payload: bytes) -> list[dict[str, Any]]:
    pa, pq = _pyarrow()
    table = pq.read_table(pa.BufferReader(payload))
    expected = _arrow_schema(pa)
    if not _same_schema(table.schema, expected):
        raise StorageDenied(
            "shard schema is not "
            f"{SHARD_SCHEMA['id']} version {SHARD_SCHEMA['version']}; "
            f"found {list(table.schema.names)}"
        )
    data = table.to_pydict()
    rows: list[dict[str, Any]] = []
    for index in range(table.num_rows):
        meta_raw = data["meta"][index]
        tokens = data["tokens"][index]
        rows.append(
            {
                "record_key": data["record_key"][index],
                "payload": data["payload"][index],
                "source_id": data["source_id"][index],
                "sensitivity": data["sensitivity"][index],
                "acl": list(data["acl"][index] or []),
                "tokens": None if tokens is None else int(tokens),
                "split": data["split"][index],
                "meta": json.loads(meta_raw) if meta_raw else {},
                "tenant": data["tenant"][index],
            }
        )
    return rows
