"""A released version is one object, named by its own hash.

The shard is what a second machine reads, so two properties matter: the same
records always produce the same bytes, and a shard whose bytes changed is refused
rather than parsed. Both are tested here, together with the release that writes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import enroll

from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.shards import (
    SHARD_SCHEMA,
    canonical_bytes,
    read_shard,
    released_rows,
    schema_digest,
    write_shard,
)
from internal_ai.domain import DatasetVersion
from internal_ai.feedback import export_feedback, file_feedback
from internal_ai.ids import now_iso
from internal_ai.services.build import release_dataset
from internal_ai.services.permissions import PermissionDenied
from internal_ai.serving.service import RetrievalService
from internal_ai.storage import LocalBlobStore, StorageDenied, digest_of


def _records() -> list[dict[str, object]]:
    return [
        {
            "id": "rec_2",
            "dataset_id": "ds_1",
            "record_key": "b.csv:2",
            "payload": "second line",
            "source_id": "b",
            "sensitivity": "internal",
            "acl": ["engineering"],
            "tokens": 3,
            "split": "train",
            "meta": {"line": 2},
        },
        {
            "id": "rec_1",
            "dataset_id": "ds_1",
            "record_key": "a.csv:1",
            "payload": "first line",
            "source_id": "a",
            "sensitivity": "internal",
            "acl": [],
            "tokens": 2,
            "split": "eval",
            "meta": {},
        },
    ]


def test_canonical_bytes_do_not_depend_on_insert_order() -> None:
    records = _records()
    assert canonical_bytes(records) == canonical_bytes(list(reversed(records)))
    assert canonical_bytes(records).endswith(b"\n")


def test_database_ids_do_not_change_the_hash() -> None:
    """Two installs holding the same corpus must produce the same shard."""
    first = _records()
    second = [dict(record, id=f"other-{index}") for index, record in enumerate(_records())]
    assert canonical_bytes(first) == canonical_bytes(second)


def test_a_shard_is_named_by_its_own_hash(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    shard = write_shard(store, _records(), dataset_id="ds_1")

    assert shard.key.startswith("datasets/ds_1/")
    assert shard.key.endswith(shard.sha256)
    assert shard.records == 2
    assert store.exists(shard.key)


def test_reading_a_shard_returns_the_records_in_key_order(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    shard = write_shard(store, _records(), dataset_id="ds_1")

    rows = read_shard(store, shard.key)

    assert [row["record_key"] for row in rows] == ["a.csv:1", "b.csv:2"]
    assert rows[0]["payload"] == "first line"
    assert "dataset_id" not in rows[0]


def test_an_altered_shard_is_refused(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    shard = write_shard(store, _records(), dataset_id="ds_1")
    path = tmp_path / shard.key
    path.chmod(0o644)
    path.write_bytes(b'{"record_key":"a.csv:1","payload":"forged"}\n')

    with pytest.raises(StorageDenied):
        read_shard(store, shard.key)


def test_a_release_writes_a_shard_and_records_it_on_the_version(platform, tmp_path) -> None:
    owner = enroll(platform, "Sam Owner", ["data_owner"], ["engineering"])
    file_feedback(platform, owner, "where is the host", "src/app.py", 4, "ds_ask")
    dataset = export_feedback(platform, owner)
    blobs = LocalBlobStore(tmp_path / "objects")

    released = DataPipeline(platform.store, platform.audit, platform.settings.root, blobs).release(
        dataset.id, owner.id, gate_approved=True
    )

    shard = released.stats["shard"]
    assert shard["records"] == released.record_count
    rows = read_shard(blobs, shard["key"])
    assert len(rows) == released.record_count
    assert all(row["payload"] for row in rows)

    events = platform.store.list_audit(action="dataset.released", limit=5)
    assert events[-1].details["shard"] == shard["key"]


def _released_export(platform, blobs: LocalBlobStore):
    owner = enroll(platform, "Sam Owner", ["data_owner"], ["engineering"])
    file_feedback(platform, owner, "where is the host", "src/app.py", 4, "ds_ask")
    dataset = export_feedback(platform, owner)
    released = DataPipeline(platform.store, platform.audit, platform.settings.root, blobs).release(
        dataset.id, owner.id, gate_approved=True
    )
    return released


def test_a_released_version_is_read_from_its_shard_not_from_the_table(platform, tmp_path) -> None:
    """The table is an index. Editing a row afterwards must not change what was released."""
    blobs = LocalBlobStore(tmp_path / "objects")
    released = _released_export(platform, blobs)
    released_rows(platform.store, blobs, released.id)

    platform.store.conn.execute(
        "UPDATE dataset_records SET payload = ? WHERE dataset_id = ?",
        ("forged line", released.id),
    )
    rows = released_rows(platform.store, blobs, released.id)

    assert rows
    assert all(row["payload"] != "forged line" for row in rows)


def test_a_recorded_shard_without_a_store_is_refused(platform, tmp_path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")
    released = _released_export(platform, blobs)

    with pytest.raises(StorageDenied, match="no object store"):
        released_rows(platform.store, None, released.id)


def test_a_version_with_no_shard_still_reads_from_the_table(platform) -> None:
    """Versions released before shards existed keep reading where they always did."""
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_old",
            name="older",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=4,
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(
        "ds_old",
        [{"record_key": "a.csv:1", "payload": "from the table", "acl": []}],
    )

    rows = released_rows(platform.store, None, "ds_old")

    assert [row["payload"] for row in rows] == ["from the table"]


def test_an_unknown_dataset_reads_as_no_documents(platform) -> None:
    """A deployment may name a version this install never built; that is not an error."""
    assert released_rows(platform.store, None, "ds_never_built") == []


def test_retrieval_reads_the_released_rows(platform, tmp_path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")
    released = _released_export(platform, blobs)
    platform.store.conn.execute(
        "UPDATE dataset_records SET payload = ? WHERE dataset_id = ?",
        ("forged line", released.id),
    )

    corpus = RetrievalService(platform.store, platform.audit, blobs).corpus(released.id)

    assert corpus
    assert all(row["payload"] != "forged line" for row in corpus)


def test_a_pipeline_without_a_store_writes_no_shard(platform) -> None:
    owner = enroll(platform, "Sam Owner", ["data_owner"], ["engineering"])
    file_feedback(platform, owner, "where is the host", "src/app.py", 4, "ds_ask")
    dataset = export_feedback(platform, owner)

    released = DataPipeline(platform.store, platform.audit, platform.settings.root).release(
        dataset.id, owner.id, gate_approved=True
    )

    assert "shard" not in released.stats


def test_parquet_and_jsonl_name_the_same_content(tmp_path: Path) -> None:
    """The object bytes differ. The content hash is the canonical rows, so lineage does not fork."""
    store = LocalBlobStore(tmp_path)
    rows = _records()
    jsonl = write_shard(store, rows, dataset_id="ds_1", encoding="jsonl")
    parquet = write_shard(store, list(reversed(rows)), dataset_id="ds_1", encoding="parquet")

    assert jsonl.content_sha256 == digest_of(canonical_bytes(rows))
    assert parquet.content_sha256 == jsonl.content_sha256
    assert parquet.sha256 != jsonl.sha256
    assert parquet.key.endswith(parquet.sha256)
    assert parquet.encoding == "parquet"
    assert parquet.schema_sha256 == schema_digest(SHARD_SCHEMA)
    assert read_shard(store, parquet.key) == read_shard(store, jsonl.key)

    again = write_shard(store, rows, dataset_id="ds_1", encoding="parquet")
    assert again.sha256 == parquet.sha256


def test_a_parquet_release_is_read_back_as_the_canonical_rows(
    platform, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("INTERNAL_AI_SHARD_ENCODING", "parquet")
    blobs = LocalBlobStore(tmp_path / "objects")
    released = _released_export(platform, blobs)

    shard = released.stats["shard"]
    assert shard["encoding"] == "parquet"
    assert shard["schema"] == SHARD_SCHEMA
    rows = released_rows(platform.store, blobs, released.id)
    assert shard["content_sha256"] == digest_of(canonical_bytes(rows))
    assert rows == read_shard(blobs, shard["key"], expect_content_sha256=shard["content_sha256"])


def test_a_parquet_file_with_another_schema_is_refused(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    store = LocalBlobStore(tmp_path)
    table = pa.table({"record_key": ["a.csv:1"], "note": ["not this version"]})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="NONE")
    blob = store.put(sink.getvalue().to_pybytes(), prefix="datasets/ds_1")

    with pytest.raises(StorageDenied, match="schema"):
        read_shard(store, blob.key)


def test_parquet_bytes_that_do_not_reproduce_the_content_hash_are_refused(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    shard = write_shard(store, _records(), dataset_id="ds_1", encoding="parquet")

    with pytest.raises(StorageDenied, match="content"):
        read_shard(store, shard.key, expect_content_sha256="0" * 64)


def test_an_unknown_shard_encoding_is_refused(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)

    with pytest.raises(StorageDenied, match="encoding"):
        write_shard(store, _records(), dataset_id="ds_1", encoding="orc")


def test_finance_cannot_write_a_shard_by_releasing(platform) -> None:
    """Releasing is what writes the shard. A role that cannot release writes nothing."""
    finance = enroll(platform, "Pat Finance", ["finance"])

    with pytest.raises(PermissionDenied, match="release dataset"):
        release_dataset(platform, finance, "ds_absent")

    denied = platform.store.list_audit(action="console.denied")
    assert denied and str(denied[0].outcome) == "denied"
    assert denied[0].actor == finance.id


def test_the_dataset_page_shows_the_shard_encoding(tmp_path: Path) -> None:
    from conftest import sign_in
    from fastapi.testclient import TestClient

    from internal_ai.api import create_app
    from internal_ai.context import Platform

    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    actor = enroll(setup, "Ada Admin", ["program_lead"])
    content = "a" * 64
    setup.store.add_dataset(
        DatasetVersion(
            id="ds_shown",
            name="shown",
            version=1,
            content_hash=content,
            record_count=1,
            token_count=4,
            created_at=now_iso(),
            released=True,
            stats={
                "shard": {
                    "encoding": "parquet",
                    "content_sha256": content,
                    "schema": {"version": 1},
                    "records": 1,
                }
            },
        )
    )
    setup.close()
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        assert sign_in(client, actor.name).status_code == 303
        page = client.get("/console/datasets/ds_shown")

    assert page.status_code == 200
    assert "parquet" in page.text
    assert content[:16] in page.text
    assert "schema 1" in page.text
