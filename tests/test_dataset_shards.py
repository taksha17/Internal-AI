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
from internal_ai.data.shards import canonical_bytes, read_shard, write_shard
from internal_ai.feedback import export_feedback, file_feedback
from internal_ai.storage import LocalBlobStore, StorageDenied


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


def test_a_pipeline_without_a_store_writes_no_shard(platform) -> None:
    owner = enroll(platform, "Sam Owner", ["data_owner"], ["engineering"])
    file_feedback(platform, owner, "where is the host", "src/app.py", 4, "ds_ask")
    dataset = export_feedback(platform, owner)

    released = DataPipeline(platform.store, platform.audit, platform.settings.root).release(
        dataset.id, owner.id, gate_approved=True
    )

    assert "shard" not in released.stats
