"""Release refuses leftover copies, and dropping them builds a new version."""

from __future__ import annotations

from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.samples import generate_corpus
from internal_ai.ids import new_id
from internal_ai.workflows import build_corpus


def _corpus(platform, records: int, name: str) -> str:
    path = platform.settings.root / "data" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_corpus(path, records=records, duplicate_ratio=0, pii_ratio=0, seed=4)
    return build_corpus(platform, [path], name=name).dataset_id


def test_identical_records_fail_release_and_a_new_version_drops_them(platform) -> None:
    dataset_id = _corpus(platform, 8, "exact-copies")
    original = platform.store.get_dataset(dataset_id)
    row = platform.store.dataset_records(dataset_id)[0]
    platform.store.conn.execute(
        "INSERT INTO dataset_records (id, dataset_id, record_key, payload, source_id,"
        " sensitivity, acl, tokens, split, meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            new_id("rec"),
            dataset_id,
            row["record_key"] + "-copy",
            row["payload"],
            row["source_id"],
            row["sensitivity"],
            "[]",
            row["tokens"],
            "",
            "{}",
        ),
    )
    platform.store.conn.execute(
        "UPDATE datasets SET record_count = record_count + 1 WHERE id = ?",
        (dataset_id,),
    )
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    try:
        pipeline.release(dataset_id, "ml_lead", gate_approved=True)
        raise AssertionError("identical text was released")
    except PermissionError as exc:
        assert "1 exact duplicate" in str(exc)
        assert row["record_key"] in str(exc)
    stored = platform.store.get_dataset(dataset_id)
    assert stored.released is False
    assert stored.record_count == original.record_count + 1
    created = pipeline.drop_duplicate_copies(dataset_id, "data_engineer")
    assert created.id != dataset_id
    assert created.released is False
    assert created.version == original.version + 1
    assert platform.store.get_dataset(dataset_id).record_count == stored.record_count
    assert platform.store.get_dataset(dataset_id).released is False


def test_a_corpus_under_the_residual_cap_records_the_counts(platform) -> None:
    dataset_id = _corpus(platform, 120, "near-copies")
    rows = platform.store.dataset_records(dataset_id)
    assert len(rows) >= 101
    platform.store.conn.execute(
        "UPDATE dataset_records SET payload = ? WHERE id = ?",
        (rows[0]["payload"] + "z", rows[1]["id"]),
    )
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    released = pipeline.release(dataset_id, "ml_lead", gate_approved=True)
    assert released.stats["exact_duplicate_groups"] == 0
    assert released.stats["near_duplicate_pairs"] >= 1
    assert released.stats["residual_rate"] <= 0.01
    assert released.released is True
