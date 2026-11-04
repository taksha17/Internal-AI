"""A new dataset version must not take the old version's rows with it.

Released versions are immutable, and the store is where that is enforced. Record
ids are derived from the record, so the same corpus rebuilt as version 2 offers
the same ids: with a primary key on the id alone, the insert replaced version 1's
rows and repointed them at version 2, and a released dataset read as empty while
still claiming its record count.
"""

from __future__ import annotations

from internal_ai.workflows import build_corpus, split_corpus


def test_an_install_written_before_the_key_change_keeps_its_rows(tmp_path) -> None:
    """The migration rebuilds the table; an existing install must not lose data."""
    import sqlite3

    from internal_ai.db import init_db

    conn = sqlite3.connect(tmp_path / "old.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            released INTEGER NOT NULL DEFAULT 0,
            parent_version TEXT,
            lineage TEXT NOT NULL DEFAULT '{}',
            stats TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE dataset_records (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(id),
            record_key TEXT NOT NULL,
            payload TEXT NOT NULL,
            source_id TEXT NOT NULL,
            sensitivity TEXT NOT NULL DEFAULT 'internal',
            acl TEXT NOT NULL DEFAULT '[]',
            tokens INTEGER NOT NULL DEFAULT 0,
            split TEXT NOT NULL DEFAULT '',
            meta TEXT NOT NULL DEFAULT '{}'
        );
        INSERT INTO datasets VALUES
            ('ds_old','rfc',1,'hash-old',1,4,'2026-01-01',1,NULL,'{}','{}');
        INSERT INTO dataset_records
            (id, dataset_id, record_key, payload, source_id, tokens, split)
            VALUES ('rec-1','ds_old','rfc1.txt:1','the ledger feed failed','rfc',4,'train');
        """
    )

    init_db(conn)

    rows = conn.execute("SELECT * FROM dataset_records").fetchall()
    assert [row["payload"] for row in rows] == ["the ledger feed failed"]
    keys = sorted(
        (row["pk"], row["name"])
        for row in conn.execute("PRAGMA table_info(dataset_records)")
        if row["pk"]
    )
    assert keys == [(1, "dataset_id"), (2, "id")]
    index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_dataset_records_dataset'"
    ).fetchone()
    assert index is not None
    conn.close()


def test_a_second_version_leaves_the_first_versions_records_alone(platform, corpus_path) -> None:
    first = build_corpus(platform, [corpus_path], name="corpus-test", version=1)
    split_corpus(platform, first.dataset_id)
    before = platform.store.dataset_records(first.dataset_id)
    before_splits = {row["record_key"]: row["split"] for row in before}
    assert before

    second = build_corpus(platform, [corpus_path], name="corpus-test", version=2)
    split_corpus(platform, second.dataset_id, document_holdout=True)

    after = platform.store.dataset_records(first.dataset_id)
    assert [row["record_key"] for row in after] == [row["record_key"] for row in before]
    assert [row["payload"] for row in after] == [row["payload"] for row in before]
    assert {row["record_key"]: row["split"] for row in after} == before_splits
    assert platform.store.dataset_records(second.dataset_id)


def test_the_same_record_key_can_exist_in_two_versions(platform, corpus_path) -> None:
    first = build_corpus(platform, [corpus_path], name="corpus-test", version=1)
    second = build_corpus(platform, [corpus_path], name="corpus-test", version=2)

    keys = {row["record_key"] for row in platform.store.dataset_records(first.dataset_id)}
    other = {row["record_key"] for row in platform.store.dataset_records(second.dataset_id)}

    assert keys == other
    assert platform.store.dataset_records(first.dataset_id) is not platform.store.dataset_records(
        second.dataset_id
    )
