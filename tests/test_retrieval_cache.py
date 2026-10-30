"""Serving embeds a released dataset once, not once per query (change 0.4)."""

from __future__ import annotations

import time

import numpy as np

from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.serving.service import Principal, RetrievalService

RECORDS = 1_000


class CountingEmbedder:
    """A cheap embedder that remembers how much of the corpus it has seen."""

    name = "counting"

    def __init__(self) -> None:
        self.document_batches = 0
        self.documents = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        if len(texts) > 1:
            self.document_batches += 1
            self.documents += len(texts)
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


def _released_dataset(platform, content_hash: str = "h1") -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_cache",
            name="cache",
            version=1,
            content_hash=content_hash,
            record_count=RECORDS,
            token_count=RECORDS * 4,
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(
        "ds_cache",
        [
            {
                "record_key": f"doc-{index:04d}",
                "payload": f"record {index} " + "note " * (index % 7),
                "source_id": "s",
                "acl": ["finance"] if index % 100 == 0 else [],
            }
            for index in range(RECORDS)
        ],
    )


def test_a_second_query_does_not_re_embed_the_corpus(platform) -> None:
    _released_dataset(platform)
    embedder = CountingEmbedder()
    service = RetrievalService(platform.store, platform.audit)
    support = Principal(user_id="alice", groups=frozenset({"support"}))

    service.search(embedder, "ds_cache", "record 12", support)
    service.search(embedder, "ds_cache", "record 34", support)

    assert embedder.documents == RECORDS
    assert embedder.document_batches == 1


def test_a_warm_cache_answers_well_under_a_hundred_milliseconds(platform) -> None:
    _released_dataset(platform)
    service = RetrievalService(platform.store, platform.audit)
    support = Principal(user_id="alice", groups=frozenset({"support"}))
    service.search(CountingEmbedder(), "ds_cache", "warm", support)

    started = time.monotonic()
    hits = service.search(CountingEmbedder(), "ds_cache", "record 34", support)
    elapsed = time.monotonic() - started

    assert hits
    assert elapsed < 0.1


def test_the_cache_never_widens_what_a_principal_may_read(platform) -> None:
    _released_dataset(platform)
    embedder = CountingEmbedder()
    service = RetrievalService(platform.store, platform.audit)
    admin = Principal(user_id="root", groups=frozenset({"platform-admins"}))
    contractor = Principal(user_id="carol", groups=frozenset({"contractor"}))

    everything = service.search(embedder, "ds_cache", "record", admin, top_k=RECORDS)
    allowed = service.search(embedder, "ds_cache", "record", contractor, top_k=RECORDS)

    assert any(hit.document_id.startswith("doc-000") for hit in everything)
    assert allowed
    assert not any(hit.document_id.endswith("00") for hit in allowed)
    assert embedder.documents == RECORDS


def test_a_new_dataset_version_re_embeds(platform) -> None:
    _released_dataset(platform)
    embedder = CountingEmbedder()
    service = RetrievalService(platform.store, platform.audit)
    support = Principal(user_id="alice", groups=frozenset({"support"}))
    service.search(embedder, "ds_cache", "record 12", support)

    platform.store.conn.execute("UPDATE datasets SET content_hash = 'h2' WHERE id = 'ds_cache'")
    service.search(embedder, "ds_cache", "record 12", support)

    assert embedder.documents == RECORDS * 2
