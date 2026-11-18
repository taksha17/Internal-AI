"""A dataset version keeps the groups its records named, and retrieval honours that ceiling."""

from __future__ import annotations

import json

from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.data.pipeline import DataPipeline, PipelineConfig
from internal_ai.data.records import Source
from internal_ai.services.build import dataset_view
from internal_ai.serving.service import Principal, RetrievalService

ENGINEERING = (
    "the engineering handbook says to restart the ledger feed when the overnight run fails"
)
LEGAL = "the legal hold on the contract review stays in place until counsel releases it"


def _build(platform, tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "eng-1", "text": ENGINEERING, "acl": ["engineering"]}),
                json.dumps({"id": "leg-1", "text": LEGAL, "acl": ["legal"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = DataPipeline(platform.store, platform.audit, tmp_path, platform.blobs).run(
        [
            Source(
                source_id="src_handbook",
                path=path,
                acl=["engineering", "legal"],
                approved=True,
                owner="tester",
            )
        ],
        PipelineConfig(name="handbook", min_chars=20),
    )
    return result.dataset


def test_the_version_carries_the_union_and_a_widened_row_cannot_raise_it(platform, tmp_path):
    dataset = _build(platform, tmp_path)
    assert dataset.acl == ["engineering", "legal"]
    stored = platform.store.get_dataset(dataset.id)
    assert stored is not None and stored.acl == ["engineering", "legal"]

    retrieval = RetrievalService(platform.store, platform.audit, platform.blobs)
    engineering = Principal(user_id="p_eng", groups=frozenset({"engineering"}))
    legal = Principal(user_id="p_leg", groups=frozenset({"legal"}))
    finance = Principal(user_id="p_fin", groups=frozenset({"finance"}))
    assert [
        hit.document_id for hit in retrieval.keyword_search(dataset.id, "ledger", engineering)
    ] == ["eng-1"]
    assert [hit.document_id for hit in retrieval.keyword_search(dataset.id, "contract", legal)] == [
        "leg-1"
    ]
    assert retrieval.keyword_search(dataset.id, "ledger", finance) == []

    platform.store.conn.execute(
        "UPDATE dataset_records SET acl = ? WHERE dataset_id = ? AND record_key = ?",
        (json.dumps(["engineering", "finance"]), dataset.id, "eng-1"),
    )
    assert retrieval.keyword_search(dataset.id, "ledger", finance) == []
    assert [
        hit.document_id for hit in retrieval.keyword_search(dataset.id, "ledger", engineering)
    ] == ["eng-1"]


def test_finance_sees_none_of_a_version_that_does_not_name_them(platform, tmp_path):
    dataset = _build(platform, tmp_path)
    owner = enroll(platform, "Dana Owner", ["data_owner"], ["engineering", "legal"])
    books = enroll(platform, "Finn Books", ["finance"], ["finance"])
    assert dataset_view(platform, owner, dataset.id)["visible_records"] == 2
    assert dataset_view(platform, books, dataset.id)["visible_records"] == 0


def test_the_dataset_page_names_the_groups(tmp_path):
    app = create_app(root=tmp_path / "home", llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        dataset = _build(platform, tmp_path)
        enroll(platform, "Dana Owner", ["data_owner"], ["engineering", "legal"])
        assert sign_in(client, "Dana Owner").status_code == 303
        page = client.get(f"/console/datasets/{dataset.id}")
        assert page.status_code == 200
        assert "engineering" in page.text
        assert "legal" in page.text
        assert "You can read 2" in page.text
