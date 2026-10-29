"""Language tags are reviewed, and a toxic match is kept only with the policy sentence."""

from __future__ import annotations

import pytest
from conftest import enroll

from internal_ai.data.language import (
    BORDERLINE_POLICY,
    check_release_quality,
    language_sample,
    store_borderline_policy,
    store_language_review,
    submit_language_review,
)
from internal_ai.data.records import Record
from internal_ai.data.transforms import filter_quality
from internal_ai.domain import DatasetVersion
from internal_ai.ids import new_id


def test_english_prose_is_tagged_en() -> None:
    records = [
        Record(
            id="1",
            record_key="note",
            text="the team will reset the vpn profile before you sign in on monday morning",
            source_id="src",
        )
    ]
    kept, _stats = filter_quality(records)
    assert kept[0].meta["language"] == "en"


def _dataset(platform, texts: list[str], languages: list[str] | None = None) -> str:
    dataset_id = new_id("ds")
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="tags",
            version=1,
            content_hash="h",
            record_count=len(texts),
            token_count=len(texts),
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": f"row-{index}",
                "payload": text,
                "meta": {"language": (languages or ["en"])[index % len(languages or ["en"])]},
            }
            for index, text in enumerate(texts)
        ],
    )
    return dataset_id


def test_four_wrong_of_thirty_fails_release(platform) -> None:
    texts = [f"the team will reset vpn profile number {index} before monday" for index in range(30)]
    dataset_id = _dataset(platform, texts)
    sample = language_sample(platform.store, dataset_id)
    marks = {row["id"]: "correct" for row in sample}
    for row in sample[:4]:
        marks[row["id"]] = "wrong"
    store_language_review(platform.store, dataset_id, marks, "ppl_dex")
    with pytest.raises(PermissionError, match="over 10 percent"):
        check_release_quality(platform.store, dataset_id)


def test_a_flagged_row_without_the_policy_sentence_fails_release(platform) -> None:
    dataset_id = _dataset(
        platform,
        [
            "the team will reset the vpn profile before monday morning",
            "that workaround is an asshole of a fix and should be rewritten",
        ],
    )
    sample = language_sample(platform.store, dataset_id)
    store_language_review(
        platform.store, dataset_id, {row["id"]: "correct" for row in sample}, "ppl_dex"
    )
    with pytest.raises(PermissionError, match="borderline policy"):
        check_release_quality(platform.store, dataset_id)
    store_borderline_policy(platform.store, dataset_id, BORDERLINE_POLICY)
    report = check_release_quality(platform.store, dataset_id)
    assert report["toxicity_flagged"] == 1
    flagged = [
        row for row in platform.store.dataset_records(dataset_id) if row["meta"].get("borderline")
    ]
    assert len(flagged) == 1


def test_only_a_domain_expert_can_file_the_language_review(platform) -> None:
    dataset_id = _dataset(platform, ["the team will reset the vpn profile before monday"])
    owner = enroll(platform, "Ada Owner", ["data_owner"])
    sample = language_sample(platform.store, dataset_id)
    with pytest.raises(PermissionError, match="domain expert"):
        submit_language_review(
            platform.store, owner, dataset_id, {row["id"]: "correct" for row in sample}
        )
