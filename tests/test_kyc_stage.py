"""The KYC corpus is staged for the rented GPU the way the RFC lane is.

A released synthetic dataset behind its gate becomes a private package: the
released training text, the dev questions by id with their documents but
never the answer block, and the repository's own trainer. Anything else
refuses.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from internal_ai.config import offline_mode
from internal_ai.data.kyc import DOC_TEMPLATES, build_corpus, evidence_lines
from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.services.kyc_stage import stage
from internal_ai.training.alpha.pairs import kyc_documents_from_text, kyc_fill_text

FORMS = 12
DECOYS = 2
FAMILIES = 6


def synthetic_dataset(
    platform, enroll_person, tmp_path, monkeypatch, released=True, legal_status="internal"
):
    from internal_ai.services import build

    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(tmp_path))
    owner = enroll_person(platform, "Dana Owner", ["data_owner"])
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("{}\n")
    source = build.register_source(
        platform,
        owner,
        str(train_file),
        "kyc-synthetic-stage-test",
        "training text for the KYC form-filling agent",
        ["engineering"],
        legal_status=legal_status,
        license_name="template-generated synthetic data",
        license_url="https://example.internal/kyc-synthetic",
        license_note="no real customer, every value carries a fake marker",
    )
    corpus = build_corpus(count=FORMS, seed=21)
    records = []
    for index, form in enumerate(corpus.forms):
        pool = [other for other in corpus.forms if other.customer_id != form.customer_id]
        rng = random.Random(21 + index)
        distractors = tuple(
            rng.choice(DOC_TEMPLATES).format(**rng.choice(pool).values) for _ in range(DECOYS)
        )
        text = kyc_fill_text(form.values, evidence_lines(form, rng, distractors))
        records.append(
            {
                "record_key": form.form_id,
                "payload": text,
                "sensitivity": "internal",
                "split": "test" if index >= FORMS - 4 else "train",
            }
        )
    dataset = DatasetVersion(
        id="ds_stage_test",
        name="kyc-synthetic-stage-test",
        version=1,
        content_hash="stage-test",
        record_count=len(records),
        token_count=1000,
        created_at=now_iso(),
        lineage={"sources": [{"source_id": source["id"]}]},
        released=released,
    )
    platform.store.add_dataset(dataset)
    platform.store.add_dataset_records("ds_stage_test", records)
    return "ds_stage_test"


def test_documents_survive_the_training_text() -> None:
    corpus = build_corpus(count=4, seed=21)
    form = corpus.forms[0]
    lines = evidence_lines(form, random.Random(21))
    assert kyc_documents_from_text(kyc_fill_text(form.values, lines)) == lines


def test_stage_refuses_an_unreleased_dataset(
    platform, enroll_person, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id = synthetic_dataset(platform, enroll_person, tmp_path, monkeypatch, released=False)
    with pytest.raises(SystemExit, match="not released"):
        stage(platform, dataset_id, tmp_path / "kaggle")


def test_stage_refuses_a_non_synthetic_source(
    platform, enroll_person, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id = synthetic_dataset(
        platform, enroll_person, tmp_path, monkeypatch, legal_status="licensed"
    )
    with pytest.raises(SystemExit, match="synthetic"):
        stage(platform, dataset_id, tmp_path / "kaggle")


def test_stage_refuses_text_without_fake_markers(
    platform, enroll_person, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    assert offline_mode() is False
    dataset_id = synthetic_dataset(platform, enroll_person, tmp_path, monkeypatch)
    platform.store.conn.execute(
        "UPDATE dataset_records SET payload = ? WHERE dataset_id = ? AND record_key = ?",
        (
            json.dumps(
                {"text": "A real customer named John Smith lives here.", "form_id": "KYC-X"}
            ),
            dataset_id,
            "KYC-00001",
        ),
    )
    with pytest.raises(SystemExit, match="fake marker"):
        stage(platform, dataset_id, tmp_path / "kaggle")


def test_stage_refuses_while_offline(
    platform, enroll_person, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "1")
    assert offline_mode() is True
    dataset_id = synthetic_dataset(platform, enroll_person, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="offline"):
        stage(platform, dataset_id, tmp_path / "kaggle")


def test_staged_package_carries_texts_questions_and_trainer(
    platform, enroll_person, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    dataset_id = synthetic_dataset(platform, enroll_person, tmp_path, monkeypatch)
    out = stage(platform, dataset_id, tmp_path / "kaggle")
    dataset_dir = Path(out["dataset_dir"])
    texts = [json.loads(line) for line in (dataset_dir / "train.jsonl").read_text().splitlines()]
    assert len(texts) == FORMS - 4
    assert all(set(item) == {"text"} for item in texts)
    queries = json.loads((dataset_dir / "queries.json").read_text())
    assert len(queries) == 4
    for item in queries:
        assert set(item) == {"id", "documents"}
        assert len(item["documents"]) == FAMILIES + DECOYS
        assert "Answer:" not in json.dumps(item)
    bundle = dataset_dir / "internal_ai.zip"
    assert bundle.exists()
    recipe = json.loads((dataset_dir / "recipe.json").read_text())
    assert recipe["dataset_id"] == dataset_id
    assert recipe["base"]["repo_id"] == "google/gemma-4-E2B-it"
    assert recipe["pair_format"] == "kyc_documents_answers_first"
    audit = platform.store.conn.execute(
        "SELECT action FROM audit_events WHERE action = 'remote_training_export'"
    ).fetchall()
    assert len(audit) == 1
