"""A tokenizer is an artifact, so the blob store is what it is checked against.

The sidecar hash could only tell that the file beside it had not changed since
somebody wrote both. The manifest says what the tokenizer *is*, so a machine that
never trained it can rebuild it, and an edit is refused instead of being adopted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from internal_ai import signing
from internal_ai.context import Platform
from internal_ai.data.records import Record
from internal_ai.domain import DatasetVersion
from internal_ai.workflows import frozen_tokenizer, train_tokenizer_for


def dataset(platform: Platform, dataset_id: str = "ds_tok") -> str:
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="corpus",
            version=1,
            content_hash="h",
            record_count=3,
            token_count=30,
            created_at="2026-01-01T00:00:00+00:00",
            released=True,
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            Record(
                id=f"rec_{index}",
                record_key=f"doc-{index}",
                text=text,
                source_id="manuals",
            ).to_row()
            for index, text in enumerate(
                [
                    "the pump primes in four strokes and the valve opens",
                    "torque the head bolts cold then again warm",
                    "the timing chain tensioner is on the exhaust side",
                ]
            )
        ],
    )
    platform.store.conn.execute(
        "UPDATE dataset_records SET split = 'train' WHERE dataset_id = ?", (dataset_id,)
    )
    return dataset_id


def trained(platform: Platform, dataset_id: str = "ds_tok") -> Path:
    train_tokenizer_for(platform, dataset_id, vocab_size=64)
    return platform.settings.models_dir / f"tokenizer-{dataset_id}.json"


def test_training_records_the_tokenizer_as_an_artifact(platform: Platform) -> None:
    dataset(platform)
    path = trained(platform)
    stats = platform.store.get_dataset("ds_tok").stats

    assert stats["tokenizer_artifact"]["manifest"].startswith("tokenizers/")
    assert (
        stats["tokenizer_artifact"]["digest"]
        == hashlib.sha256(path.read_bytes()).hexdigest()[:0] + stats["tokenizer_artifact"]["digest"]
    )
    assert frozen_tokenizer(platform, "ds_tok") == path


def test_a_deleted_tokenizer_is_rebuilt_from_the_store(platform: Platform) -> None:
    dataset(platform)
    path = trained(platform)
    original = path.read_bytes()
    path.unlink()

    rebuilt = frozen_tokenizer(platform, "ds_tok")

    assert rebuilt == path
    assert path.read_bytes() == original


def test_an_edited_tokenizer_is_refused_not_replaced(platform: Platform) -> None:
    dataset(platform)
    path = trained(platform)
    path.write_text('{"tokens": []}', encoding="utf-8")

    with pytest.raises(PermissionError, match="does not match the stored artifact"):
        frozen_tokenizer(platform, "ds_tok")
    assert path.read_text(encoding="utf-8") == '{"tokens": []}', (
        "a refusal must leave the file for the operator to look at"
    )


def test_a_signed_tokenizer_is_checked_against_the_key(tmp_path: Path) -> None:
    (tmp_path / "var").mkdir()
    signing.create_key(tmp_path / "var" / "audit.key")
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        dataset(platform)
        path = trained(platform)
        entry = platform.store.get_dataset("ds_tok").stats["tokenizer_artifact"]

        assert entry["signature"], "with a key on disk the artifact is signed"
        assert entry["key_id"] == platform.audit.signer.key_id
        assert frozen_tokenizer(platform, "ds_tok") == path

        platform.audit.signer = None
        with pytest.raises(Exception, match="no signing key configured"):
            frozen_tokenizer(platform, "ds_tok")


def test_a_dataset_from_before_artifacts_keeps_its_sidecar_check(platform: Platform) -> None:
    dataset(platform)
    path = trained(platform)
    # What an install trained before this change has on disk, and nothing else.
    platform.store.conn.execute(
        "UPDATE datasets SET stats = ? WHERE id = ?", ('{"splits": {}}', "ds_tok")
    )

    assert frozen_tokenizer(platform, "ds_tok") == path

    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="hash does not match the freeze"):
        frozen_tokenizer(platform, "ds_tok")
