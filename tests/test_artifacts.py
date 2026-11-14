"""An artifact is a directory with a manifest, not a path somebody hopes is still there.

The failures worth testing are the quiet ones: a file that changed after the card
was written, a manifest that no longer matches its own listing, a symlink that
would drag a file from outside the artifact, and a rebuild that silently accepts
bytes the registry never stored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.artifacts import (
    Artifact,
    ArtifactDenied,
    artifact_digest,
    collect,
    file_key,
    manifest_bytes,
    materialize,
    missing,
    read_card_artifact,
    read_manifest,
    store_artifact,
    verify,
)
from internal_ai.storage import LocalBlobStore, StorageDenied


def _adapter(root: Path) -> Path:
    adapter = root / "qwen-adapter"
    (adapter / "sub").mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"rank": 8}\n', encoding="utf-8")
    (adapter / "adapter_model.bin").write_bytes(b"\x00\x01weights")
    (adapter / "sub" / "tokenizer.json").write_text('{"vocab": 1}\n', encoding="utf-8")
    return adapter


def test_collect_names_files_relative_to_the_artifact(tmp_path: Path) -> None:
    files = collect(_adapter(tmp_path))

    assert [item.name for item in files] == [
        "adapter_config.json",
        "adapter_model.bin",
        "sub/tokenizer.json",
    ]
    assert all(item.size > 0 for item in files)


def test_two_copies_of_the_same_tree_have_the_same_digest(tmp_path: Path) -> None:
    first = _adapter(tmp_path / "a")
    second = _adapter(tmp_path / "b")
    (second / "adapter_model.bin").touch()  # a different mtime, same bytes

    assert artifact_digest(collect(first)) == artifact_digest(collect(second))


def test_a_changed_file_changes_the_digest(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    before = artifact_digest(collect(adapter))
    (adapter / "adapter_model.bin").write_bytes(b"\x00\x01different")

    assert artifact_digest(collect(adapter)) != before


def test_an_empty_directory_is_not_an_artifact(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(ArtifactDenied, match="no files"):
        collect(tmp_path / "empty")


def test_a_symlink_is_refused(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    (adapter / "linked.bin").symlink_to(adapter / "adapter_model.bin")

    with pytest.raises(ArtifactDenied, match="symlink"):
        collect(adapter)


def test_the_manifest_is_stored_under_its_own_hash(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")
    artifact = store_artifact(blobs, _adapter(tmp_path))

    assert artifact.manifest_key.endswith(artifact.digest)
    assert blobs.get(artifact.manifest_key) == manifest_bytes(artifact.files)
    assert read_manifest(blobs, artifact.manifest_key) == artifact.files


def test_a_rebuilt_artifact_is_byte_identical(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")
    artifact = store_artifact(blobs, _adapter(tmp_path))

    target = tmp_path / "rebuilt"
    materialize(blobs, artifact, target)

    assert (target / "adapter_config.json").read_text(encoding="utf-8") == '{"rank": 8}\n'
    assert (target / "adapter_model.bin").read_bytes() == b"\x00\x01weights"
    assert (target / "sub" / "tokenizer.json").exists()
    assert verify(target, artifact.files) == []
    assert missing(target, artifact.files) == []


def test_a_rebuild_refuses_an_object_that_changed(tmp_path: Path) -> None:
    """Fetching by content address is what makes a rebuild trustworthy."""
    blobs = LocalBlobStore(tmp_path / "objects")
    artifact = store_artifact(blobs, _adapter(tmp_path))
    item = next(entry for entry in artifact.files if entry.name == "adapter_model.bin")
    stored = (tmp_path / "objects") / file_key(
        artifact.prefix, artifact.digest, item.name, item.sha256
    )
    stored.chmod(0o644)
    stored.write_bytes(b"tampered")

    with pytest.raises(StorageDenied):
        materialize(blobs, artifact, tmp_path / "rebuilt")


def test_verify_names_the_file_that_changed(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    files = collect(adapter)
    (adapter / "sub" / "tokenizer.json").write_text('{"vocab": 2}\n', encoding="utf-8")

    assert verify(adapter, files) == ["sub/tokenizer.json"]


def test_an_altered_manifest_is_refused(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")
    artifact = store_artifact(blobs, _adapter(tmp_path))
    path = (tmp_path / "objects") / artifact.manifest_key
    path.chmod(0o644)
    path.write_bytes(b'{"name":"x","sha256":"0","size":1}\n')

    with pytest.raises(StorageDenied):
        read_manifest(blobs, artifact.manifest_key)


def test_a_card_records_the_artifact_and_reads_back(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")
    stored = store_artifact(blobs, _adapter(tmp_path))
    card = {"artifact": stored.to_dict()}

    restored = read_card_artifact(blobs, card)

    assert restored is not None
    assert restored.digest == stored.digest
    assert restored.files == stored.files
    assert restored.size == stored.size


def test_a_card_written_before_manifests_reads_as_none(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")

    assert read_card_artifact(blobs, {"base": "qwen"}) is None


def test_the_artifact_dataclass_carries_what_a_card_needs(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "objects")
    artifact = store_artifact(blobs, _adapter(tmp_path))

    payload = artifact.to_dict()

    assert set(payload) == {"manifest", "digest", "files", "size"}
    assert payload["files"] == 3
    assert isinstance(artifact, Artifact)
