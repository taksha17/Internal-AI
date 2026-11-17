"""A signed artifact is a claim about bytes that survives someone who can write the bucket.

Hashing alone catches an edited file. It does not catch a manifest rewritten
together with the files it lists, which is why the digest is signed with a key the
object store does not hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai import signing
from internal_ai.artifacts import (
    ArtifactDenied,
    artifact_from_card,
    require_intact,
    signature_message,
    store_artifact,
    verify_signature,
)
from internal_ai.storage import blob_store


def make_artifact(tmp_path: Path):
    source = tmp_path / "adapter"
    source.mkdir(exist_ok=True)
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "adapter_config.json").write_text('{"r": 8}')
    return source


def test_an_artifact_is_signed_and_the_card_records_the_key(tmp_path: Path) -> None:
    key = signing.create_key(tmp_path / "audit.key")
    blobs = blob_store(tmp_path / "objects")

    artifact = store_artifact(blobs, make_artifact(tmp_path), signer=key)

    assert artifact.signature
    assert artifact.key_id == key.key_id
    assert artifact.to_dict()["signature"] == artifact.signature
    verify_signature(artifact, key)


def test_an_unsigned_artifact_is_still_accepted(tmp_path: Path) -> None:
    blobs = blob_store(tmp_path / "objects")

    artifact = store_artifact(blobs, make_artifact(tmp_path))
    key = signing.create_key(tmp_path / "audit.key")

    assert artifact.signature == ""
    verify_signature(artifact, key)
    verify_signature(artifact, None)


def test_a_signature_from_another_key_is_refused(tmp_path: Path) -> None:
    blobs = blob_store(tmp_path / "objects")
    artifact = store_artifact(
        blobs, make_artifact(tmp_path), signer=signing.create_key(tmp_path / "a.key")
    )
    other = signing.create_key(tmp_path / "b.key")

    with pytest.raises(ArtifactDenied, match="signed with key"):
        verify_signature(artifact, other)


def test_a_rewritten_manifest_is_refused(tmp_path: Path) -> None:
    key = signing.create_key(tmp_path / "audit.key")
    blobs = blob_store(tmp_path / "objects")
    artifact = store_artifact(blobs, make_artifact(tmp_path), signer=key)

    forged = type(artifact)(
        manifest_key=artifact.manifest_key,
        digest=artifact.digest,
        files=artifact.files,
        size=artifact.size,
        prefix=artifact.prefix,
        signature="0" * 64,
        key_id=artifact.key_id,
    )

    with pytest.raises(ArtifactDenied, match="does not match its signature"):
        verify_signature(forged, key)


def test_a_signed_artifact_cannot_be_loaded_without_a_key(tmp_path: Path) -> None:
    key = signing.create_key(tmp_path / "audit.key")
    blobs = blob_store(tmp_path / "objects")
    artifact = store_artifact(blobs, make_artifact(tmp_path), signer=key)

    with pytest.raises(ArtifactDenied, match="no signing key configured"):
        verify_signature(artifact, None)


def test_the_signature_covers_the_digest_only(tmp_path: Path) -> None:
    key = signing.create_key(tmp_path / "audit.key")
    blobs = blob_store(tmp_path / "objects")
    artifact = store_artifact(blobs, make_artifact(tmp_path), signer=key)

    assert key.verify(signature_message(artifact.digest), artifact.signature)
    assert not key.verify(signature_message("deadbeef"), artifact.signature)


def test_a_card_carries_the_signature_into_the_loader(tmp_path: Path) -> None:
    key = signing.create_key(tmp_path / "audit.key")
    blobs = blob_store(tmp_path / "objects")
    artifact = store_artifact(blobs, make_artifact(tmp_path), signer=key)
    card = {"artifact": artifact.to_dict()}

    from_card = artifact_from_card(card)

    assert from_card is not None
    assert from_card.signature == artifact.signature
    assert from_card.key_id == key.key_id
    require_intact(blobs, make_artifact(tmp_path), card, signer=key)

    forged = {"artifact": {**artifact.to_dict(), "signature": "1" * 64}}
    with pytest.raises(ArtifactDenied, match="does not match its signature"):
        require_intact(blobs, make_artifact(tmp_path), forged, signer=key)
