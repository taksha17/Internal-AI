"""A base may be loaded only when its licence is one this install accepts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.training.alpha import fetch as fetch_module
from internal_ai.training.alpha.pin import (
    DEFAULT_ALLOWED,
    FALLBACK,
    PRIMARY,
    BaseRefused,
    LicenseClass,
    PinnedBase,
    WeightShard,
    available_base,
    license_class_of,
)
from internal_ai.training.alpha.promote import register_alpha


def pin(repo_id: str, licence: str, root: Path, payload: bytes = b"weights") -> PinnedBase:
    shard = WeightShard("model.safetensors", hashlib.sha256(payload).hexdigest(), len(payload))
    base = PinnedBase(repo_id=repo_id, revision="rev1", license=licence, shards=(shard,))
    directory = base.directory(root)
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(payload)
    return base


def absent(base: PinnedBase, repo_id: str) -> PinnedBase:
    """The same shard metadata pointed at a name that is not on disk."""
    return replace(base, repo_id=repo_id)


def test_the_two_pinned_bases_carry_their_licence_class() -> None:
    assert PRIMARY.license_class == LicenseClass.PERMISSIVE
    assert FALLBACK.license_class == LicenseClass.RESEARCH_ONLY
    assert FALLBACK.license_class not in DEFAULT_ALLOWED


def test_an_unreviewed_licence_is_the_strictest_class() -> None:
    assert license_class_of("Apache-2.0") == LicenseClass.PERMISSIVE
    assert license_class_of("MIT") == LicenseClass.PERMISSIVE
    assert license_class_of("CC-BY-NC-4.0") == LicenseClass.RESTRICTED
    assert license_class_of("some-new-licence") == LicenseClass.RESEARCH_ONLY


def test_the_permissive_base_is_returned_when_it_verifies(tmp_path: Path) -> None:
    allowed = pin("Example/Allowed", "Apache-2.0", tmp_path)
    assert available_base(tmp_path, bases=(allowed,)) is allowed


def test_a_research_licence_is_refused_rather_than_swapped_in(tmp_path: Path) -> None:
    research = pin("Example/Research", "qwen-research", tmp_path)
    missing = absent(research, "Example/Missing")
    with pytest.raises(BaseRefused) as refused:
        available_base(tmp_path, bases=(missing, research))
    message = str(refused.value)
    assert "Example/Research" in message
    assert "research_only" in message
    assert "permissive" in message


def test_nothing_on_disk_names_every_candidate(tmp_path: Path) -> None:
    missing = absent(pin("Example/One", "Apache-2.0", tmp_path), "Example/Gone")
    other = absent(missing, "Example/AlsoGone")
    with pytest.raises(BaseRefused, match="no pinned base is on disk"):
        available_base(tmp_path, bases=(missing, other))


def test_a_tampered_shard_is_not_a_candidate(tmp_path: Path) -> None:
    base = pin("Example/Allowed", "Apache-2.0", tmp_path)
    directory = base.directory(tmp_path)
    (directory / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(BaseRefused, match="no pinned base is on disk"):
        available_base(tmp_path, bases=(base,))


def test_fetch_refuses_the_research_pin_unless_it_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []

    def fake_fetch(base: PinnedBase, root: Path) -> Path:
        fetched.append(base.repo_id)
        if base is PRIMARY:
            raise OSError("primary is not reachable")
        return base.directory(root)

    monkeypatch.setattr(fetch_module, "fetch_base", fake_fetch)
    with pytest.raises(BaseRefused, match="research_only"):
        fetch_module.fetch_primary_or_fallback(tmp_path)
    assert fetched == [PRIMARY.repo_id], "the research pin was fetched without being asked for"


def test_fetch_uses_the_research_pin_when_it_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(base: PinnedBase, root: Path) -> Path:
        if base is PRIMARY:
            raise OSError("primary is not reachable")
        return base.directory(root)

    monkeypatch.setattr(fetch_module, "fetch_base", fake_fetch)
    assert fetch_module.fetch_primary_or_fallback(tmp_path, allow_research_licence=True) is FALLBACK


def test_registering_an_adapter_on_a_research_base_is_refused(
    platform, enroll_person, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_alpha",
            name="company",
            version=1,
            content_hash="hash-alpha",
            record_count=1,
            token_count=1,
            created_at=now_iso(),
            released=True,
        )
    )
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    with pytest.raises(PermissionError, match="research_only"):
        register_alpha(platform, FALLBACK, adapter, "ds_alpha", lead)
