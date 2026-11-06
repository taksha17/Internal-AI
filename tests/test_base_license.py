"""A base may be loaded only when its licence is one this install accepts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from internal_ai.domain import DatasetVersion, GateStatus, GateType
from internal_ai.gates import GATE_CATALOG, GateError
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


EXCEPTION_EVIDENCE = {
    "base_model_licence": "qwen-research (research_only)",
    "licence_review": "legal read the terms and limited the use to internal retrieval",
    "usage_scope": "internal retrieval only, never redistributed or served externally",
}


def released_dataset(platform, dataset_id: str = "ds_alpha") -> str:  # type: ignore[no-untyped-def]
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="company",
            version=1,
            content_hash=f"hash-{dataset_id}",
            record_count=1,
            token_count=1,
            created_at=now_iso(),
            released=True,
        )
    )
    return dataset_id


def open_exception(
    platform,  # type: ignore[no-untyped-def]
    base: PinnedBase,
    requester,  # type: ignore[no-untyped-def]
    approve=(),  # type: ignore[no-untyped-def]
    revision: str | None = None,
):
    gate = platform.gates.request(
        GateType.BASE_MODEL_EXCEPTION,
        "Load a research-licensed base",
        base.repo_id,
        requester.id,
        requested_by_kind="human",
        evidence={**EXCEPTION_EVIDENCE, "base_revision": revision or base.revision},
    )
    for actor in approve:
        platform.gates.decide(gate.id, actor.id, approved=True, note="licence reviewed")
    return platform.store.get_gate(gate.id)


def test_the_exception_gate_names_its_roles_and_evidence() -> None:
    entry = GATE_CATALOG[GateType.BASE_MODEL_EXCEPTION]
    assert str(GateType.BASE_MODEL_EXCEPTION) == "base_model_exception"
    assert entry.approvers == ("program_lead", "legal")
    assert set(entry.evidence) == {
        "base_model_licence",
        "licence_review",
        "usage_scope",
        "base_revision",
    }


def test_an_approved_exception_lets_the_adapter_register(
    platform, enroll_person, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    counsel = enroll_person(platform, "Lea Counsel", ["legal"])
    dataset_id = released_dataset(platform)
    gate = open_exception(platform, FALLBACK, lead, approve=(programme, counsel))
    assert gate.status == GateStatus.APPROVED

    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    model = register_alpha(platform, FALLBACK, adapter, dataset_id, lead, gate_id=gate.id)
    assert model.card["base_model_exception"] == gate.id
    assert model.card["base_license"] == FALLBACK.license


def test_the_approved_exception_is_found_without_a_gate_id(
    platform, enroll_person, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    counsel = enroll_person(platform, "Lea Counsel", ["legal"])
    dataset_id = released_dataset(platform)
    open_exception(platform, FALLBACK, lead, approve=(programme, counsel))

    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    model = register_alpha(platform, FALLBACK, adapter, dataset_id, lead)
    assert model.card["base_model_exception"]


def test_a_pending_exception_is_not_an_approval(platform, enroll_person, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    dataset_id = released_dataset(platform)
    gate = open_exception(platform, FALLBACK, lead)
    assert gate.status == GateStatus.PENDING

    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    with pytest.raises(PermissionError, match="base_model_exception"):
        register_alpha(platform, FALLBACK, adapter, dataset_id, lead, gate_id=gate.id)


def test_one_approver_cannot_cover_both_roles(platform, enroll_person, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    dataset_id = released_dataset(platform)
    gate = open_exception(platform, FALLBACK, lead, approve=(programme,))
    assert gate.status == GateStatus.PENDING, "program_lead alone covered legal too"

    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    with pytest.raises(PermissionError, match="base_model_exception"):
        register_alpha(platform, FALLBACK, adapter, dataset_id, lead, gate_id=gate.id)


def test_the_requester_cannot_approve_their_own_exception(platform, enroll_person) -> None:  # type: ignore[no-untyped-def]
    lead = enroll_person(platform, "Mo Lead", ["ml_lead", "program_lead"])
    gate = open_exception(platform, FALLBACK, lead)
    with pytest.raises(GateError, match="dual control"):
        platform.gates.decide(gate.id, lead.id, approved=True, note="self")


def test_an_exception_for_another_revision_does_not_cover_this_base(
    platform, enroll_person, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    programme = enroll_person(platform, "Pat Programme", ["program_lead"])
    counsel = enroll_person(platform, "Lea Counsel", ["legal"])
    dataset_id = released_dataset(platform)
    gate = open_exception(platform, FALLBACK, lead, approve=(programme, counsel), revision="0" * 40)
    assert gate.status == GateStatus.APPROVED

    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    with pytest.raises(PermissionError, match="base_model_exception"):
        register_alpha(platform, FALLBACK, adapter, dataset_id, lead, gate_id=gate.id)


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
