"""Provenance mode: the default loads nothing, and only a decided policy changes that."""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.services import governance
from internal_ai.services.alpha import train_alpha
from internal_ai.services.jobs import JobRunner
from internal_ai.training.alpha.pin import FALLBACK, PRIMARY
from internal_ai.training.alpha.promote import register_alpha
from internal_ai.training.provenance import (
    OPEN_WEIGHT_ALLOWED,
    SCRATCH_ONLY,
    ProvenanceRefused,
    allow_open_weights,
    provenance_mode,
)
from internal_ai.workflows import model_card_markdown


def decide(platform, value: str, status: str = "decided") -> None:
    decision = platform.policy.get("P6")
    decision.current_value = value
    decision.status = status
    platform.store.upsert_policy(decision)


def released_dataset(platform) -> str:
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_prov",
            name="company",
            version=1,
            content_hash="hash-prov",
            record_count=1,
            token_count=1,
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(
        "ds_prov",
        [{"record_key": "r1", "payload": "the ledger feed failed overnight", "acl": []}],
    )
    return "ds_prov"


def test_the_default_install_loads_no_pretrained_weights(platform) -> None:
    assert provenance_mode(platform) == SCRATCH_ONLY
    assert allow_open_weights(platform) is False


def test_a_pending_proposal_does_not_switch_the_install_on(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Pat Lead", ["program_lead"])
    governance.propose_policy(platform, actor, "P6", OPEN_WEIGHT_ALLOWED, "we own a licence")
    assert provenance_mode(platform) == SCRATCH_ONLY
    assert allow_open_weights(platform) is False


def test_a_decided_policy_switches_the_install_on(platform) -> None:
    decide(platform, OPEN_WEIGHT_ALLOWED)
    assert provenance_mode(platform) == OPEN_WEIGHT_ALLOWED
    assert allow_open_weights(platform) is True


def test_an_unrecognised_value_means_scratch_only(platform) -> None:
    decide(platform, "everything_is_fine")
    assert provenance_mode(platform) == SCRATCH_ONLY


def test_adaptation_is_refused_under_scratch_only(platform, enroll_person) -> None:
    actor = enroll_person(platform, "Ada Lovelace", ["ml_lead"])
    released_dataset(platform)
    with pytest.raises(ProvenanceRefused, match="scratch_only"):
        train_alpha(platform, actor, JobRunner(platform, synchronous=True), "ds_prov")
    assert platform.store.list_runs() == []


def test_the_card_names_the_mode(platform, enroll_person, tmp_path: Path) -> None:
    actor = enroll_person(platform, "Ada Lovelace", ["ml_lead"])
    released_dataset(platform)
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    model = register_alpha(platform, PRIMARY, adapter, "ds_prov", actor)
    assert model.card["provenance_mode"] == SCRATCH_ONLY
    text = model_card_markdown(model)
    assert "Provenance: scratch_only" in text
    decide(platform, OPEN_WEIGHT_ALLOWED)
    model = register_alpha(platform, PRIMARY, adapter, "ds_prov", actor)
    assert "Provenance: open_weight_allowed" in model_card_markdown(model)


def test_the_mode_does_not_bypass_the_licence_check(
    platform, enroll_person, tmp_path: Path
) -> None:
    """Two locks, not one: a decided mode still refuses a research-licensed base."""
    actor = enroll_person(platform, "Ada Lovelace", ["ml_lead"])
    released_dataset(platform)
    decide(platform, OPEN_WEIGHT_ALLOWED)
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    with pytest.raises(PermissionError, match="base_model_exception"):
        register_alpha(platform, FALLBACK, adapter, "ds_prov", actor)
