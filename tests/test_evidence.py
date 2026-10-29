"""Sign-offs for a long run, and a short run that stays on the local machine."""

from __future__ import annotations

import json

import pytest
from conftest import enroll

from internal_ai.evidence import note_run_hours, require_compute, sign_evidence


def test_the_starter_cannot_sign_run_health(platform) -> None:
    lead = enroll(platform, "Mo Lead", ["ml_lead"])
    with pytest.raises(PermissionError, match="cannot sign"):
        sign_evidence(
            platform,
            lead,
            "run_health",
            "loss is finite",
            "run_1",
            started_by=lead.id,
        )


def test_a_long_estimate_without_a_reservation_is_denied(platform) -> None:
    with pytest.raises(PermissionError, match="compute reservation"):
        require_compute(platform, 2.5, "ppl_mo")
    lead = enroll(platform, "Ada Lead", ["program_lead"])
    sign_evidence(
        platform,
        lead,
        "compute_reservation",
        json.dumps({"start": "2020-01-01", "end": "2099-01-01"}),
        "programme",
    )
    assert require_compute(platform, 2.5, lead.id)


def test_a_short_run_records_the_local_machine(platform) -> None:
    assert require_compute(platform, 0.01, "ppl_mo") == "local"
    event = platform.store.list_audit(action="train.compute", limit=1)[0]
    assert event.details["compute_reservation"] == "local"


def test_crossing_the_estimate_opens_a_gate_and_leaves_the_run(platform) -> None:
    note_run_hours(platform, "run_long", used_hours=3, estimate_hours=1, actor="ppl_mo")
    gates = platform.store.list_gates()
    assert any(gate.subject == "run_long" for gate in gates)
