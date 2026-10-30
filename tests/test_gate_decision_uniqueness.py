"""One signoff row per gate and approver, even under a race (change 0.3)."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from internal_ai.domain import GateDecision, GateType
from internal_ai.gates import GateError, GateService
from internal_ai.ids import now_iso

APPROVER = "ppl_marcus_delgado"


def _pending_gate(platform) -> tuple[GateService, str]:
    service = GateService(platform.store)
    gate = service.request(
        GateType.MODEL_PROMOTION,
        "Promote the encoder",
        "mdl_1",
        requested_by="agent:evaluation",
        evidence={
            "eval_results_vs_baseline": "recall 0.41 against 0.30",
            "model_card": "mdl_1",
            "red_team_results": "no leak across groups",
        },
    )
    return service, gate.id


def test_simultaneous_decisions_by_one_person_leave_one_row(platform) -> None:
    service, gate_id = _pending_gate(platform)

    def decide(_: int) -> str:
        try:
            service.decide(gate_id, APPROVER, True, note="reviewed")
        except GateError as exc:
            return str(exc)
        return "decided"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(decide, range(8)))

    assert outcomes.count("decided") == 1
    assert len(platform.store.gate_decisions(gate_id)) == 1


def test_the_signoff_record_refuses_a_second_row(platform) -> None:
    gate_id = _pending_gate(platform)[1]

    def record() -> None:
        platform.store.add_gate_decision(
            GateDecision(
                id=f"dec_{now_iso()}",
                gate_id=gate_id,
                approver=APPROVER,
                approver_kind="human",
                approved=True,
                note="",
                created_at=now_iso(),
            )
        )

    record()
    with pytest.raises(sqlite3.IntegrityError):
        record()


def test_distinct_approvers_still_count_towards_a_gate(platform) -> None:
    service, gate_id = _pending_gate(platform)

    service.decide(gate_id, APPROVER, True, note="ml lead signoff")

    assert platform.store.gate_decisions(gate_id)[0].approver == APPROVER
    assert service.is_approved(gate_id) is False


def test_a_duplicate_decision_is_refused_with_a_reason(platform) -> None:
    service, gate_id = _pending_gate(platform)
    platform.store.add_gate_decision(
        GateDecision(
            id="dec_first",
            gate_id=gate_id,
            approver=APPROVER,
            approver_kind="human",
            approved=True,
            note="",
            created_at=now_iso(),
        )
    )

    with pytest.raises(GateError, match="already decided"):
        service.decide(gate_id, APPROVER, True, note="again")
