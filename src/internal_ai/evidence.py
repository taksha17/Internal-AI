"""Human sign-offs the checklist will not accept as a guess in code."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import GateType
from internal_ai.ids import new_id, now_iso
from internal_ai.services.actor import Actor

KINDS = (
    "compute_reservation",
    "run_health",
    "security_review",
    "instability_playbook",
    "facilities",
)
_ROLES = {
    "run_health": "ml_lead",
    "security_review": "security",
    "compute_reservation": "program_lead",
    "facilities": "program_lead",
    "instability_playbook": "ml_lead",
}


def sign_evidence(
    platform: Platform,
    actor: Actor,
    kind: str,
    body: str,
    subject: str,
    started_by: str = "",
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown evidence kind: {kind}")
    role = _ROLES[kind]
    if not actor.holds(role):
        raise PermissionError(f"{kind} is signed by {role}")
    if kind == "run_health" and started_by and actor.id == started_by:
        raise PermissionError("the person who started the run cannot sign its health")
    row = {
        "id": new_id("evd"),
        "kind": kind,
        "body": body,
        "signed_by": actor.id,
        "signed_at": now_iso(),
        "subject": subject,
    }
    platform.store.conn.execute(
        "INSERT INTO evidence (id, kind, body, signed_by, signed_at, subject) VALUES (?,?,?,?,?,?)",
        (row["id"], kind, body, row["signed_by"], row["signed_at"], subject),
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind="human",
        action="evidence.signed",
        subject=subject,
        details={"kind": kind},
    )
    return row


def list_evidence(platform: Platform, kind: str, subject: str = "") -> list[dict[str, str]]:
    query = "SELECT * FROM evidence WHERE kind = ?"
    values: list[str] = [kind]
    if subject:
        query += " AND subject = ?"
        values.append(subject)
    rows = platform.store.conn.execute(query, values).fetchall()
    return [dict(row) for row in rows]


def require_compute(platform: Platform, estimate_hours: float, actor: str) -> str:
    """A long estimate needs a reservation that covers today. A short one is local."""
    if estimate_hours <= 1:
        platform.audit.record(
            actor=actor,
            actor_kind="human",
            action="train.compute",
            subject="local",
            details={"compute_reservation": "local"},
        )
        return "local"
    today = date.today().isoformat()
    for row in list_evidence(platform, "compute_reservation"):
        try:
            window = json.loads(row["body"])
        except json.JSONDecodeError:
            continue
        if str(window.get("start", "")) <= today <= str(window.get("end", "")):
            return row["id"]
    raise PermissionError(
        "a run longer than one GPU hour needs a compute reservation that covers today"
    )


def note_run_hours(
    platform: Platform, run_id: str, used_hours: float, estimate_hours: float, actor: str
) -> None:
    """Crossing the estimate opens a gate. It does not stop a run under its dollar cap."""
    if used_hours <= estimate_hours:
        return
    open_gates = [
        gate
        for gate in platform.store.list_gates()
        if str(gate.gate_type) == str(GateType.TRAINING_RUN_COST)
        and gate.subject == run_id
        and str(gate.status) == "pending"
    ]
    if open_gates:
        return
    platform.gates.request(
        GateType.TRAINING_RUN_COST,
        "training run crossed its GPU-hour estimate",
        run_id,
        actor,
        requested_by_kind="human",
        evidence={"used_gpu_hours": used_hours, "estimate_gpu_hours": estimate_hours},
    )


def require_playbook(platform: Platform, planning_hours: float) -> None:
    if planning_hours <= 1:
        return
    rows = list_evidence(platform, "instability_playbook")
    written = [row for row in rows if row["body"].strip()]
    if not written:
        raise PermissionError("promotion of a long run needs an instability playbook")
