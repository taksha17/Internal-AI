"""Immutable audit log with a verifiable hash chain (FR-20, FR-21)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from internal_ai.domain import AuditEvent, AuditOutcome
from internal_ai.ids import content_hash, new_id
from internal_ai.store import Store


@dataclass(slots=True)
class AuditVerification:
    ok: bool
    checked: int
    broken_at_seq: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "broken_at_seq": self.broken_at_seq,
            "reason": self.reason,
        }


def compute_hash(event: AuditEvent) -> str:
    return content_hash(
        {
            "id": event.id,
            "actor": event.actor,
            "actor_kind": event.actor_kind,
            "action": event.action,
            "subject": event.subject,
            "outcome": str(event.outcome),
            "inputs": event.inputs,
            "details": event.details,
            "cost_usd": event.cost_usd,
            "prev_hash": event.prev_hash,
            "created_at": event.created_at,
            "tenant": event.tenant,
        }
    )


class AuditLog:
    """Append-only event log.

    Every side-effecting action (job submission, dataset write, deployment,
    budget change) must be recorded here with agent identity, inputs and outcome.
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    def record(
        self,
        actor: str,
        action: str,
        subject: str,
        outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
        actor_kind: str = "agent",
        inputs: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        tenant: str = "default",
    ) -> AuditEvent:
        event = AuditEvent(
            id=new_id("aud"),
            seq=0,
            actor=actor,
            actor_kind=actor_kind,
            action=action,
            subject=subject,
            outcome=outcome if isinstance(outcome, AuditOutcome) else AuditOutcome(outcome),
            inputs=inputs or {},
            details=details or {},
            cost_usd=cost_usd,
            tenant=tenant,
        )
        return self.store.append_audit(event)

    def verify(self) -> AuditVerification:
        previous_hash = ""
        checked = 0
        for event in self.store.audit_chain():
            checked += 1
            if event.prev_hash != previous_hash:
                return AuditVerification(
                    False,
                    checked,
                    event.seq,
                    f"broken link: prev_hash {event.prev_hash[:12]} != {previous_hash[:12]}",
                )
            if compute_hash(event) != event.hash:
                return AuditVerification(
                    False, checked, event.seq, "hash mismatch: event was modified"
                )
            previous_hash = event.hash
        return AuditVerification(True, checked)

    def recent(self, limit: int = 50, **filters: Any) -> list[AuditEvent]:
        return self.store.list_audit(limit=limit, **filters)
