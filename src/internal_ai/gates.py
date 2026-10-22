"""Gate service: approval requests, signoff records, dual control (PRD section 14)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from internal_ai.domain import GateDecision, GateRequest, GateStatus, GateType
from internal_ai.ids import new_id, now_iso
from internal_ai.store import Store

HUMAN_APPROVER_ROLES: dict[GateType, tuple[str, ...]] = {
    GateType.NEW_DATA_SOURCE: ("data_owner", "compliance"),
    GateType.DATASET_RELEASE: ("ml_lead", "compliance"),
    GateType.EVAL_SET_FREEZE: ("ml_lead",),
    GateType.TRAINING_RUN_COST: ("ml_lead", "program_lead"),
    GateType.MODEL_PROMOTION: ("ml_lead", "domain_expert"),
    GateType.PRODUCTION_DEPLOYMENT: ("engineering_manager", "security"),
    GateType.BUDGET_INCREASE: ("program_lead", "finance"),
    GateType.POLICY_DECISION: ("program_lead", "legal"),
}

EVIDENCE_REQUIREMENTS: dict[GateType, tuple[str, ...]] = {
    GateType.NEW_DATA_SOURCE: ("source_description", "sensitivity_class", "purpose"),
    GateType.DATASET_RELEASE: ("quality_report", "pii_audit_sample", "lineage"),
    GateType.EVAL_SET_FREEZE: ("coverage_analysis", "leakage_check"),
    GateType.TRAINING_RUN_COST: ("design_memo", "cost_estimate", "success_criteria"),
    GateType.MODEL_PROMOTION: ("eval_results_vs_baseline", "model_card", "red_team_results"),
    GateType.PRODUCTION_DEPLOYMENT: ("load_test", "rollback_plan", "access_controls"),
    GateType.BUDGET_INCREASE: ("spend_report", "justification"),
    GateType.POLICY_DECISION: ("decision_rationale", "legal_review"),
}


class GateError(RuntimeError):
    pass


@dataclass(slots=True)
class GateCatalogEntry:
    gate_type: GateType
    approvers: tuple[str, ...]
    evidence: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_type": str(self.gate_type),
            "approvers": list(self.approvers),
            "evidence": list(self.evidence),
            "description": self.description,
        }


GATE_CATALOG: dict[GateType, GateCatalogEntry] = {
    gate_type: GateCatalogEntry(
        gate_type=gate_type,
        approvers=approvers,
        evidence=EVIDENCE_REQUIREMENTS[gate_type],
        description=gate_type.replace("_", " "),
    )
    for gate_type, approvers in HUMAN_APPROVER_ROLES.items()
}


class GateService:
    def __init__(self, store: Store) -> None:
        self.store = store

    def catalog(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in GATE_CATALOG.values()]

    def request(
        self,
        gate_type: GateType,
        title: str,
        subject: str,
        requested_by: str,
        requested_by_kind: str = "agent",
        evidence: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
    ) -> GateRequest:
        entry = GATE_CATALOG[GateType(gate_type)]
        gate = GateRequest(
            id=new_id("gate"),
            gate_type=GateType(gate_type),
            title=title,
            subject=subject,
            requested_by=requested_by,
            requested_by_kind=requested_by_kind,
            required_approvers=entry.approvers,
            evidence=evidence or {},
            status=GateStatus.PENDING,
            created_at=now_iso(),
            cost_usd=cost_usd,
        )
        self.store.create_gate(gate)
        return gate

    def approver_roles(self, approver: str) -> set[str]:
        """Roles held by an approver.

        A gate names the roles it needs. An approval is given by a person, so
        the person's roles are resolved and the approval satisfies a gate when
        the person's roles cover the required ones. A raw role name is also
        accepted for programmatic callers.
        """
        from internal_ai.people import find_person

        try:
            person = find_person(approver)
        except KeyError:
            return {approver}
        return set(person.roles)

    def decide(
        self,
        gate_id: str,
        approver: str,
        approved: bool,
        approver_kind: str = "human",
        note: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> GateRequest:
        gate = self.store.get_gate(gate_id)
        if gate is None:
            raise GateError(f"unknown gate: {gate_id}")
        if gate.status != GateStatus.PENDING:
            raise GateError(f"gate {gate_id} is already {gate.status}")
        if approver_kind == "agent":
            raise GateError("gates may only be decided by humans (dual control)")
        if approver == gate.requested_by:
            raise GateError(
                f"dual control violation: {approver} requested gate {gate_id} and cannot approve it"
            )
        missing = [
            item
            for item in EVIDENCE_REQUIREMENTS[GateType(gate.gate_type)]
            if item not in gate.evidence
        ]
        if missing:
            raise GateError(f"missing required evidence: {', '.join(missing)}")
        decision = GateDecision(
            id=new_id("dec"),
            gate_id=gate_id,
            approver=approver,
            approver_kind=approver_kind,
            approved=approved,
            note=note,
            created_at=now_iso(),
        )
        self.store.add_gate_decision(decision)
        approved_roles: set[str] = set()
        for item in self.store.gate_decisions(gate_id):
            if item.approved:
                approved_roles |= self.approver_roles(item.approver)
        if not approved:
            gate.status = GateStatus.REJECTED
            gate.decided_at = now_iso()
            gate.decision_note = note
        elif set(gate.required_approvers) <= approved_roles:
            gate.status = GateStatus.APPROVED
            gate.decided_at = now_iso()
            gate.decision_note = note
        if evidence:
            gate.evidence.update(evidence)
        self.store.update_gate(gate)
        return gate

    def cancel(self, gate_id: str, actor: str, note: str = "") -> GateRequest:
        gate = self.store.get_gate(gate_id)
        if gate is None:
            raise GateError(f"unknown gate: {gate_id}")
        if gate.status != GateStatus.PENDING:
            raise GateError(f"gate {gate_id} is already {gate.status}")
        gate.status = GateStatus.CANCELLED
        gate.decided_at = now_iso()
        gate.decision_note = note or f"cancelled by {actor}"
        return self.store.update_gate(gate)

    def is_approved(self, gate_id: str) -> bool:
        gate = self.store.get_gate(gate_id)
        return bool(gate and gate.status == GateStatus.APPROVED)

    def pending(self) -> list[GateRequest]:
        return self.store.list_gates(status=str(GateStatus.PENDING))
