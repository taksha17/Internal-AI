"""Policy decisions P1-P7.

The PRD marks these as required before Week 1 ends because each one materially
changes scope. Defaults are the PRD's stated defaults; the platform runs on
defaults until a human changes them, and every change is gated and audited.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from internal_ai.domain import GateType, PolicyDecision
from internal_ai.gates import GateService
from internal_ai.ids import now_iso
from internal_ai.services.attributes import AttributePolicyError, ReadPolicy
from internal_ai.store import Store


@dataclass(frozen=True, slots=True)
class PolicyItem:
    id: str
    question: str
    default_value: str
    rationale: str


POLICY_ITEMS: tuple[PolicyItem, ...] = (
    PolicyItem(
        id="P1",
        question="Are public or licensed datasets permitted alongside internal data?",
        default_value="internal_data_only",
        rationale=(
            "Internal data alone is likely only billions of tokens. Without outside data, "
            "models stay narrow."
        ),
    ),
    PolicyItem(
        id="P2",
        question="May agents run on third-party LLM APIs as development tooling?",
        default_value="allowed_for_code_and_config_no_raw_data",
        rationale=(
            "Agents need an LLM. If banned, agents need a local model, which conflicts with "
            "the no-base-model rule."
        ),
    ),
    PolicyItem(
        id="P3",
        question=(
            "Is implementing a published architecture (e.g. a Llama-style transformer) with "
            "random initialization acceptable?"
        ),
        default_value="acceptable_pending_legal",
        rationale="Architecture is published research, not weights. Legal must confirm.",
    ),
    PolicyItem(
        id="P4",
        question=(
            "Is an audited fully open model (published data and code) ever an acceptable exception?"
        ),
        default_value="no",
        rationale="Would shorten the path to a capable assistant to weeks.",
    ),
    PolicyItem(
        id="P5",
        question="What data classes may be used for training (PII, customer, regulated)?",
        default_value="exclude_pii_customer_confidential_regulated",
        rationale="Defines the scrubbing and exclusion scope.",
    ),
    PolicyItem(
        id="P6",
        question="May this install load a pretrained open base to adapt?",
        default_value="scratch_only",
        rationale=(
            "The default keeps the no-pretrained-weights promise. open_weight_allowed "
            "lets the adapter track load a hash-pinned, licence-allowlisted base whose "
            "name, revision and licence appear on every model card."
        ),
    ),
    PolicyItem(
        id="P7",
        question=(
            "Which data sensitivity classes may a person read, and which groups may read "
            "each? Write class=group|group, or class=anyone."
        ),
        default_value="internal=anyone",
        rationale=(
            "Attributes are checked with the capability, so this value decides what the "
            "console and the served answers may show. A class left out is unreadable by "
            "everyone, and a value nobody can read grants nothing at all."
        ),
    ),
)

POLICY_BY_ID = {item.id: item for item in POLICY_ITEMS}

P7_POLICY_ID = "P7"

POLICY_VALIDATORS: dict[str, Callable[[str], object]] = {P7_POLICY_ID: ReadPolicy.parse}
"""Items whose value is a rule rather than a word, checked before a gate opens."""


class PolicyRegistry:
    def __init__(self, store: Store, gates: GateService) -> None:
        self.store = store
        self.gates = gates

    def seed(self) -> list[PolicyDecision]:
        existing = {item.id for item in self.store.list_policy()}
        seeded: list[PolicyDecision] = []
        for item in POLICY_ITEMS:
            if item.id in existing:
                continue
            decision = PolicyDecision(
                id=item.id,
                question=item.question,
                default_value=item.default_value,
                current_value=item.default_value,
                rationale=item.rationale,
                status="undecided",
            )
            self.store.upsert_policy(decision)
            seeded.append(decision)
        return seeded

    def list(self) -> list[PolicyDecision]:
        self.seed()
        return self.store.list_policy()

    def get(self, policy_id: str) -> PolicyDecision:
        for decision in self.list():
            if decision.id == policy_id:
                return decision
        raise KeyError(f"unknown policy decision: {policy_id}")

    def effective(self, policy_id: str) -> str:
        return self.get(policy_id).current_value

    def propose(
        self,
        policy_id: str,
        value: str,
        requested_by: str,
        rationale: str = "",
    ) -> tuple[PolicyDecision, str]:
        """Open a gate for a new value. The value applies when the gate is decided.

        Recording a proposal must not change the thing it proposes. A pending
        proposal that already applies is not a gate at all: for a policy that
        governs whether an install may load pretrained weights it would let one
        person change behaviour while the approvers still had it open. The
        proposed value and the gate id are kept beside the decision, so a
        decision confirmed later cannot be confirmed against the wrong gate or
        invented from nothing.
        """
        decision = self.get(policy_id)
        validate = POLICY_VALIDATORS.get(policy_id)
        if validate is not None:
            # A value nobody can read must not reach a gate: approving it would look
            # like a decision and enforce nothing.
            validate(value)
        gate = self.gates.request(
            GateType.POLICY_DECISION,
            title=f"Policy decision {policy_id}: {decision.question}",
            subject=policy_id,
            requested_by=requested_by,
            requested_by_kind="human",
            evidence={
                "decision_rationale": rationale or decision.rationale,
                "legal_review": "pending",
                "proposed_value": value,
            },
        )
        self.store.kv_set(f"policy_proposed:{policy_id}", value)
        self.store.kv_set(f"policy_gate:{policy_id}", gate.id)
        decision.status = "pending_approval"
        self.store.upsert_policy(decision)
        return decision, gate.id

    def confirm(
        self, policy_id: str, decided_by: str, gate_id: str | None = None
    ) -> PolicyDecision:
        decision = self.get(policy_id)
        if gate_id is None:
            gate_id = self.store.kv_get(f"policy_gate:{policy_id}")
        if not gate_id:
            raise ValueError(f"no gate recorded for policy decision {policy_id}")
        if not self.gates.is_approved(gate_id):
            raise ValueError(f"gate {gate_id} for {policy_id} is not approved")
        proposed = self.store.kv_get(f"policy_proposed:{policy_id}") or ""
        if not proposed:
            raise ValueError(
                f"policy decision {policy_id} has no pending proposal; propose a value first"
            )
        decision.current_value = proposed
        decision.status = "decided"
        decision.decided_by = decided_by
        decision.decided_at = now_iso()
        self.store.upsert_policy(decision)
        self.store.kv_set(f"policy_proposed:{policy_id}", "")
        return decision

    def status_summary(self) -> dict[str, Any]:
        decisions = self.list()
        undecided = [item.id for item in decisions if item.status != "decided"]
        return {
            "total": len(decisions),
            "decided": len(decisions) - len(undecided),
            "undecided": undecided,
            "decisions": {
                item.id: {
                    "value": item.current_value,
                    "status": item.status,
                    "decided_by": item.decided_by,
                }
                for item in decisions
            },
        }


def read_policy(registry: PolicyRegistry) -> ReadPolicy:
    """The grants the decided P7 carries, for a check that reads a record.

    A value that cannot be parsed grants nothing, internal material included:
    the platform keeps answering refusals while a person corrects the decision,
    which is the only safe direction for a rule nobody can read.
    """
    try:
        return ReadPolicy.parse(registry.effective(P7_POLICY_ID))
    except AttributePolicyError:
        return ReadPolicy.denied()
