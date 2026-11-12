"""A long-horizon KYC form-filling agent on the repo's own harness.

The agent fills synthetic bank KYC forms from customer document lines. Every
read and every fill goes through `ToolRegistry.invoke`, so the harness's
capability check and audit trail apply to this agent exactly as they apply to
every other agent action — no side channel around them.

The job is selection, not recall: each form's evidence mixes that customer's
document lines with other customers' lines, and the agent ranks a capped
candidate list by cue overlap before extracting the value with the field's
own pattern. Memory carries a customer's first fill forward to their later
forms, which is what makes a thousand-form run long-horizon rather than a
thousand independent guesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from internal_ai.audit import AuditLog
from internal_ai.data.kyc import KYC_FIELDS, KycForm
from internal_ai.domain import Capability, RoleSpec
from internal_ai.evals.metrics import exact_match
from internal_ai.store import Store
from internal_ai.tools import Tool, ToolRegistry

KYC_CANDIDATE_LIMIT = 5

FIELD_CUES: dict[str, tuple[str, ...]] = {
    "full_name": ("holder", "resides", "works", "declares"),
    "date_of_birth": ("born", "birth"),
    "nationality": ("nationality",),
    "id_type": ("passport", "licence", "national-id"),
    "id_number": ("passport", "licence", "national-id", "sample-id"),
    "address": ("resides", "utility", "bill"),
    "phone": ("phone",),
    "email": ("statement", "email"),
    "occupation": ("works", "payslip"),
    "employer": ("payslip", "employer"),
    "annual_income_band": ("income", "payslip"),
    "account_type": ("account", "statement"),
    "source_of_funds": ("funds", "source"),
    "pep_declaration": ("declares", "declaration", "pep"),
}

FIELD_PATTERNS: dict[str, str] = {
    "full_name": r"holder\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+born",
    "date_of_birth": r"(\d{2}/\d{2}/\d{4})",
    "nationality": r"nationality\s+([A-Z][a-z]+)",
    "id_type": r"(SAMPLE-(?:PASSPORT|DRIVING-LICENCE|NATIONAL-ID))",
    "id_number": r"(SAMPLE-ID-\d{6})",
    "address": r"(\d+\s+Test Street,\s+Sampleton\s+ST\d{2})",
    "phone": r"(\+1-555-01\d{2})",
    "email": r"(\S+@example\.test)",
    "occupation": r"works as ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) for",
    "employer": r"for ([A-Z][A-Za-z ]+?(?:Bank|Ltd|Inc))",
    "annual_income_band": r"income band (\d+(?:-\d+)?k\+? TEST)",
    "account_type": r"account (TEST-(?:CURRENT|SAVINGS|BUSINESS))",
    "source_of_funds": r"source of funds ([a-z ]+?) account",
    "pep_declaration": r"declares\s+(.+?\(test\))",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+@.\-]+", text.lower()))


def rank_lines(field_name: str, lines: list[str], limit: int = KYC_CANDIDATE_LIMIT) -> list[str]:
    """The capped candidate list: cue overlap first, original order on ties."""
    cues = set(FIELD_CUES[field_name])
    scored = [(len(cues & _tokens(line)), -index, line) for index, line in enumerate(lines)]
    scored.sort(reverse=True)
    return [line for _, _, line in scored[:limit]]


def extract(field_name: str, lines: list[str]) -> str:
    """The first value the field's own pattern finds in the ranked lines."""
    pattern = re.compile(FIELD_PATTERNS[field_name])
    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return ""


@dataclass
class FillReport:
    forms: int
    fields_scored: int
    field_accuracy: float
    consistency: float
    memory_hits: int
    tool_calls: int
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


class KycAgent:
    """Fills forms through harness tools, remembering each customer."""

    def __init__(self, registry: ToolRegistry, role: RoleSpec, actor: str = "kyc-agent") -> None:
        self.registry = registry
        self.role = role
        self.actor = actor
        self.memory: dict[str, dict[str, str]] = {}
        self.tool_calls = 0
        self.memory_hits = 0
        self.conflicts = 0
        registry.register(
            Tool(
                name="read_evidence",
                capability=Capability.TASK_READ,
                description="Read the document lines filed for one form.",
                handler=self._read_evidence,
            )
        )
        registry.register(
            Tool(
                name="write_field",
                capability=Capability.TASK_WRITE,
                description="Record one filled field for one form.",
                handler=self._write_field,
                side_effecting=True,
            )
        )
        self._evidence: dict[str, list[str]] = {}
        self._filled: dict[str, dict[str, str]] = {}

    def _read_evidence(self, form_id: str) -> list[str]:
        return list(self._evidence.get(form_id, []))

    def _write_field(self, form_id: str, field_name: str, value: str) -> str:
        self._filled.setdefault(form_id, {})[field_name] = value
        return "ok"

    def _invoke(self, tool_name: str, args: dict[str, str | list[str]]) -> object:
        self.tool_calls += 1
        return self.registry.invoke(self.role, tool_name, args, actor=self.actor)

    def fill(self, form: KycForm, lines: list[str]) -> dict[str, str]:
        """Fill one form, reusing memory before reading evidence."""
        self._evidence[form.form_id] = lines
        known = self.memory.setdefault(form.customer_id, {})
        filled: dict[str, str] = {}
        for name in KYC_FIELDS:
            if name in known:
                value = known[name]
                self.memory_hits += 1
            else:
                evidence = self._invoke("read_evidence", {"form_id": form.form_id})
                assert isinstance(evidence, list)
                value = extract(name, rank_lines(name, evidence))
                known[name] = value
            filled[name] = value
            self._invoke(
                "write_field",
                {"form_id": form.form_id, "field_name": name, "value": value},
            )
        return filled

    def run(self, forms: list[KycForm], evidence: dict[str, list[str]]) -> FillReport:
        """Fill every form in order; later forms reuse earlier memory."""
        predictions: list[str] = []
        references: list[str] = []
        first_fill: dict[str, dict[str, str]] = {}
        consistent = 0
        compared = 0
        for form in forms:
            filled = self.fill(form, evidence[form.form_id])
            if form.customer_id in first_fill:
                for name in KYC_FIELDS:
                    compared += 1
                    consistent += filled[name] == first_fill[form.customer_id][name]
            else:
                first_fill[form.customer_id] = dict(filled)
            for name in KYC_FIELDS:
                predictions.append(filled[name])
                references.append(form.values[name])
        return FillReport(
            forms=len(forms),
            fields_scored=len(predictions),
            field_accuracy=exact_match(predictions, references),
            consistency=round(consistent / compared, 6) if compared else 1.0,
            memory_hits=self.memory_hits,
            tool_calls=self.tool_calls,
            predictions=predictions,
            references=references,
        )


def agent_role() -> RoleSpec:
    """The least capability that fills a form: read evidence, write fields."""
    return RoleSpec(
        name="kyc-filler",
        mission="Fill synthetic KYC forms from customer document lines.",
        sub_agents=(),
        inputs=("document lines",),
        outputs=("filled fields",),
        capabilities=frozenset({Capability.TASK_READ, Capability.TASK_WRITE}),
        gates=(),
        kpis=("field accuracy",),
        enabled_by_default=False,
    )


def open_registry(workspace: Path, artifacts: Path) -> ToolRegistry:
    """A registry with an audit log, the same pairing every agent gets."""
    return ToolRegistry(AuditLog(Store(workspace / "kyc-audit.db")), workspace, artifacts)
