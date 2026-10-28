"""The project's target, metrics, constraints and owners, in one record.

The acceptance checklist asks a company for these before any training starts, and
they used to live in prose that nothing could check. One record, read by the
overview and the plan, makes the gaps visible where the work begins: a plan
without a target is a list of tasks, and a run without a metric is an experiment
nobody can judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.ids import now_iso
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require
from internal_ai.store import Store

OWNER_ROLES = ("ml_lead", "data_engineer", "infra", "evaluation", "security")
DIRECTIONS = ("at_least", "at_most")


@dataclass(slots=True)
class Metric:
    name: str = ""
    threshold: str = ""
    direction: str = "at_least"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "threshold": self.threshold, "direction": self.direction}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Metric:
        direction = str(payload.get("direction", "at_least"))
        return cls(
            name=str(payload.get("name", "")),
            threshold=str(payload.get("threshold", "")),
            direction=direction if direction in DIRECTIONS else "at_least",
        )


@dataclass(slots=True)
class ProjectBrief:
    target: str = ""
    languages: str = ""
    context_length: int = 0
    quality_bar: str = ""
    metrics: list[Metric] = field(default_factory=list)
    budget_usd: float = 0.0
    timeline: str = ""
    deployment: str = ""
    data_restrictions: str = ""
    latency_ms: int = 0
    owners: dict[str, str] = field(default_factory=dict)
    non_goals: str = ""
    updated_at: str = ""
    updated_by: str = ""

    @property
    def missing(self) -> list[str]:
        """What a person still has to write, named as they would say it."""
        gaps: list[str] = []
        if not (self.target and self.languages and self.context_length > 0 and self.quality_bar):
            gaps.append("the target, languages, context length and quality bar")
        if not any(metric.name and metric.threshold for metric in self.metrics):
            gaps.append("a success metric with a threshold")
        if not (self.budget_usd > 0 and self.timeline and self.deployment and self.latency_ms > 0):
            gaps.append("the budget, timeline, deployment location and latency target")
        unowned = [role for role in OWNER_ROLES if not self.owners.get(role)]
        if unowned:
            gaps.append("owners for " + ", ".join(unowned))
        if not self.non_goals:
            gaps.append("what the model will not do")
        return gaps

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "languages": self.languages,
            "context_length": self.context_length,
            "quality_bar": self.quality_bar,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "budget_usd": self.budget_usd,
            "timeline": self.timeline,
            "deployment": self.deployment,
            "data_restrictions": self.data_restrictions,
            "latency_ms": self.latency_ms,
            "owners": dict(self.owners),
            "non_goals": self.non_goals,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectBrief:
        owners = {str(key): str(value) for key, value in dict(payload.get("owners") or {}).items()}
        return cls(
            target=str(payload.get("target", "")),
            languages=str(payload.get("languages", "")),
            context_length=int(payload.get("context_length") or 0),
            quality_bar=str(payload.get("quality_bar", "")),
            metrics=[Metric.from_dict(item) for item in payload.get("metrics") or []],
            budget_usd=float(payload.get("budget_usd") or 0.0),
            timeline=str(payload.get("timeline", "")),
            deployment=str(payload.get("deployment", "")),
            data_restrictions=str(payload.get("data_restrictions", "")),
            latency_ms=int(payload.get("latency_ms") or 0),
            owners=owners,
            non_goals=str(payload.get("non_goals", "")),
            updated_at=str(payload.get("updated_at", "")),
            updated_by=str(payload.get("updated_by", "")),
        )


def load(store: Store) -> ProjectBrief:
    row = store.conn.execute(
        "SELECT payload, updated_at, updated_by FROM project_brief WHERE id = 'brief'"
    ).fetchone()
    if row is None:
        return ProjectBrief()
    payload = json.loads(row["payload"] or "{}")
    payload["updated_at"] = row["updated_at"]
    payload["updated_by"] = row["updated_by"]
    return ProjectBrief.from_dict(payload)


def save(store: Store, audit: AuditLog, actor: Actor, brief: ProjectBrief) -> ProjectBrief:
    require(audit, actor, Action.WRITE_BRIEF, "project_brief")
    stamp = now_iso()
    store.conn.execute(
        "INSERT INTO project_brief (id, payload, updated_at, updated_by) VALUES ('brief', ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET payload = excluded.payload,"
        " updated_at = excluded.updated_at, updated_by = excluded.updated_by",
        (json.dumps(brief.to_dict(), sort_keys=True), stamp, actor.id),
    )
    audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="brief.updated",
        subject="project_brief",
        details={"complete": brief.complete, "missing": brief.missing},
    )
    return load(store)
