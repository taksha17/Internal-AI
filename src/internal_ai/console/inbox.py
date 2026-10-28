"""What is waiting, on one page and on the rail.

The rail badge and the Inbox page read this module, so they cannot disagree
about what "open" means. Only work a person can act on now is counted: a gate
they may decide, an alert nobody has acknowledged, and a job that failed in the
last week. Older failures stay on Jobs, where they can still be read; a badge
nobody can clear is noise rather than work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from internal_ai.alerts import Alert, AlertEngine
from internal_ai.context import Platform
from internal_ai.domain import GateRequest, GateStatus
from internal_ai.services import governance
from internal_ai.services.actor import Actor
from internal_ai.services.jobs import JobRunner

FAILED_WINDOW_DAYS = 7
FAILED_SCAN = 50
FAILED_SHOWN = 8


@dataclass(slots=True)
class Waiting:
    """Open work, already split by who has to act on it."""

    yours: list[GateRequest]
    blocked: list[GateRequest]
    alerts: list[Alert]
    failed: list[dict[str, Any]]
    failed_total: int = 0
    older_failures: int = 0

    @property
    def count(self) -> int:
        return len(self.yours) + len(self.alerts) + self.failed_total

    @property
    def empty(self) -> bool:
        return self.count == 0 and not self.blocked


def collect(platform: Platform, actor: Actor, jobs: JobRunner) -> Waiting:
    yours: list[GateRequest] = []
    blocked: list[GateRequest] = []
    for gate in platform.store.list_gates(status=str(GateStatus.PENDING)):
        ok, _why = governance.can_decide(platform, actor, gate)
        (yours if ok else blocked).append(gate)
    alerts = AlertEngine(platform.store, platform.audit, platform.budget).list_alerts(
        unacknowledged_only=True
    )
    failures = jobs.list(limit=FAILED_SCAN, state="failed")
    recent = sorted(
        (job for job in failures if _recent(job.get("finished_at") or job.get("created_at"))),
        key=lambda job: job.get("finished_at") or job.get("created_at") or "",
        reverse=True,
    )
    return Waiting(
        yours=sorted(yours, key=lambda gate: gate.created_at),
        blocked=sorted(blocked, key=lambda gate: gate.created_at),
        alerts=alerts,
        failed=recent[:FAILED_SHOWN],
        failed_total=len(recent),
        older_failures=max(jobs.count(state="failed") - len(recent), 0),
    )


def _recent(stamp: Any, days: int = FAILED_WINDOW_DAYS) -> bool:
    if not stamp:
        return False
    try:
        moment = datetime.fromisoformat(str(stamp))
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return datetime.now(UTC) - moment <= timedelta(days=days)
