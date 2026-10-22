"""Alerting for agents, spend and data access (PRD section 15).

Alerts are derived from the same state the orchestrator acts on, so a reviewer
sees the signal before a decision is made rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from llmforge.audit import AuditLog
from llmforge.budget import BudgetManager
from llmforge.domain import AuditOutcome
from llmforge.ids import new_id, now_iso
from llmforge.store import Store

SPEND_WARNING_RATIO = 0.8
ACCESS_ANOMALY_RATIO = 3.0
ACCESS_ANOMALY_MINIMUM = 20


@dataclass(slots=True)
class Alert:
    id: str
    kind: str
    severity: str
    subject: str
    message: str
    created_at: str
    details: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "severity": self.severity,
            "subject": self.subject,
            "message": self.message,
            "created_at": self.created_at,
            "details": self.details,
            "acknowledged": self.acknowledged,
        }


class AlertEngine:
    """Raises alerts on spend pressure, failing tasks and odd data access."""

    def __init__(self, store: Store, audit: AuditLog, budget: BudgetManager) -> None:
        self.store = store
        self.audit = audit
        self.budget = budget

    def raise_alert(
        self,
        kind: str,
        severity: str,
        subject: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Alert:
        alert = Alert(
            id=new_id("alr"),
            kind=kind,
            severity=severity,
            subject=subject,
            message=message,
            created_at=now_iso(),
            details=details or {},
        )
        self.store.conn.execute(
            """
            INSERT INTO alerts (id, kind, severity, subject, message, created_at,
            details, acknowledged) VALUES (?,?,?,?,?,?,?,0)
            """,
            (
                alert.id,
                alert.kind,
                alert.severity,
                alert.subject,
                alert.message,
                alert.created_at,
                _json(alert.details),
            ),
        )
        self.audit.record(
            actor="alert_engine",
            actor_kind="system",
            action=f"alert.{kind}",
            subject=subject,
            outcome=AuditOutcome.FAILURE
            if severity in ("critical", "high")
            else AuditOutcome.REQUESTED,
            details={"message": message, "severity": severity, **alert.details},
        )
        return alert

    def list_alerts(self, unacknowledged_only: bool = False) -> list[Alert]:
        query = "SELECT * FROM alerts"
        if unacknowledged_only:
            query += " WHERE acknowledged = 0"
        query += " ORDER BY created_at DESC, id DESC"
        return [
            Alert(
                id=row["id"],
                kind=row["kind"],
                severity=row["severity"],
                subject=row["subject"],
                message=row["message"],
                created_at=row["created_at"],
                details=_json_loads(row["details"]),
                acknowledged=bool(row["acknowledged"]),
            )
            for row in self.store.conn.execute(query).fetchall()
        ]

    def acknowledge(self, alert_id: str, actor: str) -> Alert:
        self.store.conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
        self.audit.record(
            actor=actor,
            actor_kind="human",
            action="alert.acknowledged",
            subject=alert_id,
            details={},
        )
        alerts = [item for item in self.list_alerts() if item.id == alert_id]
        if not alerts:
            raise KeyError(f"unknown alert: {alert_id}")
        return alerts[0]

    # ------------------------------------------------------------- detectors

    def check_spend(self) -> list[Alert]:
        alerts: list[Alert] = []
        for status in self.budget.all_status():
            if status.limit_usd <= 0:
                continue
            if status.exceeded:
                alerts.append(
                    self.raise_alert(
                        "budget_exhausted",
                        "critical",
                        f"{status.scope}:{status.key}",
                        f"spend {status.spent_usd:.2f} USD exceeds the cap of "
                        f"{status.limit_usd:.2f} USD",
                        {"spent_usd": status.spent_usd, "limit_usd": status.limit_usd},
                    )
                )
            elif status.utilization >= SPEND_WARNING_RATIO:
                alerts.append(
                    self.raise_alert(
                        "spend_approaching_cap",
                        "medium",
                        f"{status.scope}:{status.key}",
                        f"spend {status.spent_usd:.2f} USD is {status.utilization:.0%} "
                        f"of the {status.limit_usd:.2f} USD cap",
                        {"spent_usd": status.spent_usd, "limit_usd": status.limit_usd},
                    )
                )
        return alerts

    def check_tasks(self, attempt_threshold: int = 3) -> list[Alert]:
        alerts: list[Alert] = []
        for task in self.store.list_tasks():
            if task.attempts >= attempt_threshold and task.state not in ("accepted", "rejected"):
                alerts.append(
                    self.raise_alert(
                        "repeated_task_failures",
                        "high",
                        task.id,
                        f"task {task.title} has been attempted {task.attempts} times",
                        {"attempts": task.attempts, "state": str(task.state)},
                    )
                )
        return alerts

    def check_data_access(self) -> list[Alert]:
        """Flag a role reading far more records than its peers.

        Each actor is compared against the median of the other actors, so a
        single heavy reader is visible even when there are only two.
        """
        rows = self.store.conn.execute(
            """
            SELECT actor, COUNT(*) AS reads, COUNT(DISTINCT subject) AS subjects
            FROM audit_events
            WHERE action = 'serving.search'
            GROUP BY actor
            """
        ).fetchall()
        if len(rows) < 2:
            return []
        counts = {row["actor"]: int(row["reads"]) for row in rows}
        alerts: list[Alert] = []
        for actor, reads in counts.items():
            if reads < ACCESS_ANOMALY_MINIMUM:
                continue
            peers = [value for name, value in counts.items() if name != actor]
            if not peers:
                continue
            peer_median = median(peers)
            if peer_median <= 0:
                continue
            if reads > peer_median * ACCESS_ANOMALY_RATIO:
                alerts.append(
                    self.raise_alert(
                        "anomalous_data_access",
                        "high",
                        actor,
                        f"{actor} issued {reads} data reads against a peer median of {peer_median}",
                        {"reads": reads, "peer_median": peer_median},
                    )
                )
        return alerts

    def check_all(self) -> list[Alert]:
        return [*self.check_spend(), *self.check_tasks(), *self.check_data_access()]


def _json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, default=str)


def _json_loads(value: str) -> dict[str, Any]:
    import json

    try:
        return json.loads(value) if value else {}
    except json.JSONDecodeError:
        return {}
