"""Operational metrics: what an operator reads, derived from the record itself.

Every number here is counted from the audit log and the spend ledger rather than
held in memory, so a restart does not lose the window and a second process reads
the same answer. Nothing in this module writes: it is a read-only view of what
already happened, which is what makes it safe to expose on an authenticated
endpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import AuditEvent, AuditOutcome
from internal_ai.ids import now_iso

DEFAULT_WINDOW_HOURS = 24
MAX_EVENTS = 20000
"""A snapshot stops at this many events rather than reading a whole year of log."""


def percentile(values: Sequence[float], fraction: float) -> float:
    """A nearest-rank percentile, and ``0.0`` for a window with nothing in it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def window_start(hours: int) -> str:
    return (datetime.now(UTC) - timedelta(hours=max(hours, 1))).isoformat(timespec="seconds")


def _details(event: AuditEvent) -> dict[str, Any]:
    return event.details if isinstance(event.details, dict) else {}


def _count(events: Sequence[AuditEvent], outcome: AuditOutcome) -> int:
    return sum(1 for event in events if str(event.outcome) == str(outcome))


def operational_metrics(
    platform: Platform, window_hours: int = DEFAULT_WINDOW_HOURS
) -> dict[str, Any]:
    """One snapshot: volume, refusals, harness cost and latency, tool use, spend."""
    since = window_start(window_hours)
    events = platform.store.list_audit(since=since, limit=MAX_EVENTS)
    harness = [event for event in events if event.action == "harness.answer"]
    tools = [event for event in events if event.action == "mcp.tool"]
    latencies = [float(_details(event).get("latency_ms") or 0) for event in harness]
    return {
        "window_hours": window_hours,
        "since": since,
        "generated_at": now_iso(),
        "truncated": len(events) >= MAX_EVENTS,
        "audit": {
            "events": len(events),
            "denied": _count(events, AuditOutcome.DENIED),
            "failures": _count(events, AuditOutcome.FAILURE),
        },
        "harness": {
            "answers": sum(
                1 for event in harness if str(event.outcome) != str(AuditOutcome.DENIED)
            ),
            "passed": sum(1 for event in harness if _details(event).get("passed")),
            "exhausted": sum(1 for event in harness if _details(event).get("exhausted")),
            "denied": _count(harness, AuditOutcome.DENIED),
            "cost_usd": round(sum(event.cost_usd for event in harness), 8),
            "tool_calls": sum(int(_details(event).get("tool_calls") or 0) for event in harness),
            "latency_ms": {
                "p50": percentile(latencies, 0.5),
                "p95": percentile(latencies, 0.95),
            },
        },
        "tools": {
            "calls": len(tools),
            "denied": _count(tools, AuditOutcome.DENIED),
            "failures": _count(tools, AuditOutcome.FAILURE),
        },
        "budget": {
            "kill_switch": platform.budget.kill_switch_engaged(),
            "status": [status.to_dict() for status in platform.budget.all_status()],
        },
    }
