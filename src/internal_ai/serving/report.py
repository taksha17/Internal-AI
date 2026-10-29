"""Daily numbers for serving, read from the audit log.

The brief already names a latency target. This report says whether the requests
that were actually served stayed under it, what they cost, and how many are
older than the brief's timeline. The audit table cannot be deleted from, so
``--apply`` records a cutoff and later reports stop counting those rows.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from internal_ai.brief import load
from internal_ai.context import Platform
from internal_ai.domain import AuditEvent
from internal_ai.ids import now_iso

_SPAN = re.compile(r"(\d+)\s*(day|week|month)s?", re.IGNORECASE)
_UNITS = {"day": 1, "week": 7, "month": 30}
RETENTION_KEY = "serving_retention_before"


def retention_days(timeline: str) -> int | None:
    match = _SPAN.search(timeline or "")
    if match is None:
        return None
    return int(match.group(1)) * _UNITS[match.group(2).lower()]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _kept(platform: Platform) -> list[AuditEvent]:
    cutoff = platform.store.kv_get(RETENTION_KEY, "") or ""
    events = platform.store.list_audit(action="serving.request", limit=100_000, descending=False)
    if not cutoff:
        return events
    return [event for event in events if event.created_at >= cutoff]


def serve_report(platform: Platform, apply: bool = False) -> dict[str, Any]:
    """p50, p95, cost per 1,000 requests, and how many rows sit past retention."""
    brief = load(platform.store)
    events = _kept(platform)
    latencies = [float(event.details.get("latency_ms") or 0) for event in events]
    costs = [float(event.cost_usd) for event in events]
    days = retention_days(brief.timeline)
    past: list[AuditEvent] = []
    cutoff = ""
    if days is not None and events:
        horizon = datetime.now(UTC) - timedelta(days=days)
        cutoff = horizon.isoformat()
        past = [event for event in events if _stamp(event.created_at) < horizon]
    if apply and cutoff:
        platform.store.kv_set(RETENTION_KEY, cutoff)
        platform.audit.record(
            actor="operator",
            actor_kind="human",
            action="serving.retention_applied",
            subject="serving.request",
            details={"dropped": len(past), "before": cutoff},
        )
        events = [event for event in events if event not in past]
        latencies = [float(event.details.get("latency_ms") or 0) for event in events]
        costs = [float(event.cost_usd) for event in events]
    mean_cost = sum(costs) / len(costs) if costs else 0.0
    p95 = _percentile(latencies, 0.95)
    return {
        "requests": len(events),
        "p50_ms": round(_percentile(latencies, 0.50), 3),
        "p95_ms": round(p95, 3),
        "latency_target_ms": brief.latency_ms,
        "within_brief": bool(brief.latency_ms) and p95 <= brief.latency_ms,
        "cost_per_1000": round(1000 * mean_cost, 6),
        "past_retention": 0 if apply else len(past),
        "applied": apply,
        "reported_at": now_iso(),
    }


def _stamp(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
