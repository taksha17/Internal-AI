"""The metrics snapshot counts what the record already says, and nothing more."""

from __future__ import annotations

import pytest

from internal_ai.harness import (
    BackendChoice,
    BudgetRefused,
    Harness,
    HarnessRequest,
    ModelSignature,
)
from internal_ai.services.metrics import operational_metrics, percentile


def test_percentile_is_nearest_rank_and_an_empty_window_is_zero() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0


def test_a_snapshot_counts_answers_cost_latency_and_the_kill_switch(platform) -> None:  # type: ignore[no-untyped-def]
    harness = Harness(
        BackendChoice("scripted", platform.runtime.llm, False),
        ModelSignature("scripted", 8192),
        budget=platform.budget,
        audit=platform.audit,
    )
    harness.answer(HarnessRequest(question="where?"))
    snapshot = operational_metrics(platform)
    assert snapshot["harness"]["answers"] == 1
    assert snapshot["harness"]["cost_usd"] > 0
    assert snapshot["harness"]["latency_ms"]["p95"] >= 0
    assert snapshot["harness"]["tool_calls"] == 0
    assert snapshot["tools"]["calls"] == 0
    assert snapshot["budget"]["kill_switch"] is False
    assert snapshot["truncated"] is False


def test_a_denied_call_is_counted_as_denied_not_as_a_pass(platform) -> None:  # type: ignore[no-untyped-def]
    platform.budget.engage_kill_switch()
    harness = Harness(
        BackendChoice("scripted", platform.runtime.llm, False),
        ModelSignature("scripted", 8192),
        budget=platform.budget,
        audit=platform.audit,
    )
    with pytest.raises(BudgetRefused):
        harness.answer(HarnessRequest(question="where?"))
    snapshot = operational_metrics(platform)
    assert snapshot["harness"]["answers"] == 0
    assert snapshot["harness"]["denied"] == 1
    assert snapshot["harness"]["passed"] == 0
    assert snapshot["audit"]["denied"] >= 1
