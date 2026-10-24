"""The kill switch has to be visible and recoverable.

Engaging the kill switch stops work. If the board does not say why, and if
releasing the switch does not put the work back in the queue, the control
looks broken to the person testing it.
"""

from __future__ import annotations

from internal_ai.domain import AuditOutcome, TaskState
from internal_ai.engine import KILL_SWITCH_REASON
from internal_ai.planner import TemplatePlanner


def _approved_plan(platform) -> None:
    drafts = TemplatePlanner().plan("ship the embedding model", {})
    platform.orchestrator.submit_plan(drafts)
    platform.orchestrator.approve_plan("program_lead")


def test_kill_switch_blocks_tasks_with_a_reason(platform) -> None:
    _approved_plan(platform)
    platform.budget.engage_kill_switch()
    summary = platform.orchestrator.run(max_iterations=5)

    assert summary["kill_switch"] == "engaged"
    assert summary["dispatched"] == 0
    assert summary["spend_usd"] == 0.0

    blocked = platform.store.list_tasks(state=str(TaskState.BLOCKED))
    assert blocked
    assert all(task.blocked_reason == KILL_SWITCH_REASON for task in blocked)

    events = platform.audit.recent(limit=100, action="budget.kill_switch_blocked")
    assert events
    assert all(event.outcome == AuditOutcome.DENIED for event in events)
    assert platform.audit.verify().ok


def test_releasing_the_kill_switch_puts_the_work_back(platform) -> None:
    _approved_plan(platform)
    platform.budget.engage_kill_switch()
    platform.orchestrator.run(max_iterations=5)
    assert platform.store.list_tasks(state=str(TaskState.BLOCKED))

    platform.budget.release_kill_switch()
    platform.orchestrator.release_unblocked()
    assert not platform.store.list_tasks(state=str(TaskState.BLOCKED))

    summary = platform.orchestrator.run(max_iterations=1)
    assert summary["kill_switch"] == "released"
    assert summary["dispatched"] >= 1
