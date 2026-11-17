"""A run that dies mid-dispatch leaves a row that says so, and the next run explains it.

The point is not that the work resumes by itself: it is that nobody has to guess
whether an attempt ran, and that the task it belonged to goes back into the queue
instead of sitting in progress forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.context import Platform
from internal_ai.domain import TaskState
from internal_ai.engine import EngineConfig, Orchestrator
from internal_ai.planner import TaskDraft


def plan() -> list[TaskDraft]:
    return [
        TaskDraft(
            key="a",
            title="Write the brief",
            description="one task",
            kind="document",
            owner_role="orchestrator",
            cost_estimate_usd=0.1,
        )
    ]


def orchestrator(platform: Platform, runtime: object) -> Orchestrator:
    return Orchestrator(
        platform.store,
        platform.audit,
        platform.gates,
        platform.budget,
        runtime,  # type: ignore[arg-type]
        config=EngineConfig(max_iterations=3),
    )


def test_a_dispatch_writes_a_step_that_is_closed_when_it_returns(platform) -> None:
    orchestrator(platform, platform.runtime).submit_plan(plan(), approved_by="owner")

    summary = orchestrator(platform, platform.runtime).run()

    steps = platform.store.steps_for("tsk_a")
    # The scripted runtime can take a few attempts; every one of them is a closed row.
    assert len(steps) >= 1
    assert [step["attempt"] for step in steps] == list(range(1, len(steps) + 1))
    assert all(step["state"] in {"done", "failed"} for step in steps)
    assert all(step["finished_at"] for step in steps)
    assert platform.store.unfinished_steps() == []
    assert summary["resumed"] == 0


class Exploding:
    """A runtime that dies inside the call, which is the case the step row is for."""

    def execute(self, role: object, task: object, upstream: object = ()) -> object:
        raise RuntimeError("the machine went away")


def test_a_crashed_dispatch_leaves_a_failed_step_not_an_open_one(platform) -> None:
    engine = orchestrator(platform, Exploding())
    engine.submit_plan(plan(), approved_by="owner")

    with pytest.raises(RuntimeError, match="machine went away"):
        engine.run()

    steps = platform.store.steps_for("tsk_a")
    assert [step["state"] for step in steps] == ["failed"]
    assert "machine went away" in steps[0]["summary"]
    assert platform.store.unfinished_steps() == []


def test_an_open_step_is_abandoned_and_its_task_queued_again(platform) -> None:
    engine = orchestrator(platform, platform.runtime)
    engine.submit_plan(plan(), approved_by="owner")
    platform.store.set_task_state(
        platform.store.get_task("tsk_a"), TaskState.IN_PROGRESS, "agent", "agent", "started"
    )
    platform.store.open_step(task_id="tsk_a", attempt=1, agent="orchestrator", step_id="stp_open")

    abandoned = engine.resume()

    assert [step["id"] for step in abandoned] == ["stp_open"]
    step = platform.store.step("stp_open")
    assert step["state"] == "abandoned"
    assert platform.store.get_task("tsk_a").state == TaskState.IN_PROGRESS
    events = platform.store.conn.execute(
        "SELECT actor, details FROM audit_events WHERE action = 'plan.step_abandoned'"
    ).fetchall()
    assert [event["actor"] for event in events] == ["orchestrator"]
    assert "stp_open" in events[0]["details"]

    # The task is ready again, so the next run picks it up rather than skipping it.
    assert "tsk_a" in [task.id for task in engine.ready_tasks()]


def test_a_step_for_a_task_that_moved_on_is_closed_without_touching_the_task(platform) -> None:
    engine = orchestrator(platform, platform.runtime)
    engine.submit_plan(plan(), approved_by="owner")
    task = platform.store.get_task("tsk_a")
    platform.store.set_task_state(task, TaskState.IN_PROGRESS, "agent", "agent", "started")
    platform.store.set_task_state(task, TaskState.IN_REVIEW, "agent", "agent", "submitted")
    platform.store.set_task_state(task, TaskState.ACCEPTED, "reviewer", "agent", "accepted")
    platform.store.open_step(task_id="tsk_a", attempt=1, agent="orchestrator", step_id="stp_stale")

    engine.resume()

    assert platform.store.step("stp_stale")["state"] == "abandoned"
    assert platform.store.get_task("tsk_a").state == TaskState.ACCEPTED


def test_the_run_reports_what_it_resumed(platform) -> None:
    engine = orchestrator(platform, platform.runtime)
    engine.submit_plan(plan(), approved_by="owner")
    platform.store.set_task_state(
        platform.store.get_task("tsk_a"), TaskState.IN_PROGRESS, "agent", "agent", "started"
    )
    platform.store.open_step(task_id="tsk_a", attempt=0, agent="orchestrator", step_id="stp_x")

    summary = engine.run()

    assert summary["resumed"] == 1


def test_a_resumed_plan_is_still_one_run_in_the_audit_log(platform, tmp_path: Path) -> None:
    engine = orchestrator(platform, platform.runtime)
    engine.submit_plan(plan(), approved_by="owner")
    engine.run()
    runs = platform.store.conn.execute(
        "SELECT count(*) AS n FROM audit_events WHERE action = 'orchestrator.run'"
    ).fetchone()
    assert runs["n"] == 1
