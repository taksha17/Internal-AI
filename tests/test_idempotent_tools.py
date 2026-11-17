"""A side-effecting tool call belongs to a task, so a retry of that task reuses it.

The failure this prevents: a step is interrupted after `write_artifact` ran, the task
is queued again, and the agent's second attempt performs the same write a second time
-- or, worse, a different one that the interrupted attempt had already superseded.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from internal_ai.roles import get_role
from internal_ai.tools import CURRENT_TASK, ToolDenied


def args_hash(args: dict[str, object]) -> str:
    body = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def actions(platform) -> list[str]:
    rows = platform.store.conn.execute(
        "SELECT action FROM audit_events WHERE action LIKE 'tool.%' ORDER BY seq"
    ).fetchall()
    return [row["action"] for row in rows]


def test_the_same_write_in_one_task_runs_once(platform) -> None:
    role = get_role("orchestrator")
    args = {"name": "note", "content": "the pump primes in four strokes"}
    token = CURRENT_TASK.set("tsk_fixed")
    try:
        first = platform.tools.invoke(role, "write_artifact", args)
        second = platform.tools.invoke(role, "write_artifact", args)
    finally:
        CURRENT_TASK.reset(token)

    assert first == second
    assert actions(platform) == ["tool.write_artifact", "tool.replayed.write_artifact"]
    row = platform.store.tool_call("tsk_fixed", "write_artifact", args_hash(args))
    assert row["state"] == "done"


def test_a_different_argument_is_a_different_call(platform) -> None:
    role = get_role("orchestrator")
    token = CURRENT_TASK.set("tsk_fixed")
    try:
        platform.tools.invoke(role, "write_artifact", {"name": "note", "content": "one"})
        platform.tools.invoke(role, "write_artifact", {"name": "note", "content": "two"})
    finally:
        CURRENT_TASK.reset(token)

    assert actions(platform) == ["tool.write_artifact", "tool.write_artifact"]


def test_a_read_is_never_replayed(platform) -> None:
    """A read has to see what is there now, so it is not remembered or reused."""
    role = get_role("orchestrator")
    written = platform.tools.invoke(role, "write_artifact", {"name": "note", "content": "body"})
    token = CURRENT_TASK.set("tsk_fixed")
    try:
        platform.tools.invoke(role, "read_artifact", {"path": written})
        platform.tools.invoke(role, "read_artifact", {"path": written})
    finally:
        CURRENT_TASK.reset(token)

    assert actions(platform).count("tool.read_artifact") == 2


def test_a_call_left_running_refuses_a_retry(platform) -> None:
    role = get_role("orchestrator")
    args = {"name": "note", "content": "body"}
    platform.store.begin_tool_call(
        call_id="tcl_open",
        task_id="tsk_fixed",
        tool_name="write_artifact",
        args_hash=args_hash(args),
    )
    token = CURRENT_TASK.set("tsk_fixed")
    try:
        with pytest.raises(ToolDenied, match="never finished"):
            platform.tools.invoke(role, "write_artifact", args)
    finally:
        CURRENT_TASK.reset(token)


def test_a_call_outside_a_task_is_not_remembered(platform) -> None:
    role = get_role("orchestrator")

    platform.tools.invoke(role, "write_artifact", {"name": "note", "content": "body"})
    platform.tools.invoke(role, "write_artifact", {"name": "note", "content": "body"})

    assert actions(platform) == ["tool.write_artifact", "tool.write_artifact"]
