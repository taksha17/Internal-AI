"""The queue of failed production answers, read by someone allowed to see it.

Reading a trace is reading what a colleague asked and what the model answered,
so it needs a capability of its own rather than riding on any signed-in person.
The storage layer already filters by the trace's ACL; this adds the role check
and leaves the two independent, because a person with the role still must not
read another group's traces.
"""

from __future__ import annotations

from typing import Any

from internal_ai.context import Platform
from internal_ai.evals.harness import EvalHarness
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require
from internal_ai.serving.trace import Trace, get_trace, triage


def triage_view(platform: Platform, actor: Actor, limit: int = 25) -> dict[str, list[Trace]]:
    """Failed traces this person may read, bucketed by why they failed."""
    require(platform.audit, actor, Action.VIEW_TRACES, "console/traces")
    return triage(platform.store, actor.principal(), limit=limit)


def eval_case_from_trace(
    platform: Platform,
    actor: Actor,
    trace_id: str,
    name: str = "trace-regressions",
) -> Any:
    """Turn one failed answer into the next version of a regression set.

    The newest version is copied forward and the new case is appended, so a
    frozen set is read but never written: the alternative, mutating the frozen
    set, is the one thing the eval rules forbid. The item keeps the citations the
    answer was supposed to honour, which is the shape the alpha evaluation
    already scores answers against, and its lineage back to the trace id.
    """
    require(platform.audit, actor, Action.CREATE_EVAL_SET, trace_id)
    trace = get_trace(platform.store, actor.principal(), trace_id)
    if not trace.failure:
        raise ValueError("only a failed answer becomes an eval case; this one passed")
    hits = [
        {"path": str(item.get("path") or ""), "line": int(item.get("line") or 0)}
        for item in trace.citations
    ]
    hits = [hit for hit in hits if hit["path"] and hit["line"] > 0]
    if not hits:
        raise ValueError("this trace cites no line, so there is nothing to assert against")
    existing = [item for item in platform.store.list_eval_sets() if item.name == name]
    carried = list(existing[-1].items) if existing else []
    if any(trace_id in (item.get("trace_ids") or []) for item in carried):
        raise ValueError(f"trace {trace_id} is already a case in {name}")
    items = [
        *carried,
        {
            "query": trace.question,
            "hits": hits,
            "kind": "citation",
            "trace_ids": [trace_id],
            "dataset_id": "",
            "failure": trace.failure,
        },
    ]
    version = max((item.version for item in existing), default=0) + 1
    eval_set = EvalHarness(platform.store, platform.audit).create_eval_set(
        name, items, version=version, actor=actor.id
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="eval.case_from_trace",
        subject=eval_set.id,
        details={
            "trace_id": trace_id,
            "name": name,
            "version": version,
            "failure": trace.failure,
            "carried": len(carried),
        },
    )
    return eval_set
