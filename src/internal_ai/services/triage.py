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
from internal_ai.data.supervision import ingest_supervision
from internal_ai.domain import DatasetVersion
from internal_ai.evals.harness import EvalHarness
from internal_ai.preferences import file_preference
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require
from internal_ai.serving.trace import Trace, get_trace, triage
from internal_ai.training.alpha.citations import AskHit, score_answer


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


def sft_draft(platform: Platform, actor: Actor, trace_id: str) -> Trace:
    """The trace a person is about to correct, if they may read it at all."""
    require(platform.audit, actor, Action.VIEW_TRACES, trace_id)
    return get_trace(platform.store, actor.principal(), trace_id)


def _supervision_rows(platform: Platform, previous: DatasetVersion | None) -> list[dict[str, str]]:
    if previous is None:
        return []
    rows = []
    for row in platform.store.dataset_records(previous.id):
        meta = row["meta"] or {}
        rows.append(
            {
                "instruction": str(meta.get("instruction", "")),
                "response": str(meta.get("response", "")),
                "authored_by": str(meta.get("authored_by", "")),
                "source_id": str(row["source_id"]),
            }
        )
    return rows


def _corrected_answer(trace: Trace, answer: str) -> str:
    """A correction must be new, and cite a line the question was answered from.

    Both promotions call this before writing anything. An uncited answer that
    slipped through would become a dataset row the next evaluation scores as a
    failure, which reads as the model's mistake rather than the author's.
    """
    if not trace.failure:
        raise ValueError("only a failed answer needs a person's correction; this one passed")
    corrected = answer.strip()
    if not corrected:
        raise ValueError("a supervision example needs the answer a person would have given")
    if corrected == trace.answer.strip():
        raise ValueError("this repeats the failed answer, which is not a correction")
    hits = [
        AskHit(str(item.get("path") or ""), int(item.get("line") or 0)) for item in trace.citations
    ]
    hits = [hit for hit in hits if hit.path and hit.line > 0]
    if not hits:
        raise ValueError("this trace cites no line, so a correction has nothing to check against")
    report = score_answer(corrected, hits)
    if not report.passed:
        raise ValueError(
            "the correction must cite a line this question was answered from: "
            + "; ".join(report.mismatches[:3])
        )
    return corrected


def preference_from_trace(
    platform: Platform, actor: Actor, trace_id: str, answer: str
) -> dict[str, str]:
    """The person's answer preferred over the model's failed one.

    A preference is an opinion, not training data, so it files the same
    ``preference`` row the CLI files: the ``domain_expert`` check and the
    refusal of a model-authored pair are the ones that already existed. The
    trace id is written to the audit log, where ``export_preferences`` reads it
    back to put the lineage on the dataset.
    """
    require(platform.audit, actor, Action.VIEW_TRACES, trace_id)
    trace = get_trace(platform.store, actor.principal(), trace_id)
    corrected = _corrected_answer(trace, answer)
    row = file_preference(
        platform,
        actor,
        question=trace.question,
        answer_a=corrected,
        answer_b=trace.answer.strip(),
        preferred="a",
        source_id=trace_id,
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="preference.from_trace",
        subject=row["id"],
        details={"trace_id": trace_id},
    )
    return row


def sft_example_from_trace(
    platform: Platform,
    actor: Actor,
    trace_id: str,
    answer: str,
    name: str = "trace-sft",
) -> DatasetVersion:
    """A person's corrected answer becomes the next version of a supervision set.

    The trace supplies the question, never the answer: training on the reply the
    model got wrong would teach the mistake. What the person writes is checked
    against the lines the question was answered from before it is stored, and
    the new version carries the previous one forward, so the set grows instead
    of being replaced. The result is unreleased; releasing it still needs the
    gate that every dataset needs.
    """
    require(platform.audit, actor, Action.BUILD_DATASET, trace_id)
    trace = get_trace(platform.store, actor.principal(), trace_id)
    corrected = _corrected_answer(trace, answer)
    existing = list(platform.store.list_datasets(name=name))
    previous = existing[-1] if existing else None
    carried = _supervision_rows(platform, previous)
    if any(row["source_id"] == trace_id for row in carried):
        raise ValueError(f"trace {trace_id} already has an example in {name}")
    rows = [
        *carried,
        {
            "instruction": trace.question,
            "response": corrected,
            "authored_by": actor.id,
            "source_id": trace_id,
        },
    ]
    dataset = ingest_supervision(
        platform,
        rows,
        name=name,
        version=(previous.version + 1) if previous else 1,
        lineage={"trace_ids": [trace_id], "previous": previous.id if previous else ""},
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="sft.example_from_trace",
        subject=dataset.id,
        details={
            "trace_id": trace_id,
            "name": name,
            "version": dataset.version,
            "carried": len(carried),
        },
    )
    return dataset
