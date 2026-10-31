"""One search over the artifacts the console holds.

The search reads the same rows the pages read, so it cannot show a person
something their own pages would not: no dataset record bodies, no evidence
bodies, no accounts. A hit is a name, an id, a state and a way to the page that
owns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from internal_ai.context import Platform
from internal_ai.people import PeopleRegistry
from internal_ai.services import build
from internal_ai.services.jobs import JobRunner

CAP = 5
JOB_SCAN = 200


@dataclass(slots=True)
class Hit:
    title: str
    meta: str
    status: str
    href: str
    created_at: str = ""


@dataclass(slots=True)
class Group:
    kind: str
    label: str
    hits: list[Hit] = field(default_factory=list)
    total: int = 0
    more: str = ""


def _more(path: str, term: str) -> str:
    return f"{path}?{urlencode({'q': term})}"


def _needle(term: str) -> str:
    return term.strip().casefold()


def _has(needle: str, *values: Any) -> bool:
    return any(needle in str(value).casefold() for value in values if value is not None)


def _group(kind: str, label: str, hits: list[Hit], path: str, term: str) -> Group | None:
    if not hits:
        return None
    return Group(kind=kind, label=label, hits=hits[:CAP], total=len(hits), more=_more(path, term))


def search(platform: Platform, jobs: JobRunner, term: str) -> list[Group]:
    needle = _needle(term)
    if not needle:
        return []
    groups: list[Group | None] = []

    datasets = [
        Hit(
            title=f"{item.name} v{item.version}",
            meta=(
                f"{item.record_count} records, "
                f"{'released' if item.released else 'draft'}, {item.id}"
            ),
            status="released" if item.released else "draft",
            href=f"/console/datasets/{item.id}",
            created_at=item.created_at,
        )
        for item in platform.store.list_datasets()
        if _has(needle, item.name, item.id)
    ]
    groups.append(_group("datasets", "Datasets", datasets, "/console/datasets", term))

    models = [
        Hit(
            title=f"{item.name} v{item.version}",
            meta=f"trained on {', '.join(item.dataset_ids) or 'no dataset'}, {item.id}",
            status=str(item.status),
            href=f"/console/models?{urlencode({'q': item.name})}",
            created_at=item.created_at,
        )
        for item in platform.store.list_models()
        if _has(needle, item.name, item.id, item.status)
    ]
    groups.append(_group("models", "Models", models, "/console/models", term))

    job_hits = [
        Hit(
            title=job["title"],
            meta=f"{job['requested_by']}, {job['id']}",
            status=str(job["state"]),
            href=f"/console/jobs/{job['id']}",
            created_at=job["created_at"],
        )
        for job in jobs.list(limit=JOB_SCAN)
        if _has(needle, job["title"], job["id"], job["requested_by"], job["state"])
    ]
    groups.append(_group("jobs", "Jobs", job_hits, "/console/jobs", term))

    evals = [
        Hit(
            title=f"{item.name} v{item.version}",
            meta=f"{len(item.items)} questions, {item.id}",
            status="frozen" if item.frozen else "open",
            href=f"/console/evals?{urlencode({'q': item.name})}",
            created_at=item.created_at,
        )
        for item in platform.store.list_eval_sets()
        if _has(needle, item.name, item.id)
    ]
    groups.append(_group("evals", "Eval sets", evals, "/console/evals", term))

    sources = [
        Hit(
            title=item["name"],
            meta=f"{item['path']}, {item['purpose']}, {item['id']}",
            status="withdrawn" if item["withdrawn"] else item["gate_status"],
            href=f"/console/sources?{urlencode({'q': item['name']})}",
            created_at=item["created_at"],
        )
        for item in build.list_sources(platform)
        if _has(needle, item["name"], item["id"], item["path"], item["purpose"], *item["acl"])
    ]
    groups.append(_group("sources", "Data sources", sources, "/console/sources", term))

    gates = [
        Hit(
            title=item.title,
            meta=f"{item.gate_type}, subject {item.subject}, {item.id}",
            status=str(item.status),
            href=f"/console/approvals?{urlencode({'q': item.id})}",
            created_at=item.created_at,
        )
        for item in sorted(
            platform.store.list_gates(status=None), key=lambda row: row.created_at, reverse=True
        )
        if _has(needle, item.title, item.id, item.subject, item.gate_type, item.requested_by)
    ]
    groups.append(_group("gates", "Gates", gates, "/console/approvals", term))

    people = [
        Hit(
            title=item["name"],
            meta=(
                f"{item['title'] or 'no title'}, "
                f"{', '.join(item['roles']) or 'no role'}, {item['id']}"
            ),
            status="placeholder" if item["placeholder"] else "person",
            href=f"/console/admin/people?{urlencode({'q': item['name']})}",
            created_at="",
        )
        for item in PeopleRegistry(platform.store).list_people()
        if _has(needle, item["name"], item["id"], item["title"], *item["roles"])
    ]
    groups.append(_group("people", "People", people, "/console/admin/people", term))

    return [group for group in groups if group is not None]
