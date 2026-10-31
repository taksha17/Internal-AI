"""Finding things in the console: search, filters, order and pages.

Search reads the rows the pages read, so a hit can never be data the person
could not open. Filters, sorting and paging live in the URL, so a view can be
reloaded or sent to a colleague.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion, GateType, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.services import governance
from internal_ai.services.actor import Actor


def dataset(platform: Platform, name: str, *, released: bool = True, records: int = 10) -> str:
    item = DatasetVersion(
        id=f"ds_{name.replace('-', '_')}",
        name=name,
        version=1,
        content_hash="hash",
        record_count=records,
        token_count=records * 10,
        created_at=now_iso(),
        released=released,
    )
    platform.store.add_dataset(item)
    return item.id


def model(platform: Platform, name: str) -> str:
    item = ModelVersion(
        id=f"mdl_{name.replace('-', '_')}",
        name=name,
        version=1,
        content_hash="hash",
        created_at=now_iso(),
    )
    platform.store.add_model(item)
    return item.id


@contextmanager
def console(tmp_path: Path) -> Iterator[tuple[TestClient, Any, Actor]]:
    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    actor = enroll(setup, "Lena Lead", ["program_lead"])
    setup.close()
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        assert sign_in(client, actor.name).status_code == 303
        yield client, app, actor


def test_search_finds_a_dataset_a_model_and_a_person(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, _actor):
        dataset(app.state.platform, "support-tickets")
        model(app.state.platform, "tickets-embed")

        page = client.get("/console/search", params={"q": "tickets"})
        assert page.status_code == 200
        assert "support-tickets v1" in page.text
        assert 'href="/console/datasets/ds_support_tickets"' in page.text
        assert "tickets-embed v1" in page.text
        assert "Datasets" in page.text and "Models" in page.text
        assert 'action="/console/search"' in page.text

        person_page = client.get("/console/search", params={"q": "Lena"})
        assert "Lena Lead" in person_page.text
        assert "People" in person_page.text


def test_search_finds_a_job_by_its_id(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, actor):
        job = app.state.jobs.submit(
            actor=actor,
            kind="train_model",
            title="train tickets-embed",
            params={},
            work=lambda: {"status": "completed"},
            agent="ml_lead",
        )
        page = client.get("/console/search", params={"q": job["id"]})
        assert f"/console/jobs/{job['id']}" in page.text
        assert "Jobs" in page.text


def test_search_never_returns_a_record_body(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, _actor):
        dataset_id = dataset(app.state.platform, "payroll-notes")
        app.state.platform.store.add_dataset_records(
            dataset_id,
            [
                {
                    "record_key": "payroll",
                    "payload": "SECRET-PAYROLL-FIGURE",
                    "acl": ["payroll"],
                }
            ],
        )
        page = client.get("/console/search", params={"q": "SECRET-PAYROLL"})
        assert "Nothing matches" in page.text
        assert "SECRET-PAYROLL-FIGURE" not in page.text


def test_search_finds_a_gate_and_links_to_its_tab(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, actor):
        governance.request_gate(
            app.state.platform,
            actor,
            GateType.BUDGET_INCREASE,
            title="Raise the weekly cap",
            subject="weekly:global",
            evidence={"spend_report": "at the cap", "justification": "the next run"},
        )
        page = client.get("/console/search", params={"q": "weekly:global"})
        assert "Raise the weekly cap" in page.text
        assert "Gates" in page.text
        assert "/console/approvals?q=" in page.text


def test_an_empty_search_explains_itself(tmp_path: Path) -> None:
    with console(tmp_path) as (client, _app, _actor):
        page = client.get("/console/search")
        assert "Type a name or an id" in page.text
        missing = client.get("/console/search", params={"q": "nothing-here-at-all"})
        assert "Nothing matches" in missing.text


def test_a_filter_narrows_and_a_sort_reorders(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, _actor):
        dataset(app.state.platform, "alpha-draft", released=False, records=3)
        dataset(app.state.platform, "beta-live", released=True, records=90)

        draft = client.get("/console/datasets", params={"state": "draft"})
        assert "alpha-draft" in draft.text
        assert "beta-live" not in draft.text

        big = client.get("/console/datasets", params={"sort": "records", "dir": "desc"})
        assert 'aria-sort="descending"' in big.text
        assert big.text.index("beta-live") < big.text.index("alpha-draft")

        small = client.get("/console/datasets", params={"sort": "records", "dir": "asc"})
        assert 'aria-sort="ascending"' in small.text
        assert small.text.index("alpha-draft") < small.text.index("beta-live")


def test_paging_splits_a_list_and_clamps_the_last_page(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, _actor):
        for index in range(30):
            dataset(app.state.platform, f"batch-{index:02d}")

        first = client.get("/console/datasets")
        assert "1 to 25 of 30 datasets" in first.text
        assert "Page 1 of 2" in first.text

        second = client.get("/console/datasets", params={"page": 2})
        assert "26 to 30 of 30 datasets" in second.text
        assert "batch-26" in second.text
        assert 'rel="next"' not in second.text

        beyond = client.get("/console/datasets", params={"page": 9})
        assert "26 to 30 of 30 datasets" in beyond.text


def test_jobs_filter_by_state_and_title(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, actor):
        for title in ("build alpha dataset", "build beta dataset"):
            app.state.jobs.submit(
                actor=actor,
                kind="build_dataset",
                title=title,
                params={},
                work=lambda: {"status": "completed"},
                agent="data_owner",
            )
        page = client.get("/console/jobs", params={"q": "alpha"})
        assert "build alpha dataset" in page.text
        assert "build beta dataset" not in page.text
        assert "1 to 1 of 1 jobs" in page.text

        failed = client.get("/console/jobs", params={"state": "failed"})
        assert "No jobs match" in failed.text


def test_audit_pages_through_the_whole_chain(tmp_path: Path) -> None:
    with console(tmp_path) as (client, app, actor):
        for index in range(55):
            app.state.platform.audit.record(
                actor=actor.id, action="test.event", subject=f"subject-{index}"
            )
        first = client.get("/console/audit")
        assert "1 to 50 of" in first.text
        assert "Page 1 of 2" in first.text
        second = client.get("/console/audit", params={"page": 2})
        assert "Page 2 of 2" in second.text
        assert "51 to " in second.text

        oldest = client.get("/console/audit", params={"dir": "asc"})
        assert 'aria-sort="ascending"' in oldest.text
