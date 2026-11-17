"""The privacy re-run and the conversation exports, reached from the console.

Both call the functions the CLI calls, so these tests are about reachability and the
refusal text a person sees, not about audit or export logic.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.data.records import Record
from internal_ai.domain import DatasetVersion


@contextmanager
def console(tmp_path: Path) -> Iterator[tuple[TestClient, Any]]:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        yield client, app.state.platform


def dataset(platform: Any, *, text: str = "the pump primes in four strokes") -> str:
    dataset_id = "ds_audit"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="corpus",
            version=1,
            content_hash="h",
            record_count=1,
            token_count=12,
            created_at="2026-01-01T00:00:00+00:00",
            released=True,
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            Record(
                id="rec_1",
                record_key="doc-1",
                text=text,
                source_id="manuals",
            ).to_row()
        ],
    )
    platform.store.conn.execute(
        "UPDATE dataset_records SET split = 'train' WHERE dataset_id = ?", (dataset_id,)
    )
    return dataset_id


def test_the_sample_runs_and_reports_what_it_read(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        dataset_id = dataset(platform)
        owner = enroll(platform, "Ola Owner", ["data_owner"])
        sign_in(client, owner.name)

        response = client.post(
            f"/console/datasets/{dataset_id}/audit",
            data={"csrf_token": session_csrf(client), "sample_size": "1"},
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "1 records sampled and no critical finding" in response.text


def test_a_sample_that_finds_pii_says_not_to_release(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        dataset_id = dataset(platform, text="write to someone@example.com for the torque spec")
        owner = enroll(platform, "Ola Owner", ["data_owner"])
        sign_in(client, owner.name)

        response = client.post(
            f"/console/datasets/{dataset_id}/audit",
            data={"csrf_token": session_csrf(client), "sample_size": "1"},
            follow_redirects=True,
        )

        assert "do not release this dataset" in response.text
        assert "email" in response.text


def test_a_role_that_does_not_own_the_data_cannot_audit(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        dataset_id = dataset(platform)
        expert = enroll(platform, "Dana Expert", ["domain_expert"])
        sign_in(client, expert.name)

        response = client.post(
            f"/console/datasets/{dataset_id}/audit",
            data={"csrf_token": session_csrf(client), "sample_size": "1"},
            follow_redirects=True,
        )

        assert "audit dataset needs compliance or data_owner or ml_lead" in response.text


def test_exporting_failed_answers_makes_an_unreleased_dataset(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        owner = enroll(platform, "Ola Owner", ["data_owner"])
        sign_in(client, owner.name)

        response = client.post(
            "/console/exports/feedback",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "ask-feedback v1" in response.text
        assert "unreleased" in response.text


def test_exporting_preferences_with_none_filed_is_a_refusal(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        owner = enroll(platform, "Ola Owner", ["data_owner", "domain_expert"])
        sign_in(client, owner.name)

        response = client.post(
            "/console/exports/preferences",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=True,
        )

        assert "no preferences have been filed" in response.text


def test_an_unknown_export_names_the_ones_that_exist(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        owner = enroll(platform, "Ola Owner", ["data_owner"])
        sign_in(client, owner.name)

        response = client.post(
            "/console/exports/traces",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=True,
        )

        assert "feedback, preferences" in response.text
