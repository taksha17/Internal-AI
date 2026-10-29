"""A run waits until the brief, its owners, and the data-class policy are real."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import enroll, sign_in
from fastapi.testclient import TestClient
from programme import record_ready_programme

from internal_ai.api import create_app
from internal_ai.brief import require_programme
from internal_ai.people import PeopleRegistry
from internal_ai.services import build
from internal_ai.workflows import train_classifier_model, train_embedding_model


def test_empty_brief_denies_release_and_both_trains(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Dana Owner", ["data_owner", "ml_lead"])
    with pytest.raises(PermissionError, match="Still to record"):
        build.release_dataset(platform, owner, "ds_missing")
    with pytest.raises(PermissionError, match="Still to record"):
        train_embedding_model(platform, "ds_missing", None)  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="Still to record"):
        train_classifier_model(platform, "ds_missing", Path("missing.pt"), None)
    rows = platform.store.conn.execute(
        "SELECT outcome FROM audit_events WHERE action = 'programme.refused'"
    ).fetchall()
    assert len(rows) == 3
    assert {row["outcome"] for row in rows} == {"denied"}


def test_complete_brief_reaches_the_checks_that_already_exist(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Dana Owner", ["data_owner", "ml_lead"])
    record_ready_programme(platform, owner)
    with pytest.raises(PermissionError, match="needs an approved"):
        build.release_dataset(platform, owner, "ds_missing")
    with pytest.raises(KeyError, match="unknown dataset"):
        train_embedding_model(platform, "ds_missing", None)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="unknown dataset"):
        train_classifier_model(platform, "ds_missing", Path("missing.pt"), None)


def test_a_placeholder_cannot_own_infra_or_evaluation(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Dana Owner", ["data_owner", "ml_lead"])
    owners = record_ready_programme(platform, owner)
    placeholder = next(
        person for person in PeopleRegistry(platform.store).list_people() if person["placeholder"]
    )
    owners["infra"] = placeholder["id"]
    owners["evaluation"] = placeholder["id"]
    record_ready_programme(platform, owner, owners=owners)
    with pytest.raises(PermissionError, match="placeholder") as caught:
        require_programme(platform, owner.id, release=False, actor_kind=owner.kind)
    assert "infra is still a placeholder" in str(caught.value)
    assert "evaluation is still a placeholder" in str(caught.value)


def test_jobs_page_shows_the_same_gap(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app):
        enroll(app.state.platform, "Mo Builder", ["ml_lead"])
        client = TestClient(app)
        assert sign_in(client, "Mo Builder").status_code == 303
        page = client.get("/console/jobs")
        assert page.status_code == 200
        assert "Still to record:" in page.text
        assert "the target, languages, context length and quality bar" in page.text
