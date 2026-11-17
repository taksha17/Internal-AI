"""Compute advice from the console.

The page must show what the command shows and refuse what the command refuses, so
the last test compares the two texts instead of trusting that they match.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.compute.advise import advise_dataset, require_runnable_size
from internal_ai.domain import DatasetVersion, TaskKind  # noqa: F401


def released_dataset(platform_fixture: Any, *, tokens: int, kind: str, name: str = "corpus") -> str:
    dataset = DatasetVersion(
        id=f"ds_{name}",
        name=name,
        version=1,
        content_hash="h",
        record_count=10,
        token_count=tokens,
        created_at="2026-01-01T00:00:00+00:00",
        stats={"token_count_kind": kind},
        released=True,
    )
    platform_fixture.store.add_dataset(dataset)
    return dataset.id


@contextmanager
def console(tmp_path: Path) -> Iterator[tuple[TestClient, Any]]:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        yield client, app.state.platform


def test_the_page_lists_the_sizes_with_hours_and_machine(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        dataset_id = released_dataset(platform, tokens=200_000_000, kind="measured")
        cora = enroll(platform, "Cora Compliance", ["compliance"])
        assert sign_in(client, cora.name).status_code == 303
        platform.store.conn.execute("SELECT 1")

        before = client.get(f"/console/compute?dataset={dataset_id}")
        assert before.status_code == 200
        assert "No advice stored for this corpus yet" in before.text

        assert (
            client.post(
                "/console/compute/advise",
                data={"csrf_token": session_csrf(client), "dataset_id": dataset_id},
                follow_redirects=False,
            ).status_code
            == 303
        )

        page = client.get(f"/console/compute?dataset={dataset_id}")
        assert page.status_code == 200
        for size in ("small", "1b", "7b"):
            assert size in page.text
        assert "200,000,000 tokens" in page.text
        assert "Attached backend" in page.text
        assert "The floor is 20 tokens per parameter" in page.text


def test_the_page_shows_why_a_size_is_waiting_on_a_small_corpus(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        dataset_id = released_dataset(platform, tokens=865_165, kind="measured")
        advise_dataset(platform, dataset_id)
        cora = enroll(platform, "Cora Compliance", ["compliance"])
        assert sign_in(client, cora.name).status_code == 303

        page = client.get(f"/console/compute?dataset={dataset_id}")

        assert "below the tokens-per-parameter floor" in page.text
        assert "waiting" in page.text


def test_the_page_says_an_estimate_makes_every_size_provisional(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        dataset_id = released_dataset(platform, tokens=200_000_000, kind="estimate")
        advise_dataset(platform, dataset_id)
        cora = enroll(platform, "Cora Compliance", ["compliance"])
        assert sign_in(client, cora.name).status_code == 303

        page = client.get(f"/console/compute?dataset={dataset_id}")

        assert "The token count is an estimate" in page.text
        assert "the token count is still an estimate" in page.text


def test_an_unsigned_in_visitor_is_sent_to_sign_in(tmp_path: Path) -> None:
    with console(tmp_path) as (client, _platform):
        response = client.get("/console/compute", follow_redirects=False)
        assert response.status_code == 303


def test_only_an_infra_engineer_attaches_a_backend(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        analyst = enroll(platform, "Anna Analyst", ["analyst"])
        engineer = enroll(platform, "Eve Engineer", ["infra_engineer"])

        assert sign_in(client, analyst.name).status_code == 303
        page = client.get("/console/compute")
        assert "Attaching a backend needs infra_engineer" in page.text
        client.post(
            "/console/compute/attach",
            data={"csrf_token": session_csrf(client), "backend": "slurm"},
            follow_redirects=False,
        )
        denials = platform.store.list_audit(action="console.denied")
        assert denials and str(denials[0].outcome) == "denied"
        assert (platform.store.kv_get("compute_backend") or "local") == "local"

        client.post("/console/signout", data={"csrf_token": session_csrf(client)})
        assert sign_in(client, engineer.name).status_code == 303
        assert (
            client.post(
                "/console/compute/attach",
                data={"csrf_token": session_csrf(client), "backend": "slurm"},
                follow_redirects=False,
            ).status_code
            == 303
        )
        assert platform.store.kv_get("compute_backend") == "slurm"
        attached = platform.store.list_audit(action="compute.attached")
        assert attached and attached[0].subject == "slurm"


def test_the_page_and_the_refusal_use_the_same_words(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        dataset_id = released_dataset(platform, tokens=865_165, kind="estimate")
        advise_dataset(platform, dataset_id)
        cora = enroll(platform, "Cora Compliance", ["compliance"])
        assert sign_in(client, cora.name).status_code == 303

        page = client.get(f"/console/compute?dataset={dataset_id}")

        with pytest.raises(ValueError) as refused:
            require_runnable_size(platform, dataset_id, "7b")

        reasons = refused.value.args[0].split("(", 1)[1].rstrip(")")
        for reason in reasons.split("; "):
            assert reason in page.text, f"the page does not explain: {reason}"
