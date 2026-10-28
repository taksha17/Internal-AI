"""The console's page furniture is a rule, so it is checked rather than trusted.

Every page finds its title in the same place, every column announces itself,
every time is readable and citable, and an empty list offers the next step.
A template that draws its own table or prints a raw timestamp fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "internal_ai" / "console" / "templates"
PAGES = [
    "/console/home",
    "/console/inbox",
    "/console/approvals",
    "/console/sources",
    "/console/datasets",
    "/console/evals",
    "/console/models",
    "/console/jobs",
    "/console/ask",
    "/console/plan",
    "/console/audit",
    "/console/admin/people",
    "/console/account",
    "/console/password",
]
LIST_PAGES = ["/console/jobs", "/console/datasets", "/console/audit"]


def test_only_the_macro_draws_a_table() -> None:
    offenders = [
        path.name
        for path in sorted(TEMPLATES.glob("*.html"))
        if path.name != "_macros.html"
        and ("<table" in path.read_text() or "<thead" in path.read_text())
    ]
    assert not offenders, f"tables belong in the data_table macro: {offenders}"


def test_timestamps_are_never_printed_raw() -> None:
    offenders = []
    for path in sorted(TEMPLATES.glob("*.html")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if "{{" in line and "created_at" in line and "| when" not in line:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, "a time is printed without the `when` filter: " + ", ".join(offenders)


@pytest.fixture
def signed_in(tmp_path: Path):
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        lead = enroll(app.state.platform, "Lena Lead", ["program_lead"])
        assert sign_in(client, lead.name).status_code == 303
        yield client


def test_every_page_uses_the_same_header_and_scoped_columns(signed_in: TestClient) -> None:
    for path in PAGES:
        page = signed_in.get(path)
        assert page.status_code == 200, path
        assert '<div class="page-head">' in page.text, path
        assert '<main id="main"' in page.text, path
        assert '<a class="skip" href="#main">' in page.text, path
        assert "<th>" not in page.text, path


def test_a_list_page_names_its_columns(signed_in: TestClient) -> None:
    for path in LIST_PAGES:
        page = signed_in.get(path)
        assert '<th scope="col"' in page.text, path


def test_times_are_readable_and_citable(signed_in: TestClient) -> None:
    page = signed_in.get("/console/audit")
    assert '<time datetime="' in page.text
    assert "just now" in page.text or "min ago" in page.text or "h ago" in page.text


def test_an_empty_list_offers_the_next_step(signed_in: TestClient) -> None:
    page = signed_in.get("/console/jobs")
    assert "No jobs yet" in page.text
    assert 'href="/console/sources"' in page.text


def test_a_wrong_id_gets_a_page_not_a_json_body(signed_in: TestClient) -> None:
    page = signed_in.get("/console/datasets/ds_missing")
    assert page.status_code == 404
    assert "That page does not exist" in page.text
    assert 'action="/console/kill-switch"' in page.text
    assert 'href="/console/home"' in page.text
    assert "ds_missing" in page.text


def test_an_anonymous_wrong_id_still_goes_to_sign_in(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        page = client.get("/console/datasets/ds_missing", follow_redirects=False)
        assert page.status_code == 303
        assert page.headers["location"].endswith("/console/signin")
