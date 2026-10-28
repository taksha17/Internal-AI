"""The project brief is the record the acceptance checklist asks a company to keep.

It is checked here because it is the one page that decides whether a run has a
target to miss, a metric to judge it by, and a person to ask when it fails.
"""

from __future__ import annotations

import re
from pathlib import Path

from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai import brief
from internal_ai.api import create_app
from internal_ai.domain import AuditOutcome

FLASH = re.compile(r'class="flash flash-(?P<kind>ok|error)"[^>]*>(?P<body>[^<]*)')


def full_brief(**overrides: str) -> dict[str, str]:
    data = {
        "target": "Answer support questions from resolved tickets, with a citation.",
        "quality_bar": "Better recall than the keyword search agents use today.",
        "languages": "English",
        "context_length": "512",
        "metric_name_1": "recall@10",
        "metric_direction_1": "at_least",
        "metric_threshold_1": "0.70",
        "budget_usd": "5000",
        "timeline": "Pilot by the end of Q1",
        "deployment": "On-prem, support VLAN",
        "latency_ms": "800",
        "data_restrictions": "No customer records outside the EU region.",
        "non_goals": "Not a general assistant; no answer without a cited source.",
    }
    for role in brief.OWNER_ROLES:
        data[f"owner_{role}"] = "Lena Lead"
    data.update(overrides)
    return data


def flashes(page: str) -> list[tuple[str, str]]:
    return [(match.group("kind"), match.group("body")) for match in FLASH.finditer(page)]


def signed_in_as(app, name: str) -> tuple[TestClient, str]:
    client = TestClient(app)
    assert sign_in(client, name).status_code == 303
    return client, session_csrf(client)


def test_a_complete_brief_reaches_the_overview_and_the_plan(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        enroll(platform, "Lena Lead", ["program_lead"])

        client, csrf = signed_in_as(app, "Lena Lead")
        page = client.post("/console/brief", data={**full_brief(), "csrf_token": csrf})
        assert page.status_code == 200, page.text
        assert ("ok", "brief saved") in flashes(page.text)

        saved = brief.load(platform.store)
        assert saved.complete, saved.missing
        assert saved.metrics[0].threshold == "0.70"

        overview = client.get("/console/home").text
        assert "Project brief" in overview
        assert "Answer support questions from resolved tickets" in overview
        assert "recall@10" in overview and "≥ 0.70" in overview
        assert "Still to record" not in overview

        plan = client.get("/console/plan").text
        assert "has no complete project brief" not in plan

        event = next(
            item for item in platform.audit.recent(limit=50) if item.action == "brief.updated"
        )
        assert event.details["complete"] is True
        assert event.details["missing"] == []


def test_a_brief_without_owners_names_them_where_the_work_starts(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        enroll(platform, "Lena Lead", ["program_lead"])

        client, csrf = signed_in_as(app, "Lena Lead")
        data = full_brief(**{f"owner_{role}": "" for role in brief.OWNER_ROLES})
        page = client.post("/console/brief", data={**data, "csrf_token": csrf})
        unowned = "owners for " + ", ".join(brief.OWNER_ROLES)
        assert ("ok", f"brief saved; still missing {unowned}") in flashes(page.text)

        saved = brief.load(platform.store)
        assert saved.missing == [unowned]

        overview = client.get("/console/home").text
        assert "Still to record" in overview
        assert "owners for ml_lead" in overview

        plan = client.get("/console/plan").text
        assert "has no complete project brief" in plan
        assert "owners for ml_lead" in plan


def test_writing_the_brief_is_refused_to_a_role_that_does_not_hold_it(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        enroll(platform, "Cora Compliance", ["compliance"])

        client, csrf = signed_in_as(app, "Cora Compliance")
        page = client.post("/console/brief", data={**full_brief(), "csrf_token": csrf})
        assert page.status_code == 200
        assert any(kind == "error" for kind, _body in flashes(page.text))
        assert not brief.load(platform.store).target

        refusal = next(
            item for item in platform.audit.recent(limit=50) if item.action == "console.denied"
        )
        assert refusal.outcome == AuditOutcome.DENIED
        assert "write brief" in refusal.details["reason"]

        read_only = client.get("/console/brief").text
        assert "needs program_lead or ml_lead" in read_only
