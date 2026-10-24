"""A plan cannot be submitted twice into the same control plane.

Task ids are derived from the plan keys so runs are reproducible, which means
a second submission of the same plan collides with the tasks already stored.
That collision has to surface as a clear refusal, not a database traceback.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.cli import main
from internal_ai.context import Platform
from internal_ai.engine import PlanError
from internal_ai.planner import TemplatePlanner


def test_resubmitting_the_same_plan_is_refused(tmp_path: Path) -> None:
    with Platform.create(root=tmp_path, llm_kind="scripted") as platform:
        drafts = TemplatePlanner().plan("ship the embedding model", {})
        platform.orchestrator.submit_plan(drafts)
        try:
            platform.orchestrator.submit_plan(drafts)
        except PlanError as exc:
            assert "plan already submitted" in str(exc)
        else:
            raise AssertionError("the second submission should have been refused")


def test_cli_demo_refuses_to_run_twice_in_one_root(tmp_path: Path, capsys) -> None:
    assert main(["--root", str(tmp_path), "demo", "--budget", "50"]) == 0
    capsys.readouterr()
    assert main(["--root", str(tmp_path), "demo", "--budget", "50"]) == 1
    captured = capsys.readouterr()
    assert "plan already submitted" in captured.err
    assert "Traceback" not in captured.err


def test_api_refuses_a_duplicate_plan_with_conflict(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        first = client.post(
            "/plan", json={"goal": "ship the embedding model", "planner": "template"}
        )
        assert first.status_code == 200
        second = client.post(
            "/plan", json={"goal": "ship the embedding model", "planner": "template"}
        )
        assert second.status_code == 409
        assert "plan already submitted" in second.json()["detail"]


def test_budget_status_reports_the_kill_switch(tmp_path: Path, capsys) -> None:
    assert main(["--root", str(tmp_path), "budget", "status"]) == 0
    assert "kill switch: released" in capsys.readouterr().out
    assert main(["--root", str(tmp_path), "budget", "kill"]) == 0
    capsys.readouterr()
    assert main(["--root", str(tmp_path), "budget", "status"]) == 0
    assert "kill switch: engaged" in capsys.readouterr().out
