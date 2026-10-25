"""Control plane API and CLI surface."""

from __future__ import annotations

from pathlib import Path

from conftest import enroll, token_for
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.cli import main
from internal_ai.context import Platform


def lead_token(root: Path) -> dict[str, str]:
    setup = Platform.create(root=root, llm_kind="scripted")
    try:
        lead = enroll(setup, "Lena Lead", ["program_lead"])
        return {"Authorization": f"Bearer {token_for(setup, lead)}"}
    finally:
        setup.close()


def test_api_exposes_the_control_plane(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/tasks").json() == []
        assert client.get("/gates").json() == []
        assert client.get("/audit/verify").json()["ok"] is True
        roles = client.get("/roles").json()
        assert {role["name"] for role in roles} >= {"orchestrator", "evaluation"}
        assert client.get("/policy").json()["undecided"]


def test_api_plan_and_run(tmp_path: Path) -> None:
    headers = lead_token(tmp_path)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        response = client.post(
            "/plan",
            json={"goal": "ship the embedding model", "planner": "template"},
            headers=headers,
        )
        assert response.status_code == 200
        tasks = response.json()["tasks"]
        assert tasks
        assert all(task["state"] == "proposed" for task in tasks)
        run = client.post("/run", json={"max_iterations": 2, "approve": True}, headers=headers)
        assert run.status_code == 200
        assert client.get("/tasks").json()


def test_api_rejects_bad_transitions(tmp_path: Path) -> None:
    headers = lead_token(tmp_path)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        client.post("/plan", json={"goal": "g", "planner": "template"}, headers=headers)
        task_id = client.get("/tasks").json()[0]["id"]
        response = client.post(
            f"/tasks/{task_id}/transition", params={"state": "accepted"}, headers=headers
        )
        assert response.status_code == 409


def test_api_requires_key_when_configured(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", api_key="secret")
    with TestClient(app) as client:
        assert client.get("/tasks").status_code == 401
        assert client.get("/tasks", headers={"X-Internal-Key": "secret"}).status_code == 200


def test_cli_demo_runs_end_to_end(tmp_path: Path, capsys) -> None:
    exit_code = main(["--root", str(tmp_path), "demo", "--budget", "50"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "audit chain verified: True" in output
    assert "accepted" in output
