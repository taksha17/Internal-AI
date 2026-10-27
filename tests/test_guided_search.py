"""The novice path: paste a folder, get guided search, no settings."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.domain import GateType
from internal_ai.services import build


def test_guided_register_builds_release_and_asks(tmp_path: Path) -> None:
    from internal_ai.context import Platform

    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    enroll(setup, "Dana Owner", ["data_owner"], ["engineering"])
    enroll(setup, "Owen Owner", ["data_owner"], ["engineering"])
    enroll(setup, "Mo Builder", ["ml_lead"], ["engineering"])
    enroll(setup, "Cora Compliance", ["compliance"], ["engineering"])
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        dana = TestClient(app)
        assert sign_in(dana, "Dana Owner").status_code == 303
        csrf = session_csrf(dana)
        page = dana.post("/console/sources/sample", data={"csrf_token": csrf, "records": "80"})
        assert page.status_code == 200
        sample = next(build.samples_dir(platform).glob("*.jsonl"))

        guided = dana.post(
            "/console/guided",
            data={"csrf_token": csrf, "path": str(sample)},
            follow_redirects=False,
        )
        assert guided.status_code == 303
        assert "guided=" in guided.headers["location"]

        tracker = dana.get(guided.headers["location"])
        assert tracker.status_code == 200
        assert "Guided progress" in tracker.text
        assert "1 of 6" in tracker.text

        source = build.list_sources(platform)[0]
        assert source["gate_status"] == "pending"

        owen = TestClient(app)
        assert sign_in(owen, "Owen Owner").status_code == 303
        owen_csrf = session_csrf(owen)
        mo = TestClient(app)
        assert sign_in(mo, "Mo Builder").status_code == 303
        mo_csrf = session_csrf(mo)
        cora = TestClient(app)
        assert sign_in(cora, "Cora Compliance").status_code == 303
        cora_csrf = session_csrf(cora)
        for client, csrf_token in ((owen, owen_csrf), (cora, cora_csrf)):
            gate = build.gate_for(platform, GateType.NEW_DATA_SOURCE, source["id"])
            response = client.post(
                f"/console/approvals/{gate.id}",
                data={
                    "csrf_token": csrf_token,
                    "decision": "approve",
                    "note": "evidence checked",
                },
            )
            assert response.status_code == 200

        build_page = dana.post(f"/console/sources/{source['id']}/build", data={"csrf_token": csrf})
        assert build_page.status_code == 200
        dataset_id = build.get_source(platform, source["id"])["dataset_id"]
        assert dataset_id

        split = dana.post(f"/console/datasets/{dataset_id}/split", data={"csrf_token": csrf})
        assert split.status_code == 200
        dana.post(
            f"/console/datasets/{dataset_id}/request-release",
            data={"csrf_token": csrf},
        )
        for client, csrf_token in ((mo, mo_csrf), (cora, cora_csrf)):
            gate = build.gate_for(platform, GateType.DATASET_RELEASE, dataset_id)
            response = client.post(
                f"/console/approvals/{gate.id}",
                data={
                    "csrf_token": csrf_token,
                    "decision": "approve",
                    "note": "evidence checked",
                },
            )
            assert response.status_code == 200
        dana.post(f"/console/datasets/{dataset_id}/release", data={"csrf_token": csrf})
        assert platform.store.get_dataset(dataset_id).released

        answer = dana.post(
            "/console/ask",
            data={
                "csrf_token": csrf,
                "dataset_id": dataset_id,
                "question": "how do I reset my VPN",
            },
        )
        assert answer.status_code == 200
        assert "Sources" in answer.text
        assert re.search(r"keyword search", answer.text, re.IGNORECASE)


def test_guided_register_refuses_outside_roots(tmp_path: Path) -> None:
    from internal_ai.context import Platform

    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    enroll(setup, "Dana Owner", ["data_owner"], ["engineering"])
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        dana = TestClient(app)
        assert sign_in(dana, "Dana Owner").status_code == 303
        csrf = session_csrf(dana)
        response = dana.post("/console/guided", data={"csrf_token": csrf, "path": "/etc/hostname"})
        assert response.status_code == 200
        assert "flash-error" in response.text or "flash flash-error" in response.text
