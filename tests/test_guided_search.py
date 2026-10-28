"""The novice path: paste a folder, pick what to get out of it, no settings."""

from __future__ import annotations

import re
import time
from pathlib import Path

from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient
from programme import actor_named, record_ready_programme

from internal_ai.api import create_app
from internal_ai.context import Platform
from internal_ai.domain import GateType
from internal_ai.services import build

INSPECT_TOKEN = re.compile(r"/console/sources/inspect/([A-Za-z0-9_-]+)")


def labelled_table(platform: Platform) -> Path:
    """One file with an answer column, the one shape `decide` is ready for."""
    data_dir = platform.settings.root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "tickets.csv"
    rows = ["text,label"] + [
        f"ticket {index} about resetting a vpn,{'network' if index % 2 else 'billing'}"
        for index in range(24)
    ]
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target


def scanned(client: TestClient, page: str, timeout: float = 5.0) -> str:
    """Follow a scan the page started, as the poll would, until it has a result."""
    match = INSPECT_TOKEN.search(page)
    deadline = time.monotonic() + timeout
    body = page
    while match is not None and time.monotonic() < deadline:
        body = client.get(f"/console/sources/inspect/{match.group(1)}").text
        if "What the platform sees" in body:
            break
        time.sleep(0.02)
    return body


def test_guided_register_records_what_it_is_aiming_at(platform: Platform, enroll_person) -> None:
    dana = enroll_person(platform, "Dana Owner", ["data_owner"])
    table = labelled_table(platform)

    source = build.guided_register(platform, dana, str(table), "decide")
    assert source["purpose"] == build.GUIDED_PURPOSES["decide"]
    assert "classifier" in source["purpose"]

    fallback = build.guided_register(platform, dana, str(table), "whatever")
    assert fallback["purpose"] == build.GUIDED_PURPOSES["search"]


def test_guided_tracker_follows_the_intent_and_names_the_role(tmp_path: Path) -> None:
    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    enroll(setup, "Dana Owner", ["data_owner"], ["engineering"])
    enroll(setup, "Mo Builder", ["ml_lead"], ["engineering"])
    table = labelled_table(setup)
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        dana = TestClient(app)
        assert sign_in(dana, "Dana Owner").status_code == 303
        guided = dana.post(
            "/console/guided",
            data={"csrf_token": session_csrf(dana), "path": str(table), "intent": "decide"},
            follow_redirects=False,
        )
        assert guided.status_code == 303
        assert "intent=decide" in guided.headers["location"]
        source = build.list_sources(platform)[0]
        assert "classifier" in source["purpose"]

        page = dana.get(guided.headers["location"])
        assert "Sort it into labels" in page.text
        assert "Encoder trained on this data" in page.text
        assert "Classifier trained" in page.text
        assert "needs ml_lead" in page.text

        search = dana.get(f"/console/sources?guided={source['id']}&intent=search")
        assert "Askable" in search.text
        assert "Classifier trained" not in search.text
        assert "needs ml_lead" not in search.text

        recommended = dana.get(f"/console/sources?guided={source['id']}")
        assert "Classifier trained" in recommended.text
        assert "best fit" in recommended.text

        mo = TestClient(app)
        assert sign_in(mo, "Mo Builder").status_code == 303
        builder_view = mo.get(f"/console/sources?guided={source['id']}")
        assert "Encoder trained on this data" in builder_view.text
        assert "needs ml_lead" not in builder_view.text


def test_suggestions_start_the_path_they_recommend(tmp_path: Path) -> None:
    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    enroll(setup, "Dana Owner", ["data_owner"], ["engineering"])
    table = labelled_table(setup)
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        dana = TestClient(app)
        assert sign_in(dana, "Dana Owner").status_code == 303
        csrf = session_csrf(dana)
        body = scanned(dana, dana.get(f"/console/sources?inspect={table}").text)
        assert "What to build from this" in body
        assert re.search(r'name="path" value="[^"]*tickets\.csv"', body)
        assert 'name="intent" value="decide"' in body
        assert "Start this path" in body

        started = dana.post(
            "/console/guided",
            data={"csrf_token": csrf, "path": str(table), "intent": "decide"},
            follow_redirects=False,
        )
        assert started.status_code == 303
        assert "classifier" in build.list_sources(app.state.platform)[0]["purpose"]


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
        record_ready_programme(platform, actor_named(platform, "Mo Builder"))
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
