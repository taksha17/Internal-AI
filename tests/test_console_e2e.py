"""A novice's whole build, clicked through the console by separate people."""

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

FLASH_ERROR = re.compile(r'class="flash flash-error"[^>]*>([^<]*)')
INSPECT_TOKEN = re.compile(r"/console/sources/inspect/([A-Za-z0-9_-]+)")


class Person:
    def __init__(self, app, name: str) -> None:
        self.client = TestClient(app)
        self.name = name
        response = sign_in(self.client, name)
        assert response.status_code == 303, response.text
        self.csrf = session_csrf(self.client)

    def post(self, url: str, ok: bool = True, **data: str) -> str:
        response = self.client.post(url, data={**data, "csrf_token": self.csrf})
        assert response.status_code == 200, response.text
        errors = FLASH_ERROR.findall(response.text)
        if ok:
            assert not errors, errors
        else:
            assert errors, f"expected a refusal from {url}"
        return response.text

    def get(self, url: str) -> str:
        response = self.client.get(url)
        assert response.status_code == 200, response.text
        return response.text

    def inspected(self, page: str, timeout: float = 5.0) -> str:
        """Follow the live status of a scan the page started, as the poll would."""
        match = INSPECT_TOKEN.search(page)
        if match is None:
            return page
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = self.get(f"/console/sources/inspect/{match.group(1)}")
            if "What the platform sees" in body:
                return body
            time.sleep(0.02)
        raise AssertionError("the inspection never reported a result")


def approve(platform: Platform, person: Person, gate_type: GateType, subject: str) -> None:
    gate = build.gate_for(platform, gate_type, subject)
    assert gate is not None
    person.post(f"/console/approvals/{gate.id}", decision="approve", note="evidence checked")


def test_console_click_through_builds_and_promotes_a_model(tmp_path: Path) -> None:
    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    enroll(setup, "Lena Lead", ["program_lead"], ["engineering", "platform-admins"])
    enroll(setup, "Mo Builder", ["ml_lead"])
    enroll(setup, "Mia Reviewer", ["ml_lead"])
    enroll(setup, "Dana Owner", ["data_owner"])
    enroll(setup, "Cora Compliance", ["compliance"])
    enroll(setup, "Dex Expert", ["domain_expert"])
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        mo, mia = Person(app, "Mo Builder"), Person(app, "Mia Reviewer")
        dana, cora = Person(app, "Dana Owner"), Person(app, "Cora Compliance")
        dex = Person(app, "Dex Expert")

        assert "Build a model, step by step" in mo.get("/console/home")

        page = mo.inspected(mo.post("/console/sources/sample", records="80"))
        assert "What the platform sees" in page
        sample = next(build.samples_dir(platform).glob("*.jsonl"))
        mo.post(
            "/console/sources",
            path=str(sample),
            name="support-tickets",
            purpose="search over resolved tickets",
            acl="engineering",
            legal_status="internal",
        )
        source = build.list_sources(platform)[0]
        assert source["gate_status"] == "pending"

        mo.post(f"/console/sources/{source['id']}/build", ok=False)
        mo.post(
            f"/console/approvals/{source['gate_id']}", ok=False, decision="approve", note="mine"
        )
        approve(platform, dana, GateType.NEW_DATA_SOURCE, source["id"])
        approve(platform, cora, GateType.NEW_DATA_SOURCE, source["id"])

        page = mo.post(f"/console/sources/{source['id']}/build")
        assert "succeeded" in page
        dataset_id = build.get_source(platform, source["id"])["dataset_id"]
        assert dataset_id
        overview = mo.get("/console/home")
        assert "Active jobs" in overview and "queue is clear" in overview

        record_ready_programme(platform, actor_named(platform, "Mo Builder"))
        mo.post(f"/console/datasets/{dataset_id}/split")
        dana.post(f"/console/datasets/{dataset_id}/request-release")
        approve(platform, mo, GateType.DATASET_RELEASE, dataset_id)
        approve(platform, cora, GateType.DATASET_RELEASE, dataset_id)
        dana.post(f"/console/datasets/{dataset_id}/release")
        assert platform.store.get_dataset(dataset_id).released

        page = mo.post(f"/console/datasets/{dataset_id}/tokenizer", vocab_size="600")
        assert "succeeded" in page
        assert build.tokenizer_path(platform, dataset_id) is not None

        dex.post(f"/console/datasets/{dataset_id}/eval-set", name="tickets-eval", count="16")
        eval_set = platform.store.list_eval_sets()[0]
        dex.post(f"/console/evals/{eval_set.id}/request-freeze")
        approve(platform, mo, GateType.EVAL_SET_FREEZE, eval_set.id)
        mo.post(f"/console/evals/{eval_set.id}/freeze")
        assert platform.store.get_eval_set(eval_set.id).frozen
        mo.post(f"/console/evals/{eval_set.id}/edit", ok=False)

        page = mo.post(
            "/console/models/train",
            dataset_id=dataset_id,
            name="tickets-embed",
            steps="80",
            cost_cap_usd="1",
        )
        assert "succeeded" in page
        run = app.state.jobs.list(kind="train_model")[0]
        mo.post(f"/console/runs/{run['id']}/register")
        model = platform.store.list_models()[0]

        page = mo.post(f"/console/models/{model.id}/evaluate", eval_set_id=eval_set.id)
        assert "succeeded" in page
        report = build.latest_report(platform, model.id)
        assert report is not None
        assert report.passed, report.notes

        mo.post(f"/console/models/{model.id}/promote", ok=False)
        mo.post(
            f"/console/models/{model.id}/request-promotion",
            red_team_results="memorisation, PII and permission probes clean",
        )
        approve(platform, mia, GateType.MODEL_PROMOTION, model.id)
        approve(platform, dex, GateType.MODEL_PROMOTION, model.id)
        mo.post(f"/console/models/{model.id}/promote")
        assert platform.store.get_model(model.id).status == "promoted"

        answer = mo.post("/console/ask", dataset_id=dataset_id, question="how do I reset my VPN")
        assert "Sources" in answer

        page = mo.get("/console/audit")
        assert "The hash chain is intact" in page
        assert platform.audit.verify().ok
        actors = {event.actor for event in platform.audit.recent(limit=2000)}
        person_ids = {
            row["id"] for row in platform.store.conn.execute("SELECT id FROM people").fetchall()
        }
        assert len(actors & person_ids) >= 5
