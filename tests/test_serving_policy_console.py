"""Who may change what a served model does with flagged text, and what is recorded."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.serving.guardrails import INSTALL_POLICY_KEY

FLASH_ERROR = re.compile(r'class="flash flash-error"[^>]*>([^<]*)')
URL = "/console/models/mdl_pol/guardrail-policy"
INSTALL_URL = "/console/models/guardrail-default"


def _app(tmp_path: Path):
    platform = Platform.create(root=tmp_path, llm_kind="scripted")
    platform.store.add_model(
        ModelVersion(
            id="mdl_pol",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            status="promoted",
        )
    )
    enroll(platform, "Mo Builder", ["ml_lead"])
    enroll(platform, "Dex Expert", ["domain_expert"])
    platform.close()
    return create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)


def _config(store, model_id: str = "mdl_pol") -> dict:
    model = store.get_model(model_id)
    assert model is not None
    return model.config


def test_an_ml_lead_sets_the_install_default_and_then_hands_it_back(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Mo Builder")
        csrf = session_csrf(client)
        store = app.state.platform.store

        assert f'action="{INSTALL_URL}"' in client.get("/console/models").text
        page = client.post(INSTALL_URL, data={"policy": "refuse", "csrf_token": csrf}).text
        assert not FLASH_ERROR.findall(page)
        assert "refuse across the install" in page
        assert store.kv_get(INSTALL_POLICY_KEY) == "refuse"

        page = client.post(INSTALL_URL, data={"policy": "", "csrf_token": csrf}).text
        assert not FLASH_ERROR.findall(page)
        assert store.kv_get(INSTALL_POLICY_KEY) == ""

        events = store.list_audit(action="serving.guardrail_default_set", limit=5)
        assert [event.details["after"] for event in reversed(events)] == ["refuse", "warn"]
        assert events[0].details["follows_environment"] is True
        assert events[1].details["before"] == "warn"


def test_an_unknown_install_default_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Mo Builder")
        page = client.post(
            INSTALL_URL, data={"policy": "block", "csrf_token": session_csrf(client)}
        )

        assert page.status_code == 200
        assert "unknown guardrail_policy" in page.text
        assert app.state.platform.store.kv_get(INSTALL_POLICY_KEY) is None


def test_a_domain_expert_cannot_set_the_install_default(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Dex Expert")
        store = app.state.platform.store
        page = client.post(
            INSTALL_URL, data={"policy": "refuse", "csrf_token": session_csrf(client)}
        )

        assert "needs ml_lead" in page.text
        assert f'action="{INSTALL_URL}"' not in client.get("/console/models").text
        assert store.kv_get(INSTALL_POLICY_KEY) is None
        denied = store.list_audit(action="console.denied", limit=5)
        assert denied[0].details["action"] == "set_serving_policy"


def test_an_ml_lead_can_choose_and_then_clear_the_policy(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Mo Builder")
        csrf = session_csrf(client)
        store = app.state.platform.store

        assert f'action="{URL}"' in client.get("/console/models").text
        page = client.post(URL, data={"policy": "refuse", "csrf_token": csrf}).text
        assert not FLASH_ERROR.findall(page)
        assert _config(store)["guardrail_policy"] == "refuse"

        page = client.post(URL, data={"policy": "", "csrf_token": csrf}).text
        assert not FLASH_ERROR.findall(page)
        assert "guardrail_policy" not in _config(store)

        events = store.list_audit(action="serving.guardrail_policy_set", limit=5)
        assert [event.details["after"] for event in reversed(events)] == ["refuse", ""]
        assert events[0].details["inherits_install_default"] is True


def test_an_unknown_policy_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Mo Builder")
        page = client.post(URL, data={"policy": "block", "csrf_token": session_csrf(client)})

        assert page.status_code == 200
        assert "unknown guardrail_policy" in page.text
        assert "guardrail_policy" not in _config(app.state.platform.store)


def test_a_domain_expert_cannot_set_the_policy(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Dex Expert")
        page = client.post(URL, data={"policy": "warn", "csrf_token": session_csrf(client)})
        store = app.state.platform.store

        assert "needs ml_lead" in page.text
        assert f'action="{URL}"' not in client.get("/console/models").text
        assert "guardrail_policy" not in _config(store)
        denied = store.list_audit(action="console.denied", limit=5)
        assert denied[0].details["action"] == "set_serving_policy"
