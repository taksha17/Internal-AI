"""A provider token on the JSON API, checked the same way and worth the same.

The stub issuer from `test_oidc` signs the tokens here too, so what is verified is
a real signature over real claims rather than a dict handed to a shortcut.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from conftest import enroll
from fastapi.testclient import TestClient
from test_oidc import StubIssuer

from internal_ai.api import create_app
from internal_ai.services import oidc
from internal_ai.services.identity import IdentityService

CONSOLE_CLIENT = "internal-ai-console"
API_CLIENT = "internal-ai-api"
REDIRECT = "http://127.0.0.1:8000/console/signin/oidc/callback"
SUBJECT = "00u1-stub-subject"


@pytest.fixture()
def issuer() -> Any:
    server = StubIssuer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(autouse=True)
def no_ambient_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        oidc.ENV_ISSUER,
        oidc.ENV_CLIENT_ID,
        oidc.ENV_CLIENT_SECRET,
        oidc.ENV_REDIRECT_URI,
        oidc.ENV_API_CLIENT_ID,
    ):
        monkeypatch.delenv(name, raising=False)


def configured(monkeypatch: pytest.MonkeyPatch, issuer: StubIssuer) -> None:
    monkeypatch.setenv(oidc.ENV_ISSUER, issuer.url)
    monkeypatch.setenv(oidc.ENV_CLIENT_ID, CONSOLE_CLIENT)
    monkeypatch.setenv(oidc.ENV_API_CLIENT_ID, API_CLIENT)
    monkeypatch.setenv(oidc.ENV_REDIRECT_URI, REDIRECT)


def token(issuer: StubIssuer, **overrides: Any) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": issuer.url,
        "aud": API_CLIENT,
        "sub": SUBJECT,
        "iat": now,
        "exp": now + 300,
    }
    payload.update(overrides)
    return issuer.id_token(payload)


def bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


def linked(platform: Any, issuer: StubIssuer, roles: list[str], name: str = "Pia Person") -> Any:
    admin = enroll(platform, "Ada Admin", ["program_lead"])
    person = enroll(platform, name, roles)
    IdentityService(platform).link_identity(person.id, issuer.url, SUBJECT, admin)
    return person


def test_a_provider_token_signs_a_script_in(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        person = linked(app.state.platform, issuer, ["ml_lead"])

        response = client.get("/me", headers=bearer(token(issuer)))

        assert response.status_code == 200, response.text
        assert response.json()["id"] == person.id
        assert response.json()["roles"] == ["ml_lead"]
        assert response.json()["authenticated"] is True


def test_the_provider_token_gets_the_same_capability_check(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        admin = enroll(platform, "Ada Admin", ["program_lead"])
        finance = enroll(platform, "Finn Finance", ["finance"], ["finance"])
        lead = enroll(platform, "Lee Lead", ["program_lead"])
        identity = IdentityService(platform)
        identity.link_identity(finance.id, issuer.url, "subject-finance", admin)
        identity.link_identity(lead.id, issuer.url, "subject-lead", admin)

        refused = client.post(
            "/plan",
            headers=bearer(token(issuer, sub="subject-finance")),
            json={"goal": "summarise tickets"},
        )
        assert refused.status_code == 403, refused.text
        assert "submit plan" in refused.text

        accepted = client.post(
            "/plan",
            headers=bearer(token(issuer, sub="subject-lead")),
            json={"goal": "summarise tickets"},
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["tasks"]


def test_a_token_for_another_audience_is_refused(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        linked(app.state.platform, issuer, ["ml_lead"])

        response = client.get("/me", headers=bearer(token(issuer, aud=CONSOLE_CLIENT)))

        assert response.status_code == 401
        assert "issued for another client" in response.text


def test_an_expired_provider_token_is_refused(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        linked(app.state.platform, issuer, ["ml_lead"])

        stale = token(issuer, exp=int(time.time()) - 120)
        response = client.get("/me", headers=bearer(stale))

        assert response.status_code == 401
        assert "has expired" in response.text


def test_an_unsigned_provider_token_is_refused(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        linked(app.state.platform, issuer, ["ml_lead"])

        now = int(time.time())
        payload = {
            "iss": issuer.url,
            "aud": API_CLIENT,
            "sub": SUBJECT,
            "iat": now,
            "exp": now + 300,
        }
        response = client.get("/me", headers=bearer(issuer.id_token(payload, algorithm="none")))

        assert response.status_code == 401
        assert "accepts RS256" in response.text


def test_a_subject_nobody_linked_is_refused(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        enroll(app.state.platform, "Pia Person", ["ml_lead"])

        response = client.get("/me", headers=bearer(token(issuer, sub="stranger")))

        assert response.status_code == 401
        assert "not linked to a person here yet" in response.text


def test_a_disabled_account_is_refused_through_the_api(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        person = linked(app.state.platform, issuer, ["ml_lead"])
        admin = enroll(app.state.platform, "Ada Two", ["program_lead"])
        IdentityService(app.state.platform).set_disabled(person.id, True, admin)

        response = client.get("/me", headers=bearer(token(issuer)))

        assert response.status_code == 401
        assert "disabled" in response.text


def test_a_personal_token_still_works_beside_the_provider(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        person = linked(app.state.platform, issuer, ["ml_lead"])
        personal = IdentityService(app.state.platform).create_token(person, "tests")

        response = client.get("/me", headers=bearer(personal))

        assert response.status_code == 200
        assert response.json()["id"] == person.id


def test_a_jwt_is_refused_when_no_provider_is_configured(
    tmp_path: Path, issuer: StubIssuer
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        linked(app.state.platform, issuer, ["ml_lead"])

        response = client.get("/me", headers=bearer(token(issuer)))

        assert response.status_code == 401
        assert "invalid or revoked personal token" in response.text
