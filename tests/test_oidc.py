"""Signing in through an identity provider, against an issuer that behaves like one.

The stub below is not a mock of our own code: it is an HTTP server that publishes
a key set, hands out authorization codes with a PKCE challenge, and signs real
RS256 tokens with a real private key. The refusals here are therefore refused by
the verifier, not by a stand-in that was told to fail.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import PASSWORD, enroll, sign_in
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.services import oidc
from internal_ai.services.identity import IdentityService
from internal_ai.services.permissions import PermissionDenied

CLIENT_ID = "internal-ai-console"
REDIRECT = "http://127.0.0.1:8000/console/signin/oidc/callback"
SUBJECT = "00u1-stub-subject"
CALLBACK = "/console/signin/oidc/callback"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _number(value: int) -> str:
    return _b64(value.to_bytes((value.bit_length() + 7) // 8, "big"))


class StubIssuer:
    """An issuer that publishes keys, hands out codes and signs RS256 tokens."""

    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "stub-key-1"
        self.codes: dict[str, dict[str, Any]] = {}
        self.document_issuer = ""
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._routes())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def document(self) -> dict[str, Any]:
        return {
            "issuer": self.document_issuer or self.url,
            "authorization_endpoint": f"{self.url}/authorize",
            "token_endpoint": f"{self.url}/token",
            "jwks_uri": f"{self.url}/jwks",
        }

    def key_set(self) -> dict[str, Any]:
        numbers = self.key.public_key().public_numbers()
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": self.kid,
                    "use": "sig",
                    "alg": "RS256",
                    "n": _number(numbers.n),
                    "e": _number(numbers.e),
                }
            ]
        }

    def issue(self, challenge: str, nonce: str, **overrides: Any) -> str:
        code = f"code-{len(self.codes) + 1}"
        self.codes[code] = {"challenge": challenge, "nonce": nonce, "claims": overrides}
        return code

    def id_token(
        self, payload: dict[str, Any], algorithm: str = "RS256", kid: str | None = None
    ) -> str:
        header: dict[str, Any] = {"alg": algorithm, "typ": "JWT"}
        if algorithm != "none":
            header["kid"] = self.kid if kid is None else kid
        signing_input = _b64(json.dumps(header).encode()) + "." + _b64(json.dumps(payload).encode())
        if algorithm == "none":
            return signing_input + "."
        signature = self.key.sign(
            signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256()
        )
        return signing_input + "." + _b64(signature)

    def _routes(self) -> type[BaseHTTPRequestHandler]:
        issuer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def reply(self, status: int, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/.well-known/openid-configuration":
                    self.reply(200, issuer.document())
                elif path == "/jwks":
                    self.reply(200, issuer.key_set())
                else:
                    self.reply(404, {"error": "not_found"})

            def do_POST(self) -> None:
                size = int(self.headers.get("Content-Length", "0"))
                form = parse_qs(self.rfile.read(size).decode())
                code = (form.get("code") or [""])[0]
                verifier = (form.get("code_verifier") or [""])[0]
                record = issuer.codes.pop(code, None)
                if record is None:
                    return self.reply(400, {"error": "invalid_grant"})
                challenge = _b64(hashlib.sha256(verifier.encode()).digest())
                if challenge != record["challenge"]:
                    return self.reply(400, {"error": "invalid_grant", "hint": "pkce"})
                now = int(time.time())
                payload: dict[str, Any] = {
                    "iss": issuer.url,
                    "aud": CLIENT_ID,
                    "sub": SUBJECT,
                    "iat": now,
                    "exp": now + 300,
                    "nonce": record["nonce"],
                }
                payload.update(record["claims"])
                self.reply(200, {"id_token": issuer.id_token(payload), "token_type": "Bearer"})

        return Handler


@pytest.fixture()
def issuer() -> Any:
    server = StubIssuer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(autouse=True)
def no_ambient_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's shell cannot change what these tests see."""
    for name in (
        oidc.ENV_ISSUER,
        oidc.ENV_CLIENT_ID,
        oidc.ENV_CLIENT_SECRET,
        oidc.ENV_REDIRECT_URI,
    ):
        monkeypatch.delenv(name, raising=False)


def configured(monkeypatch: pytest.MonkeyPatch, issuer: StubIssuer) -> oidc.OidcConfig:
    monkeypatch.setenv(oidc.ENV_ISSUER, issuer.url)
    monkeypatch.setenv(oidc.ENV_CLIENT_ID, CLIENT_ID)
    monkeypatch.setenv(oidc.ENV_REDIRECT_URI, REDIRECT)
    config = oidc.OidcConfig.configured()
    assert config is not None
    return config


def claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    body: dict[str, Any] = {
        "iss": "",
        "aud": CLIENT_ID,
        "sub": SUBJECT,
        "iat": now,
        "exp": now + 300,
        "nonce": "nonce-1",
    }
    body.update(overrides)
    return body


def start(client: TestClient) -> tuple[str, str, str]:
    """Begin a sign-in and return what the browser would have taken to the provider."""
    response = client.get("/console/signin/oidc", follow_redirects=False)
    assert response.status_code == 303, response.text
    query = parse_qs(urlsplit(response.headers["location"]).query)
    return query["state"][0], query["nonce"][0], query["code_challenge"][0]


def test_a_linked_person_signs_in_through_the_provider(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        admin = enroll(platform, "Ada Admin", ["program_lead"])
        person = enroll(platform, "Pia Person", ["ml_lead"])
        IdentityService(platform).link_identity(person.id, issuer.url, SUBJECT, admin, "pia@corp")

        state, nonce, challenge = start(client)
        code = issuer.issue(challenge, nonce)
        response = client.get(f"{CALLBACK}?code={code}&state={state}", follow_redirects=False)
        assert response.status_code == 303, response.text
        assert response.headers["location"] == "/console/home"
        assert client.cookies.get("iai_session")

        page = client.get("/console/home")
        assert page.status_code == 200
        assert person.name in page.text

        signed_in = platform.store.conn.execute(
            "SELECT details FROM audit_events WHERE action = 'account.signed_in'"
            " ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        assert json.loads(signed_in["details"])["method"] == "oidc"
        assert IdentityService(platform).identities_for(person.id)[0]["email"] == "pia@corp"


def test_the_state_of_a_sign_in_cannot_be_used_twice(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        admin = enroll(platform, "Ada Admin", ["program_lead"])
        person = enroll(platform, "Pia Person", ["ml_lead"])
        IdentityService(platform).link_identity(person.id, issuer.url, SUBJECT, admin)

        state, nonce, challenge = start(client)
        first = issuer.issue(challenge, nonce)
        assert client.get(f"{CALLBACK}?code={first}&state={state}").status_code == 200

        second = issuer.issue(challenge, nonce)
        replay = client.get(f"{CALLBACK}?code={second}&state={state}")
        assert replay.status_code == 400
        assert "expired or was already used" in replay.text


def test_an_identity_nobody_linked_is_refused(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        enroll(app.state.platform, "Pia Person", ["ml_lead"])
        state, nonce, challenge = start(client)
        code = issuer.issue(challenge, nonce, sub="stranger-subject")
        response = client.get(f"{CALLBACK}?code={code}&state={state}")
        assert response.status_code == 401
        assert "not linked to a person here yet" in response.text
        # The refusal names the subject, which is the person's own identifier and
        # the one thing an administrator needs to link them.
        assert "stranger-subject" in response.text
        assert not client.cookies.get("iai_session")


def test_a_disabled_account_cannot_sign_in_through_the_provider(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        admin = enroll(platform, "Ada Admin", ["program_lead"])
        person = enroll(platform, "Pia Person", ["ml_lead"])
        identity = IdentityService(platform)
        identity.link_identity(person.id, issuer.url, SUBJECT, admin)
        identity.set_disabled(person.id, True, admin)

        state, nonce, challenge = start(client)
        code = issuer.issue(challenge, nonce)
        response = client.get(f"{CALLBACK}?code={code}&state={state}")
        assert response.status_code == 401
        assert "disabled" in response.text
        assert not client.cookies.get("iai_session")


def test_signing_in_replaces_a_session_from_before_the_sign_in(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        admin = enroll(platform, "Ada Admin", ["program_lead"])
        person = enroll(platform, "Pia Person", ["ml_lead"])
        IdentityService(platform).link_identity(person.id, issuer.url, SUBJECT, admin)

        assert sign_in(client, admin.name, PASSWORD).status_code == 303
        before = client.cookies.get("iai_session")
        assert before

        state, nonce, challenge = start(client)
        code = issuer.issue(challenge, nonce)
        assert client.get(f"{CALLBACK}?code={code}&state={state}").status_code == 200
        after = client.cookies.get("iai_session")
        assert after and after != before
        assert IdentityService(platform).resume_session(before) is None
        assert IdentityService(platform).resume_session(after) is not None


def test_linking_an_identity_needs_the_people_role(platform, issuer: StubIssuer) -> None:
    finance = enroll(platform, "Finn Finance", ["finance"], ["finance"])
    person = enroll(platform, "Pia Person", ["ml_lead"])
    identity = IdentityService(platform)

    with pytest.raises(PermissionDenied):
        identity.link_identity(person.id, issuer.url, SUBJECT, finance)
    assert identity.identities_for(person.id) == []

    denied = platform.store.conn.execute(
        "SELECT outcome, details FROM audit_events WHERE action = 'console.denied'"
        " ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert denied is not None and denied["outcome"] == "denied"
    assert json.loads(denied["details"])["action"] == "manage_people"


def test_one_subject_cannot_belong_to_two_people(platform, issuer: StubIssuer) -> None:
    admin = enroll(platform, "Ada Admin", ["program_lead"])
    first = enroll(platform, "Pia Person", ["ml_lead"])
    second = enroll(platform, "Sam Scientist", ["ml_lead"])
    identity = IdentityService(platform)
    identity.link_identity(first.id, issuer.url, SUBJECT, admin)

    with pytest.raises(ValueError, match="already linked to another person"):
        identity.link_identity(second.id, issuer.url, SUBJECT, admin)


def test_one_person_cannot_hold_two_subjects_from_the_same_issuer(
    platform, issuer: StubIssuer
) -> None:
    admin = enroll(platform, "Ada Admin", ["program_lead"])
    person = enroll(platform, "Pia Person", ["ml_lead"])
    identity = IdentityService(platform)
    identity.link_identity(person.id, issuer.url, SUBJECT, admin)

    with pytest.raises(ValueError, match="already linked to subject"):
        identity.link_identity(person.id, issuer.url, "another-subject", admin)


def test_unlinking_revokes_the_sessions_it_opened(platform, issuer: StubIssuer) -> None:
    admin = enroll(platform, "Ada Admin", ["program_lead"])
    person = enroll(platform, "Pia Person", ["ml_lead"])
    identity = IdentityService(platform)
    identity.link_identity(person.id, issuer.url, SUBJECT, admin)
    session = identity.start_session(person)

    identity.unlink_identity(person.id, issuer.url, admin)

    assert identity.resume_session(session.token) is None
    assert identity.identities_for(person.id) == []


def test_a_token_signed_with_an_unpublished_key_is_refused(
    issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = oidc.OidcClient(configured(monkeypatch, issuer))
    token = issuer.id_token(claims(iss=issuer.url), kid="a-key-nobody-published")

    with pytest.raises(oidc.OidcDenied, match="does not publish"):
        client.verify(token, "nonce-1")


def test_an_unsigned_token_is_refused(issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch) -> None:
    client = oidc.OidcClient(configured(monkeypatch, issuer))
    token = issuer.id_token(claims(iss=issuer.url), algorithm="none")

    with pytest.raises(oidc.OidcDenied, match="accepts RS256"):
        client.verify(token, "nonce-1")


def test_a_token_signed_by_a_different_key_is_refused(
    issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = oidc.OidcClient(configured(monkeypatch, issuer))
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    header = _b64(json.dumps({"alg": "RS256", "typ": "JWT", "kid": issuer.kid}).encode())
    body = _b64(json.dumps(claims(iss=issuer.url)).encode())
    signature = other.sign(f"{header}.{body}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())

    with pytest.raises(oidc.OidcDenied, match="signature does not match"):
        client.verify(f"{header}.{body}.{_b64(signature)}", "nonce-1")


def test_a_token_for_another_client_is_refused(
    issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = oidc.OidcClient(configured(monkeypatch, issuer))

    with pytest.raises(oidc.OidcDenied, match="another client"):
        client.verify(issuer.id_token(claims(iss=issuer.url, aud="someone-else")), "nonce-1")


def test_a_token_that_answers_another_nonce_is_refused(
    issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = oidc.OidcClient(configured(monkeypatch, issuer))

    with pytest.raises(oidc.OidcDenied, match="different sign-in"):
        client.verify(issuer.id_token(claims(iss=issuer.url, nonce="replayed")), "nonce-1")


def test_an_expired_token_is_refused(issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch) -> None:
    client = oidc.OidcClient(configured(monkeypatch, issuer))
    stale = claims(iss=issuer.url, exp=int(time.time()) - 120)

    with pytest.raises(oidc.OidcDenied, match="has expired"):
        client.verify(issuer.id_token(stale), "nonce-1")


def test_a_token_from_another_issuer_is_refused(
    issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = oidc.OidcClient(configured(monkeypatch, issuer))

    with pytest.raises(oidc.OidcDenied, match="was issued by"):
        client.verify(issuer.id_token(claims(iss="https://evil.example")), "nonce-1")


def test_a_discovery_document_for_another_issuer_is_refused(
    issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuer.document_issuer = "https://evil.example"
    client = oidc.OidcClient(configured(monkeypatch, issuer))

    with pytest.raises(oidc.OidcDenied, match="not the configured"):
        client.discovery()


def test_a_plain_http_provider_away_from_loopback_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(oidc.ENV_ISSUER, "http://idp.corp.example")
    monkeypatch.setenv(oidc.ENV_CLIENT_ID, CLIENT_ID)
    monkeypatch.setenv(oidc.ENV_REDIRECT_URI, "https://console.corp.example/console/callback")
    config = oidc.OidcConfig.configured()
    assert config is not None

    with pytest.raises(oidc.OidcDenied, match="must use https"):
        config.validated()


def test_a_half_configured_provider_is_refused_rather_than_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(oidc.ENV_ISSUER, "https://idp.corp.example")

    with pytest.raises(oidc.OidcDenied, match=oidc.ENV_CLIENT_ID):
        oidc.OidcConfig.configured()


def test_the_signin_page_offers_the_provider_only_when_one_is_configured(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        enroll(app.state.platform, "Pia Person", ["ml_lead"])
        assert "/console/signin/oidc" not in client.get("/console/signin").text

        monkeypatch.setenv(oidc.ENV_ISSUER, issuer.url)
        monkeypatch.setenv(oidc.ENV_CLIENT_ID, CLIENT_ID)
        monkeypatch.setenv(oidc.ENV_REDIRECT_URI, REDIRECT)
        page = client.get("/console/signin")
        assert "/console/signin/oidc" in page.text
        assert issuer.url in page.text


def test_the_people_page_links_an_identity_and_refuses_the_wrong_role(
    tmp_path: Path, issuer: StubIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, issuer)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        admin = enroll(platform, "Ada Admin", ["program_lead"])
        person = enroll(platform, "Pia Person", ["ml_lead"])
        finance = enroll(platform, "Finn Finance", ["finance"], ["finance"])

        assert sign_in(client, admin.name, PASSWORD).status_code == 303
        csrf = client.get("/console/admin/people").text
        token = csrf.split('name="csrf_token" value="', 1)[1].split('"', 1)[0]
        response = client.post(
            f"/console/admin/people/{person.id}/identity",
            data={
                "step": "link",
                "issuer": issuer.url,
                "subject": SUBJECT,
                "email": "pia@corp.example",
                "csrf_token": token,
            },
        )
        assert response.status_code == 200
        assert 'class="flash flash-ok"' in response.text
        assert IdentityService(platform).identities_for(person.id)[0]["subject"] == SUBJECT

        client.cookies.clear()
        assert sign_in(client, finance.name, PASSWORD).status_code == 303
        page = client.get("/console/admin/people").text
        token = page.split('name="csrf_token" value="', 1)[1].split('"', 1)[0]
        refused = client.post(
            f"/console/admin/people/{person.id}/identity",
            data={"step": "link", "issuer": issuer.url, "subject": "x", "csrf_token": token},
        )
        assert refused.status_code == 200
        assert 'class="flash flash-error"' in refused.text
        assert len(IdentityService(platform).identities_for(person.id)) == 1
