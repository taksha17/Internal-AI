"""Signing in through the customer's identity provider.

The password form is what an install with no directory uses. When a provider is
configured, the console hands the browser over and takes back an authorization
code: the code is exchanged for an id token, the token's signature is checked
against the keys the provider publishes, and only then is its subject looked up
in the roster here.

A token is never trusted because it arrived over TLS. It is trusted because it
verifies against a key published for the issuer named in this install's
configuration, because it answers the nonce this console generated for that one
sign-in, and because its subject is bound to a person in this database. Each of
those is checked, in that order, and a failure refuses the sign-in rather than
falling back to something weaker.

The provider's URL comes from configuration, never from a request, so a sign-in
cannot be pointed at a host a caller chose. Nothing here caches a session: it
maps one verified identity onto one person and returns, and the session rules
(including revocation) stay in `IdentityService`, where they already live.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

ENV_ISSUER = "INTERNAL_AI_OIDC_ISSUER"
ENV_CLIENT_ID = "INTERNAL_AI_OIDC_CLIENT_ID"
ENV_CLIENT_SECRET = "INTERNAL_AI_OIDC_CLIENT_SECRET"
ENV_REDIRECT_URI = "INTERNAL_AI_OIDC_REDIRECT_URI"

STATE_SECONDS = 5 * 60
"""How long a started sign-in may sit before its callback is refused."""

CLOCK_SKEW_SECONDS = 60
"""What a provider's clock may be behind this machine's before a token fails."""

ALLOWED_ALGORITHMS = {"RS256": "sha256", "RS384": "sha384", "RS512": "sha512"}
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class OidcDenied(RuntimeError):
    """A sign-in was refused rather than finished with something unverified."""


@dataclass(frozen=True, slots=True)
class OidcConfig:
    issuer: str
    client_id: str
    redirect_uri: str
    client_secret: str = ""

    @classmethod
    def configured(cls) -> OidcConfig | None:
        """The provider this install signs in through, or None if it uses passwords.

        A half-filled configuration is refused instead of quietly leaving the
        password form in place: an administrator who set an issuer meant it.
        """
        issuer = os.environ.get(ENV_ISSUER, "").strip()
        client_id = os.environ.get(ENV_CLIENT_ID, "").strip()
        redirect = os.environ.get(ENV_REDIRECT_URI, "").strip()
        secret = os.environ.get(ENV_CLIENT_SECRET, "").strip()
        if not issuer and not client_id and not redirect:
            return None
        missing = [
            name
            for name, value in (
                (ENV_ISSUER, issuer),
                (ENV_CLIENT_ID, client_id),
                (ENV_REDIRECT_URI, redirect),
            )
            if not value
        ]
        if missing:
            raise OidcDenied(
                "the identity provider is only half configured; set " + ", ".join(missing)
            )
        return cls(issuer=issuer, client_id=client_id, redirect_uri=redirect, client_secret=secret)

    def validated(self) -> OidcConfig:
        """Refuse a provider address that cannot be verified over a private path."""
        for name, value in (
            ("issuer", self.issuer),
            ("redirect uri", self.redirect_uri),
        ):
            parts = urlsplit(value)
            if parts.scheme not in ("https", "http") or not parts.hostname:
                raise OidcDenied(f"the OIDC {name} must be an absolute URL")
            if parts.scheme == "http" and parts.hostname not in LOOPBACK_HOSTS:
                raise OidcDenied(
                    f"the OIDC {name} must use https unless it is a loopback address for testing"
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        """What the console may show: never the client secret."""
        return {
            "issuer": self.issuer,
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "public_client": not self.client_secret,
        }


@dataclass(frozen=True, slots=True)
class Discovery:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str


def fetch_json(url: str) -> dict[str, Any]:
    """One GET, no redirects followed, so a moved document is a refusal."""
    try:
        response = httpx.get(url, timeout=10.0, follow_redirects=False)
    except httpx.HTTPError as exc:
        raise OidcDenied(f"the identity provider did not answer at {url}: {exc}") from exc
    if response.status_code != 200:
        raise OidcDenied(f"the identity provider answered {response.status_code} at {url}")
    try:
        document = response.json()
    except ValueError as exc:
        raise OidcDenied(f"the identity provider did not send JSON at {url}") from exc
    if not isinstance(document, dict):
        raise OidcDenied(f"the identity provider sent something other than an object at {url}")
    return document


class OidcClient:
    """One sign-in against one configured provider.

    A client is built per request, so keys and the discovery document are read
    fresh: a provider that rotates its signing key is picked up by the next
    person who signs in, with no restart.
    """

    def __init__(
        self,
        config: OidcConfig,
        *,
        fetch: Callable[[str], dict[str, Any]] = fetch_json,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config.validated()
        self._fetch = fetch
        self.clock = clock
        self._discovery: Discovery | None = None
        self._keys: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------- discovery

    def discovery(self) -> Discovery:
        if self._discovery is not None:
            return self._discovery
        document = self._fetch(self.config.issuer + "/.well-known/openid-configuration")
        issuer = str(document.get("issuer", ""))
        if issuer != self.config.issuer:
            raise OidcDenied(
                f"the discovery document names issuer {issuer or 'nothing'}, "
                f"not the configured {self.config.issuer}"
            )
        found = Discovery(
            issuer=issuer,
            authorization_endpoint=str(document.get("authorization_endpoint", "")),
            token_endpoint=str(document.get("token_endpoint", "")),
            jwks_uri=str(document.get("jwks_uri", "")),
        )
        for name in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            parts = urlsplit(getattr(found, name))
            if parts.scheme != "https" and parts.hostname not in LOOPBACK_HOSTS:
                raise OidcDenied(
                    f"the discovery document gives {name} as "
                    f"{getattr(found, name) or 'nothing'}, which is not https"
                )
        self._discovery = found
        return found

    def keys(self) -> dict[str, dict[str, Any]]:
        if self._keys:
            return self._keys
        document = self._fetch(self.discovery().jwks_uri)
        published = document.get("keys")
        if not isinstance(published, list):
            raise OidcDenied("the identity provider published no key set")
        for item in published:
            if isinstance(item, dict) and isinstance(item.get("kid"), str) and item["kid"]:
                self._keys[item["kid"]] = item
        if not self._keys:
            raise OidcDenied("the identity provider publishes no signing keys")
        return self._keys

    # ------------------------------------------------------------- sign-in

    def begin(self, state: str, nonce: str) -> tuple[str, str]:
        """The page to send the browser to, and the verifier to keep for the callback."""
        verifier = secrets.token_urlsafe(48)
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.config.client_id,
                "redirect_uri": self.config.redirect_uri,
                "scope": "openid email profile",
                "state": state,
                "nonce": nonce,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{self.discovery().authorization_endpoint}?{query}", verifier

    def complete(self, code: str, verifier: str, nonce: str) -> dict[str, Any]:
        if not code:
            raise OidcDenied("the identity provider returned no authorization code")
        token = self._exchange(code, verifier)
        id_token = token.get("id_token")
        if not isinstance(id_token, str) or not id_token:
            raise OidcDenied("the identity provider returned no id token")
        return self.verify(id_token, nonce)

    def _exchange(self, code: str, verifier: str) -> dict[str, Any]:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "code_verifier": verifier,
        }
        headers = {"Accept": "application/json"}
        if self.config.client_secret:
            pair = f"{_quote(self.config.client_id)}:{_quote(self.config.client_secret)}".encode(
                "ascii"
            )
            headers["Authorization"] = "Basic " + base64.b64encode(pair).decode("ascii")
        try:
            response = httpx.post(
                self.discovery().token_endpoint, data=form, headers=headers, timeout=10.0
            )
        except httpx.HTTPError as exc:
            raise OidcDenied(
                f"the identity provider did not answer the code exchange: {exc}"
            ) from exc
        if response.status_code != 200:
            raise OidcDenied(
                f"the identity provider refused the code exchange with {response.status_code}"
            )
        try:
            document = response.json()
        except ValueError as exc:
            raise OidcDenied("the code exchange did not answer with JSON") from exc
        if not isinstance(document, dict):
            raise OidcDenied("the code exchange did not answer with an object")
        return document

    # -------------------------------------------------------------- token

    def verify(self, id_token: str, nonce: str) -> dict[str, Any]:
        """Check a token's key, signature and claims, and return what it states."""
        parts = id_token.split(".")
        if len(parts) != 3:
            raise OidcDenied("the id token is not a signed JWT")
        header = _json_part(parts[0], "header")
        payload = _json_part(parts[1], "payload")
        algorithm = header.get("alg")
        if not isinstance(algorithm, str) or algorithm not in ALLOWED_ALGORITHMS:
            shown = algorithm if isinstance(algorithm, str) and algorithm else "no algorithm"
            raise OidcDenied(
                f"the id token is signed with {shown}; this install accepts "
                + ", ".join(sorted(ALLOWED_ALGORITHMS))
            )
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise OidcDenied("the id token names no key, so it cannot be checked")
        key = self.keys().get(kid)
        if key is None:
            raise OidcDenied(
                f"the id token is signed with key {kid}, which the provider does not publish"
            )
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        verify_rsa(key, algorithm, signing_input, _b64url_decode(parts[2]))
        self.check_claims(payload, nonce)
        return payload

    def check_claims(self, payload: dict[str, Any], nonce: str) -> None:
        """Every claim this install relies on, checked against configuration."""
        issuer = payload.get("iss")
        if issuer != self.config.issuer:
            raise OidcDenied(
                f"the id token was issued by {issuer or 'nobody'}, not {self.config.issuer}"
            )
        audience = payload.get("aud")
        audiences = [audience] if isinstance(audience, str) else list(audience or [])
        if self.config.client_id not in audiences:
            raise OidcDenied("the id token was issued for another client")
        authorized = payload.get("azp")
        if authorized is not None and authorized != self.config.client_id:
            raise OidcDenied("the id token was authorized for another party")
        now = self.clock()
        expires = payload.get("exp")
        if not isinstance(expires, int | float):
            raise OidcDenied("the id token states no expiry")
        if now - CLOCK_SKEW_SECONDS >= expires:
            raise OidcDenied("the id token has expired")
        issued = payload.get("iat")
        if isinstance(issued, int | float) and issued > now + CLOCK_SKEW_SECONDS:
            raise OidcDenied("the id token is dated in the future")
        not_before = payload.get("nbf")
        if isinstance(not_before, int | float) and not_before > now + CLOCK_SKEW_SECONDS:
            raise OidcDenied("the id token is not valid yet")
        sent = payload.get("nonce")
        if not nonce or not isinstance(sent, str) or not hmac.compare_digest(sent, nonce):
            raise OidcDenied("the id token answers a different sign-in request")
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise OidcDenied("the id token names no subject")


def verify_rsa(key: dict[str, Any], algorithm: str, signing_input: bytes, signature: bytes) -> None:
    """Check an RSASSA-PKCS1-v1_5 signature against a published key.

    `cryptography` is imported here rather than at the top of the module so an
    install that signs in with passwords does not carry it.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise OidcDenied(
            "checking an id token needs the oidc extra: pip install 'internal-ai[oidc]'"
        ) from exc
    if key.get("kty") != "RSA":
        raise OidcDenied(f"the provider published a {key.get('kty')} key, not an RSA key")
    try:
        public_key = rsa.RSAPublicNumbers(  # type: ignore[no-untyped-call]
            _b64url_int(key["e"]), _b64url_int(key["n"])
        ).public_key()
    except (KeyError, TypeError, ValueError) as exc:
        raise OidcDenied("the provider published a key this install cannot read") from exc
    digest = {
        "RS256": hashes.SHA256(),
        "RS384": hashes.SHA384(),
        "RS512": hashes.SHA512(),
    }[algorithm]
    try:
        public_key.verify(signature, signing_input, padding.PKCS1v15(), digest)
    except InvalidSignature as exc:
        raise OidcDenied(
            "the id token's signature does not match the key the provider publishes"
        ) from exc


def new_state() -> tuple[str, str]:
    """A state and a nonce for one sign-in; both are single use."""
    return secrets.token_urlsafe(24), secrets.token_urlsafe(24)


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except (ValueError, TypeError) as exc:
        raise OidcDenied("the id token is not encoded the way a JWT is") from exc


def _b64url_int(value: Any) -> int:
    if not isinstance(value, str) or not value:
        raise ValueError("not a base64url number")
    return int.from_bytes(_b64url_decode(value), "big")


def _json_part(value: str, what: str) -> dict[str, Any]:
    try:
        parsed = json.loads(_b64url_decode(value))
    except ValueError as exc:
        raise OidcDenied(f"the id token's {what} is not JSON") from exc
    if not isinstance(parsed, dict):
        raise OidcDenied(f"the id token's {what} is not an object")
    return parsed
