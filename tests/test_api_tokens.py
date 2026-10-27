"""Personal API tokens are mortal: 90 days, then they stop working."""

from __future__ import annotations

from tests.conftest import enroll

from internal_ai.context import Platform
from internal_ai.services.actor import Actor
from internal_ai.services.identity import TOKEN_LIFETIME_DAYS, IdentityService, token_hash


def test_a_fresh_token_works_and_carries_an_expiry(platform: Platform) -> None:
    actor = enroll(platform, "Ada Admin", ["program_lead"])
    identity = IdentityService(platform)
    token = identity.create_token(actor, "scripts")

    assert isinstance(identity.token_actor(token), Actor)
    listed = {item["id"]: item for item in identity.list_tokens(actor.id)}
    assert len(listed) == 1
    assert listed[next(iter(listed))]["expires_at"]


def test_an_expired_token_is_refused(platform: Platform) -> None:
    actor = enroll(platform, "Ada Admin", ["program_lead"])
    identity = IdentityService(platform)
    token = identity.create_token(actor, "scripts")
    platform.store.conn.execute(
        "UPDATE api_tokens SET expires_at = ? WHERE token_hash = ?",
        ("2000-01-01T00:00:00+00:00", token_hash(token)),
    )

    assert identity.token_actor(token) is None


def test_token_lifetime_is_ninety_days() -> None:
    assert TOKEN_LIFETIME_DAYS == 90
