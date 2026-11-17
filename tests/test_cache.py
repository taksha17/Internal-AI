"""The optional shared cache: what it saves, and what it refuses to be.

The Redis path is exercised through a fake client, because a cache that only exists
in production is a cache nobody has tested. The properties that matter are the ones
the install depends on when Redis is *not* there.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import enroll

from internal_ai import cache
from internal_ai.cache import CacheDenied, MemoryCache, RedisCache, build_cache
from internal_ai.services.identity import IdentityService


class FakeRedis:
    """Enough of a client for this module: strings, counters and set-nx."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}
        self.setex_calls: list[tuple[str, int, str]] = []

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool | None:
        if nx and key in self.values:
            return None
        self.values[key] = value
        if ex:
            self.expiries[key] = ex
        return True

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.setex_calls.append((key, ttl, value))
        self.values[key] = value
        self.expiries[key] = ttl

    def delete(self, key: str) -> None:
        self.values.pop(key, None)

    def pipeline(self) -> Any:
        client = self

        class Pipeline:
            def __init__(self) -> None:
                self.steps: list[tuple[str, str, int]] = []

            def incr(self, key: str) -> Pipeline:
                self.steps.append(("incr", key, 0))
                return self

            def expire(self, key: str, ttl: int, nx: bool = False) -> Pipeline:
                self.steps.append(("expire", key, ttl))
                return self

            def execute(self) -> list[Any]:
                results: list[Any] = []
                for op, key, ttl in self.steps:
                    if op == "incr":
                        current = int(client.values.get(key, "0")) + 1
                        client.values[key] = str(current)
                        results.append(current)
                    else:
                        client.expiries[key] = ttl
                        results.append(True)
                return results

        return Pipeline()


def test_the_local_cache_expires_on_its_own_clock() -> None:
    now = [1000.0]
    memory = MemoryCache(clock=lambda: now[0])
    memory.set("k", "v", 10)

    assert memory.get("k") == "v"
    now[0] += 11
    assert memory.get("k") is None


def test_a_counter_keeps_its_window() -> None:
    now = [1000.0]
    memory = MemoryCache(clock=lambda: now[0])

    assert memory.incr("c", 60) == 1
    assert memory.incr("c", 60) == 2
    now[0] += 61
    assert memory.incr("c", 60) == 1


def test_the_redis_path_sets_an_expiry_through_a_fake_client() -> None:
    client = FakeRedis()
    shared = RedisCache(client)

    shared.set("session:abc", "1", 60)
    shared.delete("session:abc")

    assert client.setex_calls == [("session:abc", 60, "1")]
    assert shared.get("session:abc") is None


def test_the_redis_counter_increments_through_a_pipeline() -> None:
    shared = RedisCache(FakeRedis())

    assert shared.incr("auth:a:b", 900) == 1
    assert shared.incr("auth:a:b", 900) == 2


def test_a_lock_is_taken_once() -> None:
    shared = RedisCache(FakeRedis())

    assert shared.acquire("audit-check", 60) is True
    assert shared.acquire("audit-check", 60) is False
    shared.release("audit-check")
    assert shared.acquire("audit-check", 60) is True


def test_no_url_means_a_local_cache() -> None:
    assert isinstance(build_cache(""), MemoryCache)
    assert isinstance(build_cache(None), MemoryCache)


def test_a_configured_url_without_the_driver_is_refused(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "redis", None)

    with pytest.raises(CacheDenied, match="redis package is not installed"):
        build_cache("redis://127.0.0.1:6379/0")


def test_a_shared_cache_throttles_signing_in_across_services(platform, enroll_person) -> None:
    """Two services, one cache: the counter is the part that has to be shared."""
    first = IdentityService(platform)
    second = IdentityService(platform)

    # A username that does not exist has no account row to lock, so the shared
    # counter is the only thing standing between one client and endless attempts.
    for _ in range(5):
        with pytest.raises(PermissionError, match="unknown account or wrong password"):
            second.authenticate("nobody", "wrong-password", client="10.0.0.9")

    with pytest.raises(PermissionError, match="too many failed attempts from this client"):
        first.authenticate("nobody", "wrong-password", client="10.0.0.9")

    # A real person from a different address never touched that counter.
    other = enroll_person(platform, "Bo Lead", ["ml_lead"])
    actor = first.authenticate(other.name, "correct-horse-battery", client="10.0.0.10")
    assert actor.id == other.id


def test_a_revoked_session_is_refused_even_before_the_row_is_missed(
    platform, enroll_person
) -> None:
    person = enroll_person(platform, "Ari Lead", ["ml_lead"])
    identity = IdentityService(platform)
    session = identity.start_session(person)

    identity.end_session(session.token)

    # Put the row back, as a replica that has not caught up would still serve it.
    platform.store.conn.execute(
        "INSERT INTO sessions (token_hash, person_id, csrf_token, created_at, last_seen,"
        " expires_at) VALUES (?,?,?,?,?,?)",
        (
            cache_key(session.token),
            person.id,
            session.csrf_token,
            0.0,
            0.0,
            10**12,
        ),
    )

    assert identity.resume_session(session.token) is None


def test_ending_every_session_mirrors_each_one(platform, enroll_person) -> None:
    person = enroll_person(platform, "Ari Lead", ["ml_lead"])
    identity = IdentityService(platform)
    first = identity.start_session(person)
    second = identity.start_session(person)

    identity.end_all_sessions(person.id)

    assert identity.resume_session(first.token) is None
    assert identity.resume_session(second.token) is None
    row = platform.store.conn.execute("SELECT count(*) AS n FROM sessions").fetchone()
    assert row["n"] == 0


def cache_key(token: str) -> str:
    from internal_ai.services.identity import token_hash

    return token_hash(token)


def test_the_platform_has_a_cache_even_with_nothing_configured(platform) -> None:
    assert platform.cache.get("missing") is None
    platform.cache.set("present", "1", 10)
    assert platform.cache.get("present") == "1"
    assert isinstance(platform.cache, cache.MemoryCache)


def test_an_enrolled_person_can_still_sign_in(platform) -> None:
    person = enroll(platform, "Ari Lead", ["ml_lead"])

    actor = IdentityService(platform).authenticate(person.name, "correct-horse-battery")

    assert actor.authenticated
