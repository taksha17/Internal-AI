"""A shared cache for things that can be recomputed, and nothing that cannot.

Redis is optional here, and the interface is served three ways so the choice is an
installation detail: a process-local dict with no configuration, Redis when
``INTERNAL_AI_REDIS_URL`` names one, and the database for everything the answer has
to be right about. The session row stays in the database and remains the authority;
the cache only saves the lookup that resolves it, so a Redis restart costs a slower
request and never a revoked session coming back.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

ENV_REDIS_URL = "INTERNAL_AI_REDIS_URL"

# How long a resolved actor may be reused. A role change deletes the entry, so this
# is the ceiling on how wrong a cache that missed the delete can be.
ACTOR_TTL_SECONDS = 30


class CacheDenied(RuntimeError):
    """A cache was configured and cannot be served, so the install refuses it."""


class Cache(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...

    def incr(self, key: str, ttl_seconds: int) -> int: ...

    def acquire(self, name: str, ttl_seconds: int) -> bool: ...

    def release(self, name: str) -> None: ...


@dataclass(slots=True)
class MemoryCache:
    """A cache that dies with the process, which is what an install without Redis has.

    It still makes the two cheaper paths real: a rate-limit counter and a session
    lookup within one process. It proves nothing across processes, and the tests say
    so rather than pretending this is the same thing as a server.
    """

    clock: Any = time.time
    entries: dict[str, tuple[str, float]] = field(default_factory=dict)
    counters: dict[str, tuple[int, float]] = field(default_factory=dict)

    def _live(self, store: dict[str, Any], key: str) -> Any:
        entry = store.get(key)
        if entry is None:
            return None
        if entry[1] and entry[1] <= self.clock():
            store.pop(key, None)
            return None
        return entry

    def get(self, key: str) -> str | None:
        entry = self._live(self.entries, key)
        return entry[0] if entry is not None else None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.entries[key] = (value, self.clock() + ttl_seconds if ttl_seconds else 0.0)

    def delete(self, key: str) -> None:
        self.entries.pop(key, None)
        self.counters.pop(key, None)

    def incr(self, key: str, ttl_seconds: int) -> int:
        entry = self._live(self.counters, key)
        count = (entry[0] if entry is not None else 0) + 1
        expires = entry[1] if entry is not None else self.clock() + ttl_seconds
        self.counters[key] = (count, expires)
        return count

    def acquire(self, name: str, ttl_seconds: int) -> bool:
        if self.get(f"lock:{name}") is not None:
            return False
        self.set(f"lock:{name}", "1", ttl_seconds)
        return True

    def release(self, name: str) -> None:
        self.delete(f"lock:{name}")


class RedisCache:
    """The same interface over a Redis client, which is the only shared one here.

    The client is injected rather than built from a URL in the constructor so the
    whole path is exercised by a fake in the suite: a Redis that is only reachable
    from production is a Redis nobody tests against.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    @classmethod
    def from_url(cls, url: str) -> RedisCache:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise CacheDenied(
                f"{ENV_REDIS_URL} is set but the redis package is not installed; install "
                "the 'redis' extra or unset the variable. Falling back silently would make "
                "a shared cache look like a local one"
            ) from exc
        return cls(redis.Redis.from_url(url, decode_responses=True))

    def get(self, key: str) -> str | None:
        value = self.client.get(key)
        return None if value is None else str(value)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if ttl_seconds:
            self.client.setex(key, ttl_seconds, value)
        else:
            self.client.set(key, value)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def incr(self, key: str, ttl_seconds: int) -> int:
        pipeline = self.client.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, ttl_seconds, nx=True)
        count, _ = pipeline.execute()
        return int(count)

    def acquire(self, name: str, ttl_seconds: int) -> bool:
        return bool(self.client.set(f"lock:{name}", "1", nx=True, ex=ttl_seconds))

    def release(self, name: str) -> None:
        self.client.delete(f"lock:{name}")


def build_cache(url: str | None) -> Cache:
    """A shared cache when one is configured, a local one otherwise.

    A configured URL that cannot be served is refused rather than downgraded: an
    install that believes it has a shared cache and quietly does not is worse than
    one that fails at the first request.
    """
    configured = (url or "").strip()
    if not configured:
        return MemoryCache()
    return RedisCache.from_url(configured)


def cached_actor(cache: Cache, person_id: str) -> dict[str, Any] | None:
    payload = cache.get(f"actor:{person_id}")
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        cache.delete(f"actor:{person_id}")
        return None


def remember_actor(cache: Cache, person_id: str, payload: dict[str, Any]) -> None:
    cache.set(f"actor:{person_id}", json.dumps(payload, sort_keys=True), ACTOR_TTL_SECONDS)


def forget_actor(cache: Cache, person_id: str) -> None:
    cache.delete(f"actor:{person_id}")
