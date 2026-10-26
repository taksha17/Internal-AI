"""Model serving: authenticated API, rate limits, permission-aware retrieval.

Privacy rules that shape this module (PRD sections 8.5 and 13):
- every request is authenticated and rate limited per user
- request logs record metadata only, never raw query or document content
- retrieval filters documents by the access-control metadata inherited from the
  source system, so a user never sees a document their group cannot read
- rollback is a pointer change, so it takes effect within seconds
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from internal_ai.audit import AuditLog
from internal_ai.domain import AuditOutcome, ModelVersion
from internal_ai.store import Store

SUPERUSER_GROUP = "platform-admins"


@dataclass(slots=True)
class Principal:
    user_id: str
    groups: frozenset[str]
    api_key: str = ""

    def can_read(self, acl: list[str], sensitivity: str = "internal") -> bool:
        if SUPERUSER_GROUP in self.groups:
            return True
        if not acl:
            return True
        return bool(self.groups & set(acl))


class AuthError(Exception):
    pass


class RateLimitError(Exception):
    pass


class RateLimiter:
    """Sliding window limiter keyed by user."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.limit = requests_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, user_id: str, now: float | None = None) -> bool:
        moment = now if now is not None else time.monotonic()
        window = self._hits[user_id]
        while window and moment - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(moment)
        return True

    def remaining(self, user_id: str, now: float | None = None) -> int:
        moment = now if now is not None else time.monotonic()
        window = self._hits[user_id]
        while window and moment - window[0] > 60.0:
            window.popleft()
        return max(self.limit - len(window), 0)


def parse_api_keys(raw: str) -> dict[str, Principal]:
    """Parse serving keys from `key:user:group1,group2` pairs or JSON."""
    principals: dict[str, Principal] = {}
    raw = raw.strip()
    if not raw:
        return principals
    if raw.startswith("{"):
        payload = json.loads(raw)
        for key, item in payload.items():
            principals[key] = Principal(
                user_id=str(item.get("user_id", "unknown")),
                groups=frozenset(item.get("groups", [])),
                api_key=key,
            )
        return principals
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 2:
            continue
        key, user_id = parts[0], parts[1]
        groups = parts[2].split("|") if len(parts) > 2 and parts[2] else []
        principals[key] = Principal(user_id=user_id, groups=frozenset(groups), api_key=key)
    return principals


def load_principals(home: Path | None = None) -> dict[str, Principal]:
    from internal_ai.secrets import SERVING_KEYS, operator_secret

    raw = operator_secret(home, SERVING_KEYS) if home is not None else ""
    if not raw:
        raw = os.environ.get("INTERNAL_AI_SERVING_KEYS", "")
    return parse_api_keys(raw)


@dataclass(slots=True)
class RetrievalHit:
    document_id: str
    score: float
    snippet: str = ""
    source_id: str = ""


@dataclass(slots=True)
class ModelRouter:
    """Active model version per model name, with rollback support."""

    store: Store
    audit: AuditLog
    active: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str) -> ModelVersion | None:
        candidates = self.store.list_models(name=name)
        if not candidates:
            return None
        model_id = self.active.get(name)
        if model_id:
            for candidate in candidates:
                if candidate.id == model_id:
                    return candidate
        promoted = [item for item in candidates if item.status == "promoted"]
        pool = promoted or candidates
        return pool[-1]

    def set_active(self, name: str, model_id: str, actor: str, reason: str) -> ModelVersion:
        model = self.store.get_model(model_id)
        if model is None:
            raise KeyError(f"unknown model: {model_id}")
        previous = self.active.get(name)
        self.active[name] = model_id
        self.audit.record(
            actor=actor,
            actor_kind="human",
            action="serving.version_activated",
            subject=name,
            details={"model_id": model_id, "previous_model_id": previous, "reason": reason},
        )
        return model

    def rollback(self, name: str, model_id: str, actor: str) -> ModelVersion:
        return self.set_active(name, model_id, actor, reason="rollback")


class RetrievalService:
    """Dense retrieval with document-level permission enforcement (FR-18)."""

    def __init__(self, store: Store, audit: AuditLog) -> None:
        self.store = store
        self.audit = audit

    def corpus(self, dataset_id: str) -> list[dict[str, Any]]:
        return self.store.dataset_records(dataset_id)

    def search(
        self,
        embedder: Any,
        dataset_id: str,
        query: str,
        principal: Principal,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        rows = self.corpus(dataset_id)
        if not rows:
            return []
        visible = [row for row in rows if principal.can_read(row["acl"], row["sensitivity"])]
        if not visible:
            return []
        vectors = embedder.embed([row["payload"] for row in visible])
        query_vector = embedder.embed([query])[0]
        scores = vectors @ query_vector
        order = np.argsort(-scores)[:top_k]
        hits: list[RetrievalHit] = []
        for index in order:
            row = visible[int(index)]
            hits.append(
                RetrievalHit(
                    document_id=row["record_key"],
                    score=round(float(scores[int(index)]), 6),
                    snippet=" ".join(row["payload"].split()[:24]),
                    source_id=row["source_id"],
                )
            )
        self.audit.record(
            actor=principal.user_id,
            actor_kind="human",
            action="serving.search",
            subject=dataset_id,
            outcome=AuditOutcome.SUCCESS,
            inputs={
                "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
                "query_length": len(query),
                "top_k": top_k,
            },
            details={
                "hits": len(hits),
                "visible_documents": len(visible),
                "total_documents": len(rows),
            },
        )
        return hits


def log_request(
    audit: AuditLog,
    principal: Principal,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    extra: dict[str, Any] | None = None,
) -> None:
    audit.record(
        actor=principal.user_id,
        actor_kind="human",
        action="serving.request",
        subject=endpoint,
        outcome=AuditOutcome.SUCCESS if status_code < 400 else AuditOutcome.FAILURE,
        details={
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            **(extra or {}),
        },
    )
