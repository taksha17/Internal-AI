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
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
        if sensitivity not in ("", "internal"):
            return False
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
        self._lock = threading.Lock()

    def allow(self, user_id: str, now: float | None = None) -> bool:
        moment = now if now is not None else time.monotonic()
        with self._lock:
            window = self._hits[user_id]
            while window and moment - window[0] > 60.0:
                window.popleft()
            if len(window) >= self.limit:
                return False
            window.append(moment)
            return True

    def remaining(self, user_id: str, now: float | None = None) -> int:
        moment = now if now is not None else time.monotonic()
        with self._lock:
            window = self._hits[user_id]
            while window and moment - window[0] > 60.0:
                window.popleft()
            return max(self.limit - len(window), 0)


def parse_api_keys(raw: str) -> dict[str, Principal]:
    """Parse serving keys from `key:user:group1|group2` entries or JSON.

    Entries are separated by commas, so the groups inside one entry use ``|``.
    """
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


def load_principals() -> dict[str, Principal]:
    return parse_api_keys(os.environ.get("INTERNAL_AI_SERVING_KEYS", ""))


@dataclass(slots=True)
class RetrievalHit:
    document_id: str
    score: float
    snippet: str = ""
    source_id: str = ""


LEGACY_PERMISSION_NOTE = "permission enforcement is verified by the retrieval service tests"


def out_of_service(store: Store, model: ModelVersion) -> str | None:
    """Why a model must not answer a request, or ``None`` when it may.

    A promoted model stops being servable the moment its newest report on a
    still-frozen eval set fails: a later failing re-run is evidence about this
    model now, and the gate that promoted it approved an older report. Reports
    that are missing, or that sit on a set that is not frozen, are not evidence
    either way and leave the model as it was.
    """
    reports = store.list_eval_reports(model.id)
    if not reports:
        return None
    newest = reports[-1]
    eval_set = store.get_eval_set(newest.eval_set_id)
    if eval_set is None or not eval_set.frozen:
        return None
    leak = newest.checks.get("permission_leak") or {}
    if leak.get("note") == LEGACY_PERMISSION_NOTE:
        return (
            f"{model.id}'s newest report still carries the legacy permission note "
            "and is not evidence"
        )
    if newest.passed:
        return None
    recall = newest.metrics.get("recall_at_10")
    baseline = newest.baseline_metrics.get("recall_at_10")
    detail = ""
    if recall is not None and baseline is not None:
        detail = f" at recall@10 {recall:.3f} against a keyword baseline of {baseline:.3f}"
    return (
        f"{model.id}'s newest report on {newest.eval_set_id}{detail} failed; "
        "it is out of service until a later evaluation passes"
    )


@dataclass(slots=True)
class ModelRouter:
    """Active model version per model name, with rollback support."""

    store: Store
    audit: AuditLog
    active: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str) -> ModelVersion | None:
        all_valid = [
            item
            for item in self.store.list_models(name=name)
            if out_of_service(self.store, item) is None
        ]
        promoted = [item for item in all_valid if item.status == "promoted"]
        if not promoted:
            return None
        model_id = self.active.get(name)
        if model_id:
            for candidate in promoted:
                if candidate.id == model_id:
                    return candidate
        return promoted[-1]

    def set_active(self, name: str, model_id: str, actor: str, reason: str) -> ModelVersion:
        model = self.store.get_model(model_id)
        if model is None:
            raise KeyError(f"unknown model: {model_id}")
        if model.status != "promoted":
            raise PermissionError(
                f"only a promoted model may serve requests; {model_id} is {model.status}"
            )
        refused = out_of_service(self.store, model)
        if refused is not None:
            raise PermissionError(f"cannot serve a model that failed its evaluation: {refused}")
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


def served_model(store: Store, router: ModelRouter) -> ModelVersion | None:
    """The model a request uses when the caller does not name one.

    The sample path is named embedding-v1. A deployment of a promoted model
    under another name is what gets served when that sample name is absent.
    """
    named = router.resolve("embedding-v1")
    if named is not None:
        return named
    promoted = [
        item
        for item in store.list_models()
        if item.status == "promoted" and out_of_service(store, item) is None
    ]
    return promoted[-1] if promoted else None


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


_HIDDEN_DETAIL_KEYS = frozenset({"question", "query", "prompt"})


def public_details(extra: dict[str, Any] | None, question: str = "") -> dict[str, Any]:
    """Details safe to store. The question is not one of them."""
    hidden = {question} if question else set()
    cleaned: dict[str, Any] = {}
    for key, value in (extra or {}).items():
        if key in _HIDDEN_DETAIL_KEYS:
            continue
        if isinstance(value, str) and any(secret and secret in value for secret in hidden):
            continue
        cleaned[key] = value
    return cleaned


def log_request(
    audit: AuditLog,
    principal: Principal,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    extra: dict[str, Any] | None = None,
    question: str = "",
    cost_usd: float = 0.0,
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
            **public_details(extra, question),
        },
        cost_usd=cost_usd,
    )
