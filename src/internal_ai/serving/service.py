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
from internal_ai.data.records import version_admits
from internal_ai.data.shards import released_rows
from internal_ai.domain import AuditOutcome, ModelVersion
from internal_ai.evals.metrics import bm25_scores
from internal_ai.services import attributes
from internal_ai.services.attributes import DEFAULT_CLASS, ReadPolicy
from internal_ai.services.tenancy import DEFAULT_TENANT, normalise, same_tenant
from internal_ai.services.tenancy import check as check_tenant
from internal_ai.storage import BlobStore
from internal_ai.store import Store
from internal_ai.training.alpha.citations import hit_from_key

# Named by the attribute rule, so a key file and the console cannot disagree about
# which group is the one that skips a record's ACL.
SUPERUSER_GROUP = attributes.SUPERUSER_GROUP


@dataclass(slots=True)
class Principal:
    user_id: str
    groups: frozenset[str]
    tenant: str = DEFAULT_TENANT
    api_key: str = ""
    # None means "nobody has told this key what the install decided". The serving
    # process fills it from policy P7 on every request; a principal constructed
    # with a rule of its own keeps it, which is what tests and the console do.
    read_policy: ReadPolicy | None = None

    @property
    def grants(self) -> ReadPolicy:
        """The rule this principal reads by: the strict one unless it was given."""
        return self.read_policy or ReadPolicy.strict()

    def can_read(
        self,
        acl: list[str],
        sensitivity: str = DEFAULT_CLASS,
        tenant: str = DEFAULT_TENANT,
    ) -> bool:
        """Whether this principal may read a record, by the one attribute rule.

        Tenancy is asked first: a record of another tenant is out of reach
        whatever class it carries and whatever group the caller is in, because
        the tenant is a boundary and not a grant.
        """
        if not same_tenant(self.tenant, tenant):
            return False
        return self.grants.allows(self.groups, sensitivity, acl)

    def can_read_row(self, row: dict[str, Any]) -> bool:
        """The same answer for a row read from the table or from a shard."""
        return self.can_read(
            row.get("acl") or [],
            row.get("sensitivity") or DEFAULT_CLASS,
            row.get("tenant") or DEFAULT_TENANT,
        )


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


class TenantQuota:
    """Requests per minute one tenant may make, when its operator sets a number.

    The per-user window answers "is this caller asking too often". This one
    answers "is this customer's share of the deployment being spent", which is
    the quota an agreement names. A tenant with no entry is not capped here.
    """

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        self.limits = {normalise(name): int(value) for name, value in (limits or {}).items()}
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def limit(self, tenant: str) -> int | None:
        return self.limits.get(normalise(tenant))

    def _window(self, tenant: str, now: float | None) -> tuple[deque[float], float]:
        moment = now if now is not None else time.monotonic()
        window = self._hits[normalise(tenant)]
        while window and moment - window[0] > 60.0:
            window.popleft()
        return window, moment

    def allow(self, tenant: str, now: float | None = None) -> bool:
        allowance = self.limit(tenant)
        if allowance is None:
            return True
        with self._lock:
            window, moment = self._window(tenant, now)
            if len(window) >= allowance:
                return False
            window.append(moment)
            return True

    def remaining(self, tenant: str, now: float | None = None) -> int | None:
        allowance = self.limit(tenant)
        if allowance is None:
            return None
        with self._lock:
            window, _ = self._window(tenant, now)
            return max(allowance - len(window), 0)


class TenantQuotaError(ValueError):
    """A quota list that cannot be read."""


def parse_tenant_quotas(raw: str) -> dict[str, int]:
    """``acme:30,globex:10``. A malformed entry refuses rather than uncapping."""
    limits: dict[str, int] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, value = entry.partition(":")
        value = value.strip()
        if not separator or not value.isdigit():
            raise TenantQuotaError(f"{entry!r} is not written tenant:requests-per-minute")
        tenant = check_tenant(name)
        if int(value) <= 0:
            raise TenantQuotaError(f"{tenant} names {value} requests a minute; the floor is 1")
        limits[tenant] = int(value)
    return limits


def load_tenant_quotas() -> dict[str, int]:
    return parse_tenant_quotas(os.environ.get("INTERNAL_AI_TENANT_QUOTAS", ""))


def parse_api_keys(raw: str) -> dict[str, Principal]:
    """Parse serving keys from `key:user:group1|group2:tenant` entries or JSON.

    Entries are separated by commas, so the groups inside one entry use ``|``.
    The tenant is the fourth field and is ``default`` when a key file leaves it
    out, which is the install's own tenant.
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
                tenant=normalise(str(item.get("tenant", ""))),
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
        tenant = normalise(parts[3]) if len(parts) > 3 and parts[3].strip() else DEFAULT_TENANT
        principals[key] = Principal(
            user_id=user_id, groups=frozenset(groups), tenant=tenant, api_key=key
        )
    return principals


def load_principals() -> dict[str, Principal]:
    return parse_api_keys(os.environ.get("INTERNAL_AI_SERVING_KEYS", ""))


@dataclass(slots=True)
class RetrievalHit:
    document_id: str
    score: float
    snippet: str = ""
    source_id: str = ""
    path: str = ""
    line: int = 0


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
    """Dense retrieval with document-level permission enforcement (FR-18).

    Document vectors are computed once per released dataset version and model,
    not once per query. The cache holds vectors for every record, and the
    caller's ACLs are applied to the ranking, so a cached matrix can never
    widen what a principal is allowed to read.
    """

    def __init__(
        self,
        store: Store,
        audit: AuditLog,
        blobs: BlobStore | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.blobs = blobs
        self._cache: dict[str, tuple[str, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    def corpus(self, dataset_id: str) -> list[dict[str, Any]]:
        """The released rows, read from the shard that names them where one exists.

        Serving from the table would answer from whatever the rows say today;
        the shard is what was released, so that is what a reader is given.
        """
        return released_rows(self.store, self.blobs, dataset_id)

    def _embedder_identity(self, embedder: Any) -> str:
        """What the cached vectors belong to. A different model must not hit them."""
        return "|".join(
            (
                type(embedder).__name__,
                str(getattr(embedder, "name", "")),
                str(getattr(embedder, "max_seq_len", "")),
            )
        )

    def _corpus_vectors(
        self, embedder: Any, dataset_id: str, rows: list[dict[str, Any]]
    ) -> np.ndarray:
        dataset = self.store.get_dataset(dataset_id)
        content_hash = dataset.content_hash if dataset is not None else ""
        if not content_hash:
            return embedder.embed([row["payload"] for row in rows])
        key = f"{self._embedder_identity(embedder)}:{content_hash}:{len(rows)}"
        cached = self._cache.get(dataset_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        vectors = embedder.embed([row["payload"] for row in rows])
        with self._cache_lock:
            self._cache[dataset_id] = (key, vectors)
        return vectors

    def _version_acl(self, dataset_id: str) -> list[str]:
        dataset = self.store.get_dataset(dataset_id)
        return list(dataset.acl) if dataset is not None else []

    def visible_indexes(
        self,
        rows: list[dict[str, Any]],
        principal: Principal,
        version_acl: list[str] | None = None,
    ) -> list[int]:
        """The rows this principal may read, by every filter at once.

        Another tenant's rows never reach the cached matrix, a class nobody
        granted is never ranked, the version's own groups are a ceiling a later
        edit of a row cannot raise, and the record's own ACL is the last gate.
        Both retrieval paths call this one function, so a filter cannot be added
        to the dense path and forgotten in the keyword one.
        """
        if not version_admits(principal.groups, version_acl or []):
            return []
        return [index for index, row in enumerate(rows) if principal.can_read_row(row)]

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
        visible = self.visible_indexes(rows, principal, self._version_acl(dataset_id))
        hits: list[RetrievalHit] = []
        # A search nobody may run is still a search: it is recorded with no hits
        # and its tenant, so a reviewer can tell an empty corpus from a caller
        # asking about another tenant's.
        if visible:
            vectors = self._corpus_vectors(embedder, dataset_id, rows)
            query_vector = embedder.embed([query])[0]
            scores = vectors[visible] @ query_vector
            order = np.argsort(-scores)[:top_k]
            for index in order:
                row = rows[visible[int(index)]]
                cited = hit_from_key(row["record_key"])
                hits.append(
                    RetrievalHit(
                        document_id=row["record_key"],
                        score=round(float(scores[int(index)]), 6),
                        snippet=" ".join(row["payload"].split()[:24]),
                        source_id=row["source_id"],
                        path=cited.path,
                        line=cited.line,
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
                "tenant": normalise(principal.tenant),
            },
            tenant=normalise(principal.tenant),
        )
        return hits

    def keyword_search(
        self,
        dataset_id: str,
        query: str,
        principal: Principal,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """BM25 over what the caller may read, for a deployment with no embedder.

        A fine-tuned generator has no embedder, so retrieval cannot borrow its
        weights. Keyword search needs none, applies the same permission filter as
        the dense path, and returns only documents that share a term with the
        query: a zero score is not a match.
        """
        rows = self.corpus(dataset_id)
        if not rows:
            return []
        visible = self.visible_indexes(rows, principal, self._version_acl(dataset_id))
        hits: list[RetrievalHit] = []
        if visible:
            scores = bm25_scores(query, [rows[index]["payload"] for index in visible])
            ranked = sorted(range(len(visible)), key=lambda i: -scores[i])[:top_k]
            for position in ranked:
                if scores[position] <= 0:
                    continue
                row = rows[visible[position]]
                cited = hit_from_key(row["record_key"])
                hits.append(
                    RetrievalHit(
                        document_id=row["record_key"],
                        score=round(float(scores[position]), 6),
                        snippet=" ".join(row["payload"].split()[:24]),
                        source_id=row["source_id"],
                        path=cited.path,
                        line=cited.line,
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
                "retriever": "bm25",
            },
            details={
                "hits": len(hits),
                "visible_documents": len(visible),
                "total_documents": len(rows),
                "tenant": normalise(principal.tenant),
            },
            tenant=normalise(principal.tenant),
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
        tenant=normalise(principal.tenant),
        cost_usd=cost_usd,
    )
