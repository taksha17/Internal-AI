"""What a request asked, answered and got wrong, kept without keeping secrets.

Audit events record a hash of the question, which is the right default and also
why "turn production failures into improvements" could not work: after the
request ends there is nothing left to turn. A trace keeps the question, the
answer and the citations so a failed one can become an eval case, but it is
scrubbed before it is stored and tagged with the groups that may read it, so the
privacy stance moves from "we store nothing" to "we store nothing raw".

Two rules shape the code:

- Scrubbing happens before the insert, never on read. A row that exists is a row
  that was already safe; a reader cannot forget to scrub.
- A trace inherits the groups of the principal that caused it. Someone outside
  those groups cannot read it even though the store is shared, which is the same
  rule documents follow.

Redaction errs towards removing too much: a 13-to-19 digit run is treated as a
card number whether or not it passes Luhn, because a wrongly redacted order
number costs a triage note and a wrongly kept card number costs an incident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.data.transforms import PII_PATTERNS, REDACTION_TEMPLATE
from internal_ai.ids import new_id, now_iso
from internal_ai.serving.service import Principal
from internal_ai.store import Store

RETENTION_DAYS = 30
FAILURE_CITATION = "citation_mismatch"
FAILURE_GUARDRAIL = "guardrail_block"
FAILURE_EMPTY = "no_evidence"
FAILURES: tuple[str, ...] = (FAILURE_CITATION, FAILURE_GUARDRAIL, FAILURE_EMPTY)


@dataclass(slots=True)
class Trace:
    id: str
    created_at: str
    user_id: str
    groups: list[str]
    endpoint: str
    model_id: str
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    acl: list[str] = field(default_factory=list)
    sensitivity: str = "internal"
    latency_ms: float = 0.0
    status_code: int = 200
    failure: str = ""
    redactions: dict[str, int] = field(default_factory=dict)
    expires_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "user_id": self.user_id,
            "groups": list(self.groups),
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "retrieved": list(self.retrieved),
            "acl": list(self.acl),
            "sensitivity": self.sensitivity,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "failure": self.failure,
            "redactions": dict(self.redactions),
            "expires_at": self.expires_at,
        }


def scrub(text: str) -> tuple[str, dict[str, int]]:
    """Remove detector matches, returning the counts so a reader knows it happened."""
    counts: dict[str, int] = {}
    cleaned = text or ""
    for kind, pattern in PII_PATTERNS.items():
        found = len(pattern.findall(cleaned))
        if not found:
            continue
        counts[kind] = found
        cleaned = pattern.sub(REDACTION_TEMPLATE.format(kind=kind), cleaned)
    return cleaned, counts


def _expiry(now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    return (moment + timedelta(days=RETENTION_DAYS)).isoformat()


def record_trace(
    store: Store,
    audit: AuditLog,
    principal: Principal,
    *,
    endpoint: str,
    model_id: str = "",
    question: str = "",
    answer: str = "",
    citations: list[dict[str, Any]] | None = None,
    retrieved: list[str] | None = None,
    latency_ms: float = 0.0,
    status_code: int = 200,
    failure: str = "",
) -> Trace:
    """Store one scrubbed, tagged request. The audit event carries no text."""
    clean_question, question_counts = scrub(question)
    clean_answer, answer_counts = scrub(answer)
    redactions = {
        kind: question_counts.get(kind, 0) + answer_counts.get(kind, 0)
        for kind in set(question_counts) | set(answer_counts)
    }
    trace = Trace(
        id=new_id("trc"),
        created_at=now_iso(),
        user_id=principal.user_id,
        groups=sorted(principal.groups),
        endpoint=endpoint,
        model_id=model_id,
        question=clean_question,
        answer=clean_answer,
        citations=list(citations or []),
        retrieved=list(retrieved or []),
        acl=sorted(principal.groups),
        sensitivity="internal",
        latency_ms=round(latency_ms, 3),
        status_code=status_code,
        failure=failure if failure in FAILURES else "",
        redactions=redactions,
        expires_at=_expiry(),
    )
    store.conn.execute(
        "INSERT INTO traces (id, created_at, user_id, groups, endpoint, model_id, question,"
        " answer, citations, retrieved, acl, sensitivity, latency_ms, status_code, failure,"
        " redactions, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            trace.id,
            trace.created_at,
            trace.user_id,
            json.dumps(trace.groups),
            trace.endpoint,
            trace.model_id,
            trace.question,
            trace.answer,
            json.dumps(trace.citations, sort_keys=True),
            json.dumps(trace.retrieved),
            json.dumps(trace.acl),
            trace.sensitivity,
            trace.latency_ms,
            trace.status_code,
            trace.failure,
            json.dumps(trace.redactions, sort_keys=True),
            trace.expires_at,
        ),
    )
    audit.record(
        actor=principal.user_id,
        actor_kind="human",
        action="serving.trace",
        subject=trace.id,
        details={
            "endpoint": endpoint,
            "status_code": status_code,
            "failure": trace.failure,
            "question_hash": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            "redactions": trace.redactions,
        },
    )
    return trace


def _row_to_trace(row: Any) -> Trace:
    return Trace(
        id=row["id"],
        created_at=row["created_at"],
        user_id=row["user_id"],
        groups=json.loads(row["groups"]),
        endpoint=row["endpoint"],
        model_id=row["model_id"],
        question=row["question"],
        answer=row["answer"],
        citations=json.loads(row["citations"]),
        retrieved=json.loads(row["retrieved"]),
        acl=json.loads(row["acl"]),
        sensitivity=row["sensitivity"],
        latency_ms=float(row["latency_ms"]),
        status_code=int(row["status_code"]),
        failure=row["failure"],
        redactions=json.loads(row["redactions"]),
        expires_at=row["expires_at"],
    )


def get_trace(store: Store, principal: Principal, trace_id: str) -> Trace:
    """One trace, if this principal may read it. Another group's trace is not a 404.

    It is refused as a permission error rather than hidden, because the caller
    typed an id that exists: pretending it does not would make a shared store
    look emptier than it is without making it safer.
    """
    row = store.conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown trace: {trace_id}")
    trace = _row_to_trace(row)
    if not principal.can_read(trace.acl, trace.sensitivity):
        raise PermissionError("this trace belongs to a group you are not in")
    return trace


def list_traces(
    store: Store,
    principal: Principal,
    *,
    limit: int = 50,
    failures_only: bool = False,
) -> list[Trace]:
    """Traces this principal may read, newest first."""
    query = "SELECT * FROM traces ORDER BY created_at DESC, id DESC LIMIT ?"
    rows = store.conn.execute(query, (max(limit * 4, limit),)).fetchall()
    readable = [
        _row_to_trace(row)
        for row in rows
        if principal.can_read(json.loads(row["acl"]), row["sensitivity"])
    ]
    if failures_only:
        readable = [trace for trace in readable if trace.failure]
    return readable[:limit]


def triage(store: Store, principal: Principal, *, limit: int = 50) -> dict[str, list[Trace]]:
    """The queue a person works: what failed, why, and what to turn it into."""
    readable = list_traces(store, principal, limit=limit * 4)
    buckets: dict[str, list[Trace]] = {name: [] for name in FAILURES}
    for trace in readable:
        if trace.failure in buckets:
            buckets[trace.failure].append(trace)
    return {name: items[:limit] for name, items in buckets.items()}


def purge_expired(store: Store, now: datetime | None = None) -> int:
    """Delete traces past retention. Returns how many rows went."""
    moment = (now or datetime.now(UTC)).isoformat()
    cursor = store.conn.execute("DELETE FROM traces WHERE expires_at <= ?", (moment,))
    return int(cursor.rowcount or 0)
