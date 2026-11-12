"""Fetch a URL through an approved web-access gate, as a person.

Agents fetch through ``ToolRegistry.web_fetch`` with a ``NET_EGRESS`` role.
This is the person's path to the same bytes: the same refusals (offline, no
approved gate, host off the allowlist), the same audit trail, and no role
capability to hold, because the gate's dual-control approval is the
authorization. The permission check only proves a signed-in person is asking.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlsplit

from internal_ai import tools
from internal_ai.config import offline_mode
from internal_ai.context import Platform
from internal_ai.domain import AuditOutcome, GateType
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require
from internal_ai.tools import ToolDenied, approved_web_allowlist


@dataclass(frozen=True, slots=True)
class WebFetchResult:
    gate_id: str
    url: str
    host: str
    size_bytes: int
    sha256: str
    body: bytes


def _host_or_url(url: str) -> str:
    return urlsplit(url).hostname or url


def fetch_url(platform: Platform, actor: Actor, gate_id: str, url: str) -> WebFetchResult:
    """Fetch ``url`` under an approved ``WEB_ACCESS`` gate and log the fetch.

    Refusals are audited as well as raised, so a denied fetch is reviewable
    next to an approved one. The log carries the host, the size and the hash,
    never the query string and never the body.
    """
    require(platform.audit, actor, Action.USE_WEB_GATE, gate_id)
    try:
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise ToolDenied("web fetch allows https URLs only")
        allowlist = approved_web_allowlist(platform.store, gate_id)
        host = (parts.hostname or "").lower()
        if host not in allowlist:
            raise ToolDenied("host is not on the approved allowlist")
        if offline_mode():
            raise ToolDenied("web fetch is disabled while the install is offline")
        body = tools.fetch_url_bytes(url)
    except ToolDenied as exc:
        platform.audit.record(
            actor=actor.id,
            actor_kind=actor.kind,
            action="service.web_fetch",
            subject=gate_id,
            outcome=AuditOutcome.DENIED,
            inputs={"gate_id": gate_id, "host": _host_or_url(url)},
            details={"reason": str(exc)},
        )
        raise
    result = WebFetchResult(
        gate_id=gate_id,
        url=url,
        host=host,
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        body=body,
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="service.web_fetch",
        subject=gate_id,
        outcome=AuditOutcome.SUCCESS,
        inputs={
            "gate_id": gate_id,
            "host": host,
            "size_bytes": len(body),
            "sha256": result.sha256,
        },
        details={"result_type": "bytes"},
    )
    return result


def approved_web_gates(platform: Platform) -> list:
    """Every approved web-access gate, oldest first, for the fetch page."""
    return [
        gate
        for gate in platform.store.list_gates(status="approved")
        if str(gate.gate_type) == str(GateType.WEB_ACCESS)
    ]
