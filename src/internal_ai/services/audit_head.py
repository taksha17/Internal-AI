"""Anchoring the audit head, from the console, the CLI or an agent.

One function so all three clients take the same path: the permission check, the
signature, the file and the audit event happen together or not at all.
"""

from __future__ import annotations

from typing import Any

from internal_ai.context import Platform
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, allowed, require


def anchor_status(platform: Platform) -> dict[str, Any]:
    """What the audit page shows: is a key configured, and where is the last anchor."""
    latest = platform.audit.newest_anchor()
    signer = platform.audit.signer
    return {
        "signing": signer is not None,
        "key_id": signer.key_id if signer is not None else "",
        "anchors": len(platform.audit.anchors()),
        "latest_seq": latest.seq if latest is not None else None,
        "latest_at": latest.signed_at if latest is not None else "",
        "directory": str(platform.audit.anchor_dir or ""),
    }


def anchor_head(platform: Platform, actor: Actor) -> dict[str, Any]:
    require(platform.audit, actor, Action.ANCHOR_AUDIT, "audit.head")
    record = platform.audit.anchor(actor=actor.id, actor_kind=actor.kind)
    return record.to_dict()


def may_anchor(actor: Actor) -> bool:
    return allowed(actor, Action.ANCHOR_AUDIT)
