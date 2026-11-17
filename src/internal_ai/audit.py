"""Immutable audit log with a verifiable hash chain (FR-20, FR-21).

The chain proves the rows it holds were not edited. An anchor proves the tail was
not deleted: the head is signed with a key kept outside the database, so a rollback
leaves an anchor pointing at an event that is no longer there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from internal_ai.domain import AuditEvent, AuditOutcome
from internal_ai.ids import content_hash, new_id, now_iso
from internal_ai.signing import ANCHOR_UNSIGNED, Anchor, AuditSigner, SigningRefused
from internal_ai.store import Store

CHECKED_AT = "audit_checked_at"
CHECKED_OK = "audit_checked_ok"


@dataclass(slots=True)
class AuditVerification:
    ok: bool
    checked: int
    broken_at_seq: int | None = None
    reason: str = ""
    anchored: bool = False
    anchor_seq: int | None = None
    anchor_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "broken_at_seq": self.broken_at_seq,
            "reason": self.reason,
            "anchored": self.anchored,
            "anchor_seq": self.anchor_seq,
            "anchor_note": self.anchor_note,
        }


def compute_hash(event: AuditEvent) -> str:
    return content_hash(
        {
            "id": event.id,
            "actor": event.actor,
            "actor_kind": event.actor_kind,
            "action": event.action,
            "subject": event.subject,
            "outcome": str(event.outcome),
            "inputs": event.inputs,
            "details": event.details,
            "cost_usd": event.cost_usd,
            "prev_hash": event.prev_hash,
            "created_at": event.created_at,
            "tenant": event.tenant,
        }
    )


class AuditLog:
    """Append-only event log.

    Every side-effecting action (job submission, dataset write, deployment,
    budget change) must be recorded here with agent identity, inputs and outcome.
    """

    def __init__(
        self,
        store: Store,
        signer: AuditSigner | None = None,
        anchor_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.signer = signer
        self.anchor_dir = anchor_dir
        self._health: AuditVerification | None = None

    def health(self) -> AuditVerification:
        """The last verification, without walking the chain again.

        A page render cannot afford a full scan, and a result from a moment ago is
        what a banner needs: this is an integrity signal, not a live audit.
        """
        if self._health is None:
            self._health = self.verify()
        return self._health

    def anchors(self) -> list[Anchor]:
        """Every anchor on disk, oldest first -- the files are the evidence."""
        if self.anchor_dir is None or not self.anchor_dir.exists():
            return []
        found: list[Anchor] = []
        for path in sorted(self.anchor_dir.glob("*.json")):
            try:
                found.append(Anchor.from_dict(json.loads(path.read_text())))
            except (ValueError, KeyError):
                continue
        return found

    def newest_anchor(self) -> Anchor | None:
        anchors = self.anchors()
        return anchors[-1] if anchors else None

    def anchor(self, actor: str, actor_kind: str = "human") -> Anchor:
        """Sign the current head so a later truncation contradicts a file.

        The event that records the anchoring is written after it, so it is the
        next head rather than part of what was signed -- which is what makes the
        anchor checkable against a log that has since grown.
        """
        if self.signer is None:
            raise SigningRefused(ANCHOR_UNSIGNED)
        chain = self.store.audit_chain()
        if not chain:
            raise SigningRefused("there is nothing to anchor: the audit log is empty")
        head = chain[-1]
        record = self.signer.anchor(
            seq=head.seq, hash_=head.hash, count=len(chain), signed_at=now_iso()
        )
        if self.anchor_dir is not None:
            self.anchor_dir.mkdir(parents=True, exist_ok=True)
            name = f"{record.seq:012d}-{record.hash[:12]}.json"
            (self.anchor_dir / name).write_text(json.dumps(record.to_dict(), indent=2))
        self.record(
            actor=actor,
            actor_kind=actor_kind,
            action="audit.anchored",
            subject=str(record.seq),
            details={
                "seq": record.seq,
                "hash": record.hash,
                "count": record.count,
                "key_id": record.key_id,
                "anchor_file": self.anchor_dir is not None,
            },
        )
        return record

    def record(
        self,
        actor: str,
        action: str,
        subject: str,
        outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
        actor_kind: str = "agent",
        inputs: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        tenant: str = "default",
    ) -> AuditEvent:
        event = AuditEvent(
            id=new_id("aud"),
            seq=0,
            actor=actor,
            actor_kind=actor_kind,
            action=action,
            subject=subject,
            outcome=outcome if isinstance(outcome, AuditOutcome) else AuditOutcome(outcome),
            inputs=inputs or {},
            details=details or {},
            cost_usd=cost_usd,
            tenant=tenant,
        )
        return self.store.append_audit(event)

    def check_due(self, interval_hours: float = 24.0) -> bool:
        """Whether enough time has passed since the last full check to run another.

        There is no daemon here, so the schedule is whatever opens the platform: a
        cron entry runs `audit check`, and this decides whether that run walks the
        chain or says it is too soon. An unreadable stamp is due, because a stamp
        nobody can parse is not evidence that a check happened.
        """
        if interval_hours <= 0:
            return True
        last = self.store.kv_get(CHECKED_AT, "")
        if not last:
            return True
        try:
            then = datetime.fromisoformat(last)
        except ValueError:
            return True
        return datetime.now(UTC) - then >= timedelta(hours=interval_hours)

    def verify(self) -> AuditVerification:
        previous_hash = ""
        checked = 0
        chain = self.store.audit_chain()
        for event in chain:
            checked += 1
            if event.prev_hash != previous_hash:
                return AuditVerification(
                    False,
                    checked,
                    event.seq,
                    f"broken link: prev_hash {event.prev_hash[:12]} != {previous_hash[:12]}",
                )
            if compute_hash(event) != event.hash:
                return AuditVerification(
                    False, checked, event.seq, "hash mismatch: event was modified"
                )
            previous_hash = event.hash
        result = self._check_anchor({event.seq: event for event in chain}, checked)
        self._health = result
        self.store.kv_set(CHECKED_AT, now_iso())
        self.store.kv_set(CHECKED_OK, "1" if result.ok else "0")
        return result

    def _check_anchor(self, by_seq: dict[int, AuditEvent], checked: int) -> AuditVerification:
        """Compare the newest anchor with the chain, and verify its signature if we can.

        No key configured is not a failure: the chain still links and the anchor
        still catches a deleted tail, because the anchor names an event by number
        and hash. What cannot be done without the key is detect a forged anchor.
        """
        latest = self.newest_anchor()
        if latest is None:
            return AuditVerification(
                True,
                checked,
                anchor_note=(
                    "no anchor yet: the chain links event to event, but a deleted tail "
                    "would still look like a shorter chain"
                ),
            )
        event = by_seq.get(latest.seq)
        if event is None or event.hash != latest.hash:
            return AuditVerification(
                False,
                checked,
                latest.seq,
                f"the anchor at #{latest.seq} names an event the log no longer holds "
                "(truncated or rewritten)",
            )
        if self.signer is not None and self.signer.key_id == latest.key_id:
            message = json.dumps(latest.payload(), sort_keys=True, separators=(",", ":"))
            if not self.signer.verify(message.encode("utf-8"), latest.signature):
                return AuditVerification(
                    False,
                    checked,
                    latest.seq,
                    "the anchor signature does not match its contents",
                )
            return AuditVerification(
                True,
                checked,
                anchored=True,
                anchor_seq=latest.seq,
                anchor_note=f"head #{latest.seq} is anchored and signed with key {latest.key_id}",
            )
        return AuditVerification(
            True,
            checked,
            anchored=True,
            anchor_seq=latest.seq,
            anchor_note=(
                f"head #{latest.seq} matches its anchor, but key {latest.key_id} is not "
                "configured here, so the signature itself is unchecked"
            ),
        )

    def recent(self, limit: int = 50, **filters: Any) -> list[AuditEvent]:
        return self.store.list_audit(limit=limit, **filters)
