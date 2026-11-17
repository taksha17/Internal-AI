"""Anchors that make a truncated chain visible.

A hash chain proves the rows it still holds were not edited. It does not prove
there are no more rows: delete the tail and the remainder still links perfectly.
An anchor is the head (sequence, hash, count) signed with a key the database does
not hold, so a rollback leaves an anchor pointing at an event that is no longer
there -- which is the only way to notice.

The key lives outside the database on purpose. An anchor signed with a key stored
in the same rows is a checksum, not proof.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_AUDIT_KEY = "INTERNAL_AI_AUDIT_KEY_FILE"

ANCHOR_UNSIGNED = (
    "no audit signing key is configured, so a head cannot be anchored: set "
    f"{ENV_AUDIT_KEY} to a key file outside the database, or run "
    "`internal-ai audit key` once. Without it the chain still links event to "
    "event, but a deleted tail would look like a shorter chain."
)


class SigningRefused(RuntimeError):
    """An anchor was refused rather than written with a key it cannot trust."""


def canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class Anchor:
    seq: int
    hash: str
    count: int
    signed_at: str
    signature: str
    key_id: str

    def payload(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "hash": self.hash,
            "count": self.count,
            "signed_at": self.signed_at,
            "key_id": self.key_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Anchor:
        return cls(
            seq=int(data["seq"]),
            hash=str(data["hash"]),
            count=int(data["count"]),
            signed_at=str(data["signed_at"]),
            signature=str(data["signature"]),
            key_id=str(data["key_id"]),
        )


class AuditSigner:
    """HMAC-SHA256 over the canonical head record.

    Shaped like a KMS call -- ``sign(bytes) -> bytes`` with a key id -- so an
    install that has Vault or a cloud KMS replaces the key handling and keeps the
    call site, which is the whole point of not putting a hash function in
    ``audit.py``.
    """

    def __init__(self, key: bytes, key_id: str) -> None:
        if len(key) < 32:
            raise SigningRefused("an audit signing key must be at least 32 bytes")
        self._key = key
        self.key_id = key_id

    def sign(self, message: bytes) -> str:
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def verify(self, message: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(message), signature)

    def anchor(self, seq: int, hash_: str, count: int, signed_at: str) -> Anchor:
        record = {
            "seq": seq,
            "hash": hash_,
            "count": count,
            "signed_at": signed_at,
            "key_id": self.key_id,
        }
        signature = self.sign(canonical(record).encode("utf-8"))
        return Anchor(
            seq=seq,
            hash=hash_,
            count=count,
            signed_at=signed_at,
            signature=signature,
            key_id=self.key_id,
        )


def key_path(configured: str | None = None) -> Path | None:
    value = (configured or os.environ.get(ENV_AUDIT_KEY, "")).strip()
    return Path(value).expanduser() if value else None


def create_key(path: Path) -> AuditSigner:
    """Write a new key with owner-only permissions, never overwriting one."""
    if path.exists():
        raise SigningRefused(f"{path} already exists; refusing to replace a signing key")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(48)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(key)
    return AuditSigner(key, _key_id(key))


def load_signer(path: Path) -> AuditSigner:
    if not path.exists():
        raise SigningRefused(
            f"{path} does not exist; run `internal-ai audit key` to create one before "
            "anchoring, because a key that is recreated later cannot verify older anchors"
        )
    # Not ``.strip()``: a random key can contain a space or a newline byte, and
    # stripping it turns the key into a different key -- the anchors then verify
    # against nothing and the failure looks like tampering.
    key = path.read_bytes()
    if key.endswith(b"\n"):
        key = key[:-1]
    return AuditSigner(key, _key_id(key))


def signer_from_env(configured: str | None = None) -> AuditSigner | None:
    """The signer this install configured, or nothing -- never a default key."""
    path = key_path(configured)
    if path is None:
        return None
    return load_signer(path)


def signer_for(home: Path) -> AuditSigner | None:
    """The key this install has, if it has one at all.

    ``INTERNAL_AI_AUDIT_KEY_FILE`` wins and must exist when it is set; otherwise a
    key at ``home/audit.key`` is used when present. An install with no key is not
    broken -- it just cannot anchor, and says so instead of inventing a key.
    """
    configured = key_path()
    if configured is not None:
        return load_signer(configured)
    local = home / "audit.key"
    return load_signer(local) if local.exists() else None


def _key_id(key: bytes) -> str:
    """A public name for the key so an anchor says which key signed it."""
    return hashlib.sha256(key).hexdigest()[:16]
