"""Accounts, sessions and personal API tokens.

A decision is only as trustworthy as the identity behind it, so the actor on
every governance write comes from here: a session cookie in the console, a
personal token on the API, or a password or token on the command line.
Secrets are never stored, only their hashes: scrypt for passwords, SHA-256
for random session and API tokens.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import AuditOutcome
from internal_ai.ids import new_id, now_iso
from internal_ai.people import PeopleRegistry
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require

MIN_PASSWORD_LENGTH = 12
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60
SESSION_IDLE_SECONDS = 30 * 60
SESSION_ABSOLUTE_SECONDS = 8 * 60 * 60
TOKEN_LIFETIME_DAYS = 90
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1


class AuthenticationError(PermissionError):
    pass


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = encoded.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt_hex),
        n=int(n),
        r=int(r),
        p=int(p),
        dklen=len(digest_hex) // 2,
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def check_password_strength(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"a password needs at least {MIN_PASSWORD_LENGTH} characters")
    if password.strip() != password:
        raise ValueError("a password cannot start or end with whitespace")


@dataclass(slots=True)
class Session:
    token: str
    csrf_token: str
    person_id: str
    must_change_password: bool


class IdentityService:
    def __init__(self, platform: Platform, clock: Callable[[], float] = time.time) -> None:
        self.platform = platform
        self.store = platform.store
        self.audit = platform.audit
        self.people = PeopleRegistry(platform.store)
        self.clock = clock

    # ----------------------------------------------------------- accounts

    def has_accounts(self) -> bool:
        return self.store.conn.execute("SELECT 1 FROM accounts LIMIT 1").fetchone() is not None

    def person(self, identifier: str) -> dict[str, Any] | None:
        return self.people.get(identifier)

    def account(self, person_id: str) -> dict[str, Any] | None:
        row = self.store.conn.execute(
            "SELECT * FROM accounts WHERE person_id = ?", (person_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_accounts(self) -> list[dict[str, Any]]:
        rows = self.store.conn.execute(
            "SELECT person_id, must_change_password, failed_attempts, locked_until, disabled,"
            " created_by, created_at, rotated_at FROM accounts ORDER BY person_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def create_account(self, person_id: str, created_by: Actor | None) -> str:
        """Create an account and return its one-time password.

        Only the very first account may be created without an actor, from the
        command line; after that an administrator creates accounts.
        """
        if created_by is None:
            if self.has_accounts():
                raise AuthenticationError("accounts exist; an administrator must create this one")
        else:
            require(self.audit, created_by, Action.MANAGE_PEOPLE, person_id)
        person = self.person(person_id)
        if person is None:
            raise KeyError(f"unknown person: {person_id}")
        if person["placeholder"]:
            raise ValueError(
                f"{person['name']} is a placeholder identity; replace them with a real person "
                "before creating an account"
            )
        if self.account(person["id"]) is not None:
            raise ValueError(f"{person['name']} already has an account")
        one_time = secrets.token_urlsafe(12)
        stamp = now_iso()
        self.store.conn.execute(
            "INSERT INTO accounts (person_id, password_hash, must_change_password, created_by,"
            " created_at, rotated_at) VALUES (?,?,1,?,?,?)",
            (
                person["id"],
                hash_password(one_time),
                created_by.id if created_by else "bootstrap",
                stamp,
                stamp,
            ),
        )
        self.audit.record(
            actor=created_by.id if created_by else "bootstrap",
            actor_kind="human",
            action="account.created",
            subject=person["id"],
            details={"bootstrap": created_by is None},
        )
        return one_time

    def reset_password(self, person_id: str, by: Actor) -> str:
        require(self.audit, by, Action.MANAGE_PEOPLE, person_id)
        if self.account(person_id) is None:
            raise KeyError(f"{person_id} has no account")
        one_time = secrets.token_urlsafe(12)
        self.store.conn.execute(
            "UPDATE accounts SET password_hash = ?, must_change_password = 1, failed_attempts = 0,"
            " locked_until = 0, rotated_at = ? WHERE person_id = ?",
            (hash_password(one_time), now_iso(), person_id),
        )
        self.end_all_sessions(person_id)
        self.audit.record(
            actor=by.id, actor_kind="human", action="account.password_reset", subject=person_id
        )
        return one_time

    def set_disabled(self, person_id: str, disabled: bool, by: Actor) -> None:
        require(self.audit, by, Action.MANAGE_PEOPLE, person_id)
        if by.id == person_id:
            raise ValueError("you cannot disable your own account")
        self.store.conn.execute(
            "UPDATE accounts SET disabled = ? WHERE person_id = ?", (int(disabled), person_id)
        )
        if disabled:
            self.end_all_sessions(person_id)
            self.store.conn.execute(
                "UPDATE api_tokens SET revoked = 1 WHERE person_id = ?", (person_id,)
            )
        self.audit.record(
            actor=by.id,
            actor_kind="human",
            action="account.disabled" if disabled else "account.enabled",
            subject=person_id,
        )

    def change_password(self, person_id: str, current: str, new: str) -> None:
        account = self.account(person_id)
        if account is None or not verify_password(current, account["password_hash"]):
            self.audit.record(
                actor=person_id,
                actor_kind="human",
                action="account.password_change",
                subject=person_id,
                outcome=AuditOutcome.DENIED,
                details={"reason": "current password did not match"},
            )
            raise AuthenticationError("the current password is not correct")
        check_password_strength(new)
        if new == current:
            raise ValueError("the new password must differ from the current one")
        self.store.conn.execute(
            "UPDATE accounts SET password_hash = ?, must_change_password = 0, rotated_at = ?"
            " WHERE person_id = ?",
            (hash_password(new), now_iso(), person_id),
        )
        self.audit.record(
            actor=person_id,
            actor_kind="human",
            action="account.password_changed",
            subject=person_id,
        )

    # ----------------------------------------------------- authentication

    def authenticate(self, identifier: str, password: str, client: str = "") -> Actor:
        """Check a password, applying the lockout rule, and return the actor."""
        person = self.person(identifier.strip())
        now = self.clock()
        account = self.account(person["id"]) if person else None
        if person is None or account is None:
            self._failed(identifier, client, "unknown account")
            raise AuthenticationError("unknown account or wrong password")
        if account["disabled"]:
            self._failed(person["id"], client, "account disabled")
            raise AuthenticationError("this account is disabled")
        if person["placeholder"]:
            self._failed(person["id"], client, "placeholder identity")
            raise AuthenticationError("placeholder identities cannot sign in")
        if account["locked_until"] > now:
            self._failed(person["id"], client, "account locked")
            minutes = int((account["locked_until"] - now) // 60) + 1
            raise AuthenticationError(f"too many failed attempts; try again in {minutes} minutes")
        if not verify_password(password, account["password_hash"]):
            attempts = account["failed_attempts"] + 1
            locked_until = now + LOCKOUT_SECONDS if attempts >= MAX_FAILED_ATTEMPTS else 0
            self.store.conn.execute(
                "UPDATE accounts SET failed_attempts = ?, locked_until = ? WHERE person_id = ?",
                (0 if locked_until else attempts, locked_until, person["id"]),
            )
            self._failed(
                person["id"], client, "account locked" if locked_until else "wrong password"
            )
            raise AuthenticationError("unknown account or wrong password")
        self.store.conn.execute(
            "UPDATE accounts SET failed_attempts = 0, locked_until = 0 WHERE person_id = ?",
            (person["id"],),
        )
        self.audit.record(
            actor=person["id"],
            actor_kind="human",
            action="account.signed_in",
            subject=person["id"],
            details={"client": client},
        )
        return Actor.from_person(person, authenticated=True)

    def _failed(self, identifier: str, client: str, why: str) -> None:
        self.audit.record(
            actor=identifier or "anonymous",
            actor_kind="human",
            action="account.sign_in_failed",
            subject=identifier or "anonymous",
            outcome=AuditOutcome.DENIED,
            details={"reason": why, "client": client},
        )

    def actor_for(self, person_id: str) -> Actor | None:
        person = self.person(person_id)
        account = self.account(person_id) if person else None
        if person is None or account is None or account["disabled"] or person["placeholder"]:
            return None
        return Actor.from_person(person, authenticated=True)

    # ------------------------------------------------------------ sessions

    def start_session(self, actor: Actor) -> Session:
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        now = self.clock()
        self.store.conn.execute(
            "INSERT INTO sessions (token_hash, person_id, csrf_token, created_at, last_seen,"
            " expires_at) VALUES (?,?,?,?,?,?)",
            (token_hash(token), actor.id, csrf, now, now, now + SESSION_ABSOLUTE_SECONDS),
        )
        account = self.account(actor.id) or {}
        return Session(
            token=token,
            csrf_token=csrf,
            person_id=actor.id,
            must_change_password=bool(account.get("must_change_password")),
        )

    def resume_session(self, token: str) -> tuple[Actor, Session] | None:
        if not token:
            return None
        digest = token_hash(token)
        row = self.store.conn.execute(
            "SELECT * FROM sessions WHERE token_hash = ?", (digest,)
        ).fetchone()
        if row is None:
            return None
        now = self.clock()
        if now > row["expires_at"] or now - row["last_seen"] > SESSION_IDLE_SECONDS:
            self.store.conn.execute("DELETE FROM sessions WHERE token_hash = ?", (digest,))
            return None
        actor = self.actor_for(row["person_id"])
        if actor is None:
            self.store.conn.execute("DELETE FROM sessions WHERE token_hash = ?", (digest,))
            return None
        self.store.conn.execute(
            "UPDATE sessions SET last_seen = ? WHERE token_hash = ?", (now, digest)
        )
        account = self.account(actor.id) or {}
        return actor, Session(
            token=token,
            csrf_token=row["csrf_token"],
            person_id=actor.id,
            must_change_password=bool(account.get("must_change_password")),
        )

    def end_session(self, token: str) -> None:
        self.store.conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))

    def end_all_sessions(self, person_id: str) -> None:
        self.store.conn.execute("DELETE FROM sessions WHERE person_id = ?", (person_id,))

    # ---------------------------------------------------------- api tokens

    def create_token(self, actor: Actor, label: str) -> str:
        """A personal token for scripts. It acts as the person who created it."""
        if not actor.authenticated:
            raise AuthenticationError("only a signed-in person can create a personal token")
        token = "iai_" + secrets.token_urlsafe(32)
        token_id = new_id("tok")
        created = now_iso()
        expires = (datetime.fromisoformat(created) + timedelta(days=TOKEN_LIFETIME_DAYS)).isoformat(
            timespec="seconds"
        )
        self.store.conn.execute(
            "INSERT INTO api_tokens (id, token_hash, person_id, label, created_at, expires_at)"
            " VALUES (?,?,?,?,?,?)",
            (token_id, token_hash(token), actor.id, label.strip() or "unnamed", created, expires),
        )
        self.audit.record(
            actor=actor.id,
            actor_kind="human",
            action="api_token.created",
            subject=token_id,
            details={"label": label},
        )
        return token

    def list_tokens(self, person_id: str) -> list[dict[str, Any]]:
        rows = self.store.conn.execute(
            "SELECT id, label, created_at, expires_at, last_used_at, revoked FROM api_tokens"
            " WHERE person_id = ? ORDER BY created_at, id",
            (person_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def revoke_token(self, token_id: str, actor: Actor) -> None:
        row = self.store.conn.execute(
            "SELECT person_id FROM api_tokens WHERE id = ?", (token_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown token: {token_id}")
        if row["person_id"] != actor.id:
            require(self.audit, actor, Action.MANAGE_PEOPLE, token_id)
        self.store.conn.execute("UPDATE api_tokens SET revoked = 1 WHERE id = ?", (token_id,))
        self.audit.record(
            actor=actor.id, actor_kind="human", action="api_token.revoked", subject=token_id
        )

    def token_actor(self, token: str) -> Actor | None:
        if not token:
            return None
        row = self.store.conn.execute(
            "SELECT id, person_id, revoked, expires_at FROM api_tokens WHERE token_hash = ?",
            (token_hash(token),),
        ).fetchone()
        if row is None or row["revoked"]:
            return None
        if row["expires_at"] and row["expires_at"] <= now_iso():
            return None
        actor = self.actor_for(row["person_id"])
        if actor is None:
            return None
        self.store.conn.execute(
            "UPDATE api_tokens SET last_used_at = ? WHERE id = ?", (now_iso(), row["id"])
        )
        return actor
