"""Provisioning a directory's people and groups into this install.

A customer's identity provider is the system of record for who works there. This
module lets it say so over SCIM 2.0 instead of an administrator retyping a
roster, and it draws the line in the same place the OIDC slice did: the directory
may create a person, put them in a group and take them away again, and it may not
decide a gate, hold a role the owner did not map, or sign anybody in. A
provisioned person still signs in only through a `(issuer, subject)` link an
administrator made, because a directory that can rewrite its own subject list
must not be able to walk into an account.

The caller is authenticated by one bearer token from configuration, never by a
person's credential: a program lead's personal token is not a directory's word,
and is refused here for that reason.

Roles are derived, not supplied. The owner writes `INTERNAL_AI_SCIM_GROUP_ROLES`
— a JSON object of directory group to role names — and a person's roles are the
union of what their groups map to. A group the owner did not map is refused
rather than provisioned with no roles, so a typo in a chart does not quietly
create an unprivileged account or, worse, an over-privileged one. People the
directory manages have their roles recomputed on every membership change;
people this install manages keep theirs.

Deprovisioning disables the account and ends its sessions and personal tokens.
It refuses outright when the person is the last enabled program lead, because a
directory mistake that locks the customer out of their own control plane is not
a change that should complete.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import JSONResponse

from internal_ai.context import Platform
from internal_ai.domain import AuditOutcome
from internal_ai.people import PeopleRegistry
from internal_ai.services.identity import IdentityService

ENV_TOKEN = "INTERNAL_AI_SCIM_TOKEN"
ENV_GROUP_ROLES = "INTERNAL_AI_SCIM_GROUP_ROLES"

SERVICE = "scim"
SCIM_JSON = "application/scim+json"

USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"
ERROR_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:Error"
LIST_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
CONFIG_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"

PROVISIONABLE_ROLES = frozenset(
    {
        "program_lead",
        "ml_lead",
        "data_owner",
        "domain_expert",
        "finance",
        "legal",
        "compliance",
        "security",
        "engineering_manager",
        "infra_engineer",
    }
)
"""The roles a directory may hand out.

These are the roles the permission map and the gates already name. Anything
else — an agent role, a typo, a role someone invented in a chart — is refused
when the mapping is read, so a bad mapping is a startup refusal rather than a
person who silently holds nothing, or something this code does not know about.
"""


class ScimError(Exception):
    """A SCIM request that was refused, with the status and body SCIM expects."""

    status = 400

    def __init__(self, detail: str, status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status is not None:
            self.status = status

    def payload(self) -> dict[str, Any]:
        return {"schemas": [ERROR_SCHEMA], "detail": self.detail, "status": str(self.status)}


class ScimBadRequest(ScimError):
    pass


class ScimDenied(ScimError):
    status = 401


class ScimForbidden(ScimError):
    status = 403


class ScimNotFound(ScimError):
    status = 404


class ScimConflict(ScimError):
    status = 409


@dataclass(frozen=True, slots=True)
class ScimConfig:
    token: str
    group_roles: dict[str, tuple[str, ...]]

    @classmethod
    def configured(cls) -> ScimConfig | None:
        """This install's directory integration, or None when it has none.

        A token with no mapping provisions people with no roles; a mapping with
        no token exports the roster to anyone who asks, so it is refused rather
        than half-running.
        """
        token = os.environ.get(ENV_TOKEN, "").strip()
        raw = os.environ.get(ENV_GROUP_ROLES, "").strip()
        if not token and not raw:
            return None
        if not token:
            raise ScimError(f"provisioning is only half configured; set {ENV_TOKEN}", status=503)
        if not raw:
            return cls(token=token, group_roles={})
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ScimError(f"{ENV_GROUP_ROLES} is not valid JSON: {exc.msg}", status=503) from exc
        if not isinstance(parsed, dict):
            raise ScimError(
                f"{ENV_GROUP_ROLES} must be an object of group names to role lists", status=503
            )
        mapping: dict[str, tuple[str, ...]] = {}
        for key, roles in parsed.items():
            group = str(key).strip()
            if not group:
                raise ScimError(f"{ENV_GROUP_ROLES} has a group with no name", status=503)
            if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
                raise ScimError(
                    f"{ENV_GROUP_ROLES}: the roles for {group} must be a list of role names",
                    status=503,
                )
            unknown = sorted({role.strip() for role in roles} - PROVISIONABLE_ROLES)
            if unknown:
                raise ScimError(
                    f"{ENV_GROUP_ROLES}: {group} names roles this install does not have: "
                    + ", ".join(unknown),
                    status=503,
                )
            mapping[group] = tuple(sorted({role.strip() for role in roles if role.strip()}))
        return cls(token=token, group_roles=mapping)

    def matches(self, presented: str) -> bool:
        return bool(presented) and hmac.compare_digest(self.token, presented)

    def roles_for(self, groups: Iterable[str]) -> tuple[str, ...]:
        roles: set[str] = set()
        for group in groups:
            roles.update(self.group_roles.get(group, ()))
        return tuple(sorted(roles))

    def requires(self, group: str) -> tuple[str, ...]:
        try:
            return self.group_roles[group]
        except KeyError:
            known = ", ".join(sorted(self.group_roles)) or "none"
            raise ScimNotFound(
                f"{group} is not a group this install maps; it knows: {known}"
            ) from None


def service_provider_config() -> dict[str, Any]:
    """What this implementation actually supports, so a client does not guess.

    PATCH is the honest omission: membership changes arrive as a PUT of the whole
    resource, and `bulk` would have to be transactional across several people to
    be worth claiming.
    """
    return {
        "schemas": [CONFIG_SCHEMA],
        "patch": {"supported": False},
        "bulk": {"supported": False, "maxOperations": 0, "maxPayloadSize": 0},
        "filter": {"supported": True, "maxResults": 200},
        "changePassword": {"supported": False},
        "sort": {"supported": False},
        "etag": {"supported": False},
        "authenticationSchemes": [
            {
                "type": "oauthbearertoken",
                "name": "Bearer token",
                "description": f"the service token in {ENV_TOKEN}",
                "primary": True,
            }
        ],
    }


def error_response(_: Request, exc: Exception) -> JSONResponse:
    """The handler api.py registers, so a refusal keeps its SCIM shape."""
    if not isinstance(exc, ScimError):
        exc = ScimError(str(exc))
    return JSONResponse(status_code=exc.status, content=exc.payload(), media_type=SCIM_JSON)


def refuse(platform: Platform, why: str, subject: str = "") -> ScimDenied:
    """Record a refused caller and return the error to raise."""
    platform.audit.record(
        actor=SERVICE,
        actor_kind="service",
        action="scim.denied",
        subject=subject or "directory",
        outcome=AuditOutcome.DENIED,
        details={"reason": why},
    )
    return ScimDenied(why)


class ScimService:
    """Users and groups, as the directory sees them through this API."""

    def __init__(self, platform: Platform, config: ScimConfig) -> None:
        self.platform = platform
        self.config = config
        self.people = PeopleRegistry(platform.store)
        self.identity = IdentityService(platform)

    # ---------------------------------------------------------------- users

    def create_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = self._name(payload)
        if self.people.get(name) is not None:
            raise ScimConflict(f"a person named {name} already exists here")
        external = self._external_id(payload)
        if external and self.people.by_external_id(external) is not None:
            raise ScimConflict(f"externalId {external} is already provisioned here")
        groups = self._groups(payload)
        roles = self.config.roles_for(groups)
        person = self.people.new_directory_person(name, groups, roles, external)
        self.identity.provision_from_directory(person["id"], SERVICE)
        self._record(
            "scim.user_created",
            person["id"],
            {
                "userName": name,
                "externalId": external,
                "groups": list(groups),
                "roles": list(roles),
            },
        )
        if payload.get("active") is False:
            # Disabled straight away rather than through `deprovision`: this
            # person was never an enabled administrator, so the last-program-lead
            # guard has nothing to protect and would refuse a valid creation.
            self._disable(person["id"], person["name"])
        return self.user_resource(person["id"])

    def replace_user(self, person_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        person = self._managed(person_id)
        name = self._name(payload)
        if name != person["name"]:
            other = self.people.get(name)
            if other is not None and other["id"] != person["id"]:
                raise ScimConflict(f"a person named {name} already exists here")
        external = self._external_id(payload) or person["external_id"]
        if external != person["external_id"]:
            other = self.people.by_external_id(external)
            if other is not None and other["id"] != person["id"]:
                raise ScimConflict(f"externalId {external} is already provisioned here")
        groups = self._groups(payload) if "groups" in payload else list(person["groups"])
        roles = self.config.roles_for(groups)
        if payload.get("active") is False:
            # Checked before anything is written, so a refusal leaves the person
            # exactly as the directory found them.
            self._last_program_lead(person)
        self.people.update_directory_person(person["id"], name, groups, roles, external)
        self._record(
            "scim.user_updated",
            person["id"],
            {
                "userName": name,
                "externalId": external,
                "groups": list(groups),
                "roles": list(roles),
                "roles_before": list(person["roles"]),
            },
        )
        active = payload.get("active")
        if active is False:
            self._disable(person["id"], person["name"])
        elif active is True:
            self._enable(person["id"])
        return self.user_resource(person["id"])

    def delete_user(self, person_id: str) -> None:
        self.deprovision(person_id)

    def deprovision(self, person_id: str) -> None:
        person = self._managed(person_id)
        self._last_program_lead(person)
        self._disable(person["id"], person["name"])

    def list_users(self, filter_text: str = "") -> list[dict[str, Any]]:
        if filter_text.strip():
            wanted = self._filter_name(filter_text)
            person = self.people.get(wanted)
            if person is None or not person["scim_managed"]:
                return []
            return [self.user_resource(person["id"])]
        return [self.user_resource(person["id"]) for person in self._managed_people()]

    def user_resource(self, person_id: str) -> dict[str, Any]:
        person = self._managed(person_id)
        account = self.identity.account(person["id"]) or {}
        return {
            "schemas": [USER_SCHEMA],
            "id": person["id"],
            "externalId": person["external_id"],
            "userName": person["name"],
            "displayName": person["name"],
            "active": not bool(account.get("disabled", 0)),
            "groups": [{"value": group, "display": group} for group in person["groups"]],
            "meta": {
                "resourceType": "User",
                "location": f"/scim/v2/Users/{person['id']}",
            },
        }

    # --------------------------------------------------------------- groups

    def list_groups(self) -> list[dict[str, Any]]:
        return [self.group_resource(name) for name in sorted(self.config.group_roles)]

    def group_resource(self, name: str) -> dict[str, Any]:
        self.config.requires(name)
        members = [
            {"value": person["id"], "display": person["name"]}
            for person in self._directory_people()
            if name in person["groups"]
        ]
        return {
            "schemas": [GROUP_SCHEMA],
            "id": name,
            "displayName": name,
            "members": members,
            "meta": {
                "resourceType": "Group",
                "location": f"/scim/v2/Groups/{quote(name, safe='')}",
            },
        }

    def set_members(self, name: str, member_ids: list[str]) -> dict[str, Any]:
        """Replace a group's membership, and recompute what its roles mean.

        Only people this install considers real are members: the shipped roster
        is a set of examples nobody at the customer's company holds, and a
        directory that emptied a group would otherwise rewrite it.
        """
        self.config.requires(name)
        wanted: list[str] = []
        for member_id in member_ids:
            person = self.people.get(member_id)
            if person is None or person["placeholder"]:
                raise ScimNotFound(f"unknown user: {member_id}")
            if person["id"] not in wanted:
                wanted.append(person["id"])
        added: list[str] = []
        removed: list[str] = []
        changed: list[dict[str, Any]] = []
        for person in self._directory_people():
            in_group = name in person["groups"]
            if in_group == (person["id"] in wanted):
                continue
            groups = [group for group in person["groups"] if group != name]
            if person["id"] in wanted:
                groups.append(name)
            roles = list(person["roles"])
            if person["scim_managed"]:
                roles = list(self.config.roles_for(groups))
                if roles != person["roles"]:
                    changed.append(
                        {
                            "person": person["id"],
                            "before": person["roles"],
                            "after": roles,
                        }
                    )
            self.people.set_directory_membership(person["id"], groups, roles)
            if person["id"] in wanted:
                added.append(person["id"])
            else:
                removed.append(person["id"])
        self._record(
            "scim.group_updated",
            name,
            {
                "added": added,
                "removed": removed,
                "roles_recomputed": changed,
                "members": wanted,
            },
        )
        return self.group_resource(name)

    # -------------------------------------------------------------- helpers

    def _managed(self, person_id: str) -> dict[str, Any]:
        """The directory's person, refusing one this install manages.

        A SCIM client that guesses an id gets the same answer as one that guesses
        a name: the roster is not its resource to edit, and provisioning a
        second time must not overwrite a person an administrator created.
        """
        person = self.people.get(person_id)
        if person is None or not person["scim_managed"]:
            raise ScimNotFound(f"unknown user: {person_id}")
        return person

    def _managed_people(self) -> list[dict[str, Any]]:
        return [person for person in self.people.list_people() if person["scim_managed"]]

    def _directory_people(self) -> list[dict[str, Any]]:
        """The people a directory may hold group membership for."""
        return [person for person in self.people.list_people() if not person["placeholder"]]

    def _name(self, payload: dict[str, Any]) -> str:
        name = str(payload.get("userName", "")).strip()
        if not name:
            raise ScimBadRequest("a user needs a userName")
        if len(name) > 200:
            raise ScimBadRequest("userName is longer than 200 characters")
        return name

    def _external_id(self, payload: dict[str, Any]) -> str:
        value = str(payload.get("externalId", "") or "").strip()
        if len(value) > 200:
            # Truncating would make a long id silently equal a different long id.
            raise ScimBadRequest("externalId is longer than 200 characters")
        return value

    def _groups(self, payload: dict[str, Any]) -> list[str]:
        raw = payload.get("groups", [])
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ScimBadRequest("groups must be a list")
        groups: list[str] = []
        for item in raw:
            value = item.get("value") if isinstance(item, dict) else item
            group = str(value or "").strip()
            if not group:
                continue
            self.config.requires(group)
            if group not in groups:
                groups.append(group)
        return groups

    def _filter_name(self, filter_text: str) -> str:
        """Read the one filter an IdP needs: `userName eq "someone"`."""
        spacing = filter_text.strip()
        lowered = spacing.lower()
        if not lowered.startswith("username eq "):
            raise ScimBadRequest("only `userName eq` filters are supported")
        value = spacing[len("userName eq ") :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value.strip():
            raise ScimBadRequest("the filter names no userName")
        return value.strip()

    def _last_program_lead(self, person: dict[str, Any]) -> None:
        if "program_lead" not in person["roles"]:
            return
        rows = self.platform.store.conn.execute(
            "SELECT p.id, p.roles, p.placeholder FROM people p"
            " JOIN accounts a ON a.person_id = p.id"
            " WHERE a.disabled = 0"
        ).fetchall()
        others = [
            row
            for row in rows
            if row["id"] != person["id"]
            and not row["placeholder"]
            and "program_lead" in row["roles"].split(",")
        ]
        if not others:
            raise ScimConflict(
                f"deprovisioning {person['name']} would leave this install with no enabled "
                "program lead; add another before the directory removes this one"
            )

    def _disable(self, person_id: str, name: str) -> None:
        self.identity.set_disabled_from_directory(person_id, True, SERVICE)
        self._record(
            "scim.user_deprovisioned",
            person_id,
            {"userName": name, "sessions_revoked": True, "tokens_revoked": True},
        )

    def _enable(self, person_id: str) -> None:
        self.identity.set_disabled_from_directory(person_id, False, SERVICE)
        self._record("scim.user_reprovisioned", person_id, {})

    def _record(self, action: str, subject: str, details: dict[str, Any]) -> None:
        self.platform.audit.record(
            actor=SERVICE,
            actor_kind="service",
            action=action,
            subject=subject,
            details=details,
        )
