"""Who is acting.

An actor is a person from the roster. When the identity came from a session,
a personal token or a CLI credential it is authenticated; a bare name typed on
the command line in a root with no accounts is asserted, and is recorded as
such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from internal_ai.serving.service import Principal
from internal_ai.store import Store


@dataclass(frozen=True, slots=True)
class Actor:
    id: str
    name: str
    roles: frozenset[str]
    groups: frozenset[str]
    authenticated: bool
    kind: str = "human"

    def holds(self, *roles: str) -> bool:
        return bool(self.roles & set(roles))

    def principal(self) -> Principal:
        return Principal(user_id=self.id, groups=self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "roles": sorted(self.roles),
            "groups": sorted(self.groups),
            "authenticated": self.authenticated,
        }

    @classmethod
    def from_person(cls, person: dict[str, Any], authenticated: bool) -> Actor:
        return cls(
            id=person["id"],
            name=person["name"],
            roles=frozenset(person["roles"]),
            groups=frozenset(person["groups"]),
            authenticated=authenticated,
        )

    @classmethod
    def asserted(cls, store: Store, name: str) -> Actor:
        """Resolve a typed name the way gate approvals always have.

        The id stays the name as typed, so audit records and dual-control
        messages read exactly as they did before accounts existed.
        """
        from internal_ai.gates import GateService
        from internal_ai.people import PeopleRegistry

        person = PeopleRegistry(store).get(name)
        groups = frozenset(person["groups"]) if person else frozenset()
        roles = frozenset(GateService(store).approver_roles(name))
        return cls(id=name, name=name, roles=roles, groups=groups, authenticated=False)
