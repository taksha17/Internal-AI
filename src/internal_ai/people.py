"""The human roster.

Gates are decided by humans, and the audit trail records who approved what.
This module holds the people the platform knows about and the roles they hold.

The shipped roster is a set of placeholder identities. They exist so the
approval flows, gates and audit records have consistent, distinguishable
actors, and so no single persona accumulates every approval. They are marked
as placeholders and must be replaced with real people before any decision
carves real consequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from internal_ai.ids import new_id, now_iso
from internal_ai.store import Store


@dataclass(frozen=True, slots=True)
class Person:
    id: str
    name: str
    title: str
    roles: tuple[str, ...]
    placeholder: bool = True
    groups: tuple[str, ...] = ()

    def can_approve(self, approver_role: str) -> bool:
        return approver_role in self.roles

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "roles": list(self.roles),
            "groups": list(self.groups),
            "placeholder": self.placeholder,
        }


ROSTER: tuple[Person, ...] = (
    Person(
        id="ppl_priya_raghunathan",
        name="Priya Raghunathan",
        title="Program lead",
        roles=("program_lead", "finance"),
        groups=("platform-admins",),
    ),
    Person(
        id="ppl_marcus_delgado",
        name="Marcus Delgado",
        title="Senior ML lead",
        roles=("ml_lead",),
        groups=("platform-admins",),
    ),
    Person(
        id="ppl_aisha_okonkwo",
        name="Aisha Okonkwo",
        title="Data engineering lead",
        roles=("data_owner", "domain_expert"),
        groups=("support", "engineering"),
    ),
    Person(
        id="ppl_tomas_lindqvist",
        name="Tomás Lindqvist",
        title="Security and compliance partner",
        roles=("legal", "compliance", "security"),
        groups=("security",),
    ),
    Person(
        id="ppl_naomi_bergstrom",
        name="Naomi Bergström",
        title="Operations domain expert",
        roles=("domain_expert",),
        groups=("support",),
    ),
    Person(
        id="ppl_ravi_chandrasekaran",
        name="Ravi Chandrasekaran",
        title="Finance business partner",
        roles=("finance",),
        groups=(),
    ),
    Person(
        id="ppl_elena_vasquez",
        name="Elena Vasquez",
        title="Customer operations lead",
        roles=("data_owner", "domain_expert"),
        groups=("support",),
    ),
    Person(
        id="ppl_samuel_adeyemi",
        name="Samuel Adeyemi",
        title="Engineering manager",
        roles=("engineering_manager", "data_owner"),
        groups=("engineering", "platform-admins"),
    ),
)

ROSTER_BY_ROLE: dict[str, Person] = {}
for _person in ROSTER:
    for _role in _person.roles:
        ROSTER_BY_ROLE.setdefault(_role, _person)


def person_for_role(role: str) -> Person:
    try:
        return ROSTER_BY_ROLE[role]
    except KeyError as exc:
        raise KeyError(f"no human holds the role: {role}") from exc


def find_person(name: str) -> Person:
    for person in ROSTER:
        if person.name == name or person.id == name:
            return person
    raise KeyError(f"unknown person: {name}")


def roster() -> tuple[Person, ...]:
    return ROSTER


class PeopleRegistry:
    """Persisted view of the human roster."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def seed(self) -> list[Person]:
        seeded: list[Person] = []
        for person in ROSTER:
            self.store.conn.execute(
                """
                INSERT INTO people (id, name, title, roles, groups, placeholder, created_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, title=excluded.title, roles=excluded.roles,
                    groups=excluded.groups, placeholder=excluded.placeholder
                """,
                (
                    person.id,
                    person.name,
                    person.title,
                    ",".join(person.roles),
                    ",".join(person.groups),
                    int(person.placeholder),
                    now_iso(),
                ),
            )
            seeded.append(person)
        return seeded

    def list_people(self) -> list[dict[str, Any]]:
        self.seed()
        rows = self.store.conn.execute("SELECT * FROM people ORDER BY name").fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "title": row["title"],
                "roles": [role for role in row["roles"].split(",") if role],
                "groups": [group for group in row["groups"].split(",") if group],
                "placeholder": bool(row["placeholder"]),
            }
            for row in rows
        ]

    def approvers_for(self, gate_type: str) -> list[dict[str, Any]]:
        from internal_ai.domain import GateType
        from internal_ai.gates import HUMAN_APPROVER_ROLES

        try:
            roles = HUMAN_APPROVER_ROLES[GateType(gate_type)]
        except KeyError:
            return []
        people = []
        for role in roles:
            person = person_for_role(role)
            people.append({**person.to_dict(), "approver_role": role})
        return people

    def replace(self, person_id: str, name: str, title: str, roles: list[str]) -> dict[str, Any]:
        """Replace a placeholder identity with a real person."""
        self.seed()
        self.store.conn.execute(
            "UPDATE people SET name = ?, title = ?, roles = ?, placeholder = 0 WHERE id = ?",
            (name, title, ",".join(roles), person_id),
        )
        return {
            "id": person_id,
            "name": name,
            "title": title,
            "roles": roles,
            "placeholder": False,
        }

    def new_person(
        self, name: str, title: str, roles: list[str], groups: list[str] | None = None
    ) -> dict[str, Any]:
        self.seed()
        person_id = new_id("ppl")
        self.store.conn.execute(
            "INSERT INTO people (id, name, title, roles, groups, placeholder, created_at)"
            " VALUES (?,?,?,?,?,0,?)",
            (
                person_id,
                name,
                title,
                ",".join(roles),
                ",".join(groups or []),
                now_iso(),
            ),
        )
        return {
            "id": person_id,
            "name": name,
            "title": title,
            "roles": roles,
            "groups": groups or [],
            "placeholder": False,
        }
