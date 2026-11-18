"""What a person's attributes grant, evaluated with the same check as the role.

`permissions.ACTION_ROLES` answers "may this person take this action at all".
It cannot answer "on which record": a person who may read traces still must not
read another group's trace, and no role makes confidential material readable.
Those are attributes -- the person's groups, the class of the data and the ACL
the record inherited from its source -- and they are evaluated by the same
`require(...)` call as the role, because a second enforcement point is a second
place to forget one.

The grants are written by a person in policy decision P7, never assumed here.
The default is the rule the platform enforced before attributes existed: a
signed-in person reads internal material, nothing higher, and a record carrying
an ACL also needs one of that person's groups in it.

A value that cannot be read grants nothing at all, internal material included.
An unreadable policy is not a licence, and a parser that fell back to "open"
would turn a typo into an incident.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

SUPERUSER_GROUP = "platform-admins"
"""The one group that bypasses a record's ACL. It does not bypass a class grant."""

DEFAULT_CLASS = "internal"

ANYONE: frozenset[str] = frozenset()
"""No group needed, which is what ``class=anyone`` says in a written grant."""

_CLASS_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_GROUP_NAME = re.compile(r"^[A-Za-z0-9_.@-]+$")


class AttributePolicyError(ValueError):
    """A written grant list that cannot be read."""


@dataclass(frozen=True, slots=True)
class Resource:
    """What a check is about: the kind of record, its class and its ACL."""

    kind: str
    sensitivity: str = DEFAULT_CLASS
    acl: frozenset[str] = frozenset()

    @classmethod
    def of(cls, kind: str, sensitivity: str = "", acl: Iterable[str] = ()) -> Resource:
        return cls(kind=kind, sensitivity=sensitivity or DEFAULT_CLASS, acl=frozenset(acl))

    @property
    def data_class(self) -> str:
        return (self.sensitivity or DEFAULT_CLASS).strip().lower()


@dataclass(frozen=True, slots=True)
class ReadPolicy:
    """Which data classes each group may read."""

    grants: Mapping[str, frozenset[str]]

    @classmethod
    def strict(cls) -> ReadPolicy:
        return cls({DEFAULT_CLASS: ANYONE})

    @classmethod
    def denied(cls) -> ReadPolicy:
        return cls({})

    @classmethod
    def parse(cls, raw: str) -> ReadPolicy:
        """Read ``class=group|group`` entries, or ``class=anyone``.

        Anything else is refused rather than guessed at: a class named twice and
        an entry with no group both read as "somebody meant something", and
        resolving that in the reader's favour is how a policy stops meaning what
        its author wrote.
        """
        grants: dict[str, frozenset[str]] = {}
        for entry in (raw or "").split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, separator, granted = entry.partition("=")
            name = name.strip().lower()
            granted = granted.strip()
            if not separator or not name:
                raise AttributePolicyError(f"{entry!r} is not written class=groups")
            if not _CLASS_NAME.match(name):
                raise AttributePolicyError(f"{name!r} is not a data class name")
            if name in grants:
                raise AttributePolicyError(f"{name} is granted twice")
            if not granted:
                raise AttributePolicyError(
                    f"{name} names no group; leave the class out to make it unreadable"
                )
            if granted == "anyone":
                grants[name] = ANYONE
                continue
            groups = frozenset(part.strip() for part in granted.split("|") if part.strip())
            if not groups:
                raise AttributePolicyError(f"{name} names no group")
            unknown = sorted(group for group in groups if not _GROUP_NAME.match(group))
            if unknown:
                raise AttributePolicyError(f"{unknown[0]!r} is not a group name")
            grants[name] = groups
        if not grants:
            raise AttributePolicyError("a read policy names at least one data class")
        return cls(grants)

    def allows(
        self, groups: frozenset[str], sensitivity: str = "", acl: Iterable[str] = ()
    ) -> bool:
        """May someone in ``groups`` read a record of this class and ACL?

        A class that is not granted is not readable, whatever the ACL says. A
        granted class with no group is readable by anyone signed in. An ACL is a
        second gate on the same record, and the superuser group is the only one
        that skips it.
        """
        klass = (sensitivity or DEFAULT_CLASS).strip().lower()
        granted = self.grants.get(klass)
        if granted is None:
            return False
        if granted and not (groups & granted):
            return False
        restricted = frozenset(acl)
        if not restricted or SUPERUSER_GROUP in groups:
            return True
        return bool(groups & restricted)

    def describe(self) -> list[dict[str, str]]:
        """The grants as rows for a page, sorted so two reads agree."""
        return [
            {"class": name, "groups": "anyone" if not groups else "|".join(sorted(groups))}
            for name, groups in sorted(self.grants.items())
        ]
