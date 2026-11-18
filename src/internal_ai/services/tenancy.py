"""Which tenant a record, a person and a request belong to.

A group decides whether a person may read a class of material
(`services/attributes.py`). A tenant decides whether the material is theirs to
be asked about at all, and that is not a grant: no role and no group widens it,
the superuser group included. An install that let one customer's administrator
read another customer's corpus would have to be one deployment per customer,
and this is the check that says so before the first document is read.

The tenant travels with the record. A source is registered into one tenant and
every record read from it carries that tenant whatever the row itself claims,
the way a row cannot lower the class of the file it came from.
"""

from __future__ import annotations

import re

DEFAULT_TENANT = "default"
"""The tenant an install is, before anyone names another."""

_TENANT_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class TenantError(ValueError):
    """A tenant name that cannot be used."""


def normalise(tenant: str) -> str:
    """The name a record, a source and a principal are compared on."""
    return (tenant or DEFAULT_TENANT).strip().lower()


def check(tenant: str) -> str:
    """Refuse a tenant name that would not survive a header or a URL."""
    name = normalise(tenant)
    if not _TENANT_NAME.match(name):
        raise TenantError(
            f"{tenant!r} is not a tenant name: lower case letters, digits and '-' only"
        )
    return name


def same_tenant(left: str, right: str) -> bool:
    return normalise(left) == normalise(right)


def other_tenant(kind: str, tenant: str) -> str:
    """Why a record of another tenant is out of reach, without naming the record."""
    return (
        f"{kind} belongs to tenant {normalise(tenant)}, not this one; "
        "a tenant is a boundary, not a grant"
    )
