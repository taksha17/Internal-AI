"""Who a model may answer for: the ACL groups its training text was restricted to.

Retrieval enforces a document's ACL on every request, but a model carries what it
learned past retrieval: an adapter trained on a restricted document can repeat it
to anyone who can call the model. The audience is the intersection of the ACL
groups of the restricted records it learned from, recorded on the card, and
serving refuses a principal outside it.
"""

from __future__ import annotations

from typing import Any


def audience_for(rows: list[dict[str, Any]]) -> list[str]:
    """The groups that may read every restricted record the model learned from.

    A record with no ACL is readable by anyone with a key and does not narrow the
    audience. A record that names groups narrows it to the groups every such
    record shares. No restricted record at all means nobody is excluded, and an
    empty list says so.
    """
    groups: set[str] | None = None
    for row in rows:
        acl = {str(group).strip() for group in (row.get("acl") or []) if str(group).strip()}
        if not acl:
            continue
        groups = acl if groups is None else groups & acl
    return sorted(groups or ())
