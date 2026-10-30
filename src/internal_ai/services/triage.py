"""The queue of failed production answers, read by someone allowed to see it.

Reading a trace is reading what a colleague asked and what the model answered,
so it needs a capability of its own rather than riding on any signed-in person.
The storage layer already filters by the trace's ACL; this adds the role check
and leaves the two independent, because a person with the role still must not
read another group's traces.
"""

from __future__ import annotations

from internal_ai.context import Platform
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require
from internal_ai.serving.trace import Trace, triage


def triage_view(platform: Platform, actor: Actor, limit: int = 25) -> dict[str, list[Trace]]:
    """Failed traces this person may read, bucketed by why they failed."""
    require(platform.audit, actor, Action.VIEW_TRACES, "console/traces")
    return triage(platform.store, actor.principal(), limit=limit)
