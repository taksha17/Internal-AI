"""A complete programme record for tests that have to pass the brief gate."""

from __future__ import annotations

from internal_ai.brief import OWNER_ROLES, Metric, ProjectBrief, save
from internal_ai.context import Platform
from internal_ai.people import PeopleRegistry
from internal_ai.services.actor import Actor
from internal_ai.services.identity import IdentityService

OWNER_NAMES = (
    ("ml_lead", "Ira Lead"),
    ("data_engineer", "Ned Engineer"),
    ("infra", "Ian Infra"),
    ("evaluation", "Eva Eval"),
    ("security", "Sam Secure"),
)


def record_ready_programme(
    platform: Platform,
    writer: Actor,
    owners: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write every brief field and mark the data-class policy decided."""
    if owners is None:
        from conftest import enroll

        owners = {}
        for role, name in OWNER_NAMES:
            existing = PeopleRegistry(platform.store).get(name)
            if existing is None:
                owners[role] = enroll(platform, name, [role]).id
            else:
                owners[role] = existing["id"]
    missing = [role for role in OWNER_ROLES if role not in owners]
    if missing:
        raise ValueError(f"owners missing {missing}")
    save(
        platform.store,
        platform.audit,
        writer,
        ProjectBrief(
            target="answer from internal tickets",
            languages="en",
            context_length=256,
            quality_bar="cite the file the sentence came from",
            metrics=[Metric(name="recall_at_10", threshold="0.2")],
            budget_usd=10,
            timeline="this quarter",
            deployment="on the company network",
            latency_ms=2000,
            owners=dict(owners),
            non_goals="no public model and no third-party answers",
        ),
    )
    decision = platform.policy.get("P5")
    decision.status = "decided"
    decision.decided_by = writer.id
    platform.store.upsert_policy(decision)
    return owners


def actor_named(platform: Platform, name: str) -> Actor:
    person = PeopleRegistry(platform.store).get(name)
    if person is None:
        raise KeyError(name)
    actor = IdentityService(platform).actor_for(person["id"])
    if actor is None:
        raise KeyError(name)
    return actor
