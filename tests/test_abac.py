"""Attributes are checked with the capability, and a grant is only what P7 says.

The rules under test: a class nobody granted is unreadable, a grant applies only
after its gate is decided, a pending proposal changes nothing, a value that
cannot be read grants nothing at all, and the roles that must not decide P7 are
refused before any value is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.policy import P7_POLICY_ID, POLICY_ITEMS, read_policy
from internal_ai.services import governance, triage
from internal_ai.services.actor import Actor
from internal_ai.services.attributes import AttributePolicyError, ReadPolicy, Resource
from internal_ai.services.permissions import Action, PermissionDenied, allowed, require
from internal_ai.serving.service import Principal
from internal_ai.serving.trace import FAILURE_EMPTY, record_trace

pytestmark = pytest.mark.filterwarnings("ignore:Using `httpx` with `starlette.testclient`")


def person(roles: set[str], groups: set[str], kind: str = "human") -> Actor:
    return Actor(
        id="p_reader",
        name="Reader",
        roles=frozenset(roles),
        groups=frozenset(groups),
        authenticated=True,
        kind=kind,
    )


def approve_gate(platform, gate_id: str) -> None:
    """Close the gate the way an approved decision leaves it.

    The gate's own role checks have their own tests; what is checked here is
    whether a value applies before or after the gate is decided.
    """
    stored = platform.store.get_gate(gate_id)
    assert stored is not None
    stored.status = type(stored.status)("approved")
    platform.store.update_gate(stored)


def decide_p7(platform, actor: Actor, value: str) -> None:
    _decision, gate_id = governance.propose_policy(
        platform, actor, P7_POLICY_ID, value, "counsel reviewed the classes"
    )
    approve_gate(platform, gate_id)
    governance.confirm_policy(platform, actor, P7_POLICY_ID)


def last_denial(platform) -> dict[str, object]:
    row = platform.store.conn.execute(
        "SELECT details FROM audit_events WHERE action = 'console.denied' ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    return json.loads(row["details"])


def test_the_seeded_grants_are_the_rule_that_already_held() -> None:
    policy = ReadPolicy.strict()
    assert policy.allows(frozenset({"engineering"}), "internal", [])
    assert policy.allows(frozenset(), "", [])
    assert not policy.allows(frozenset({"engineering"}), "confidential", [])
    # A superuser skips a record's ACL, never the class grant.
    assert not policy.allows(frozenset({"platform-admins"}), "confidential", [])
    assert not policy.allows(frozenset({"engineering"}), "internal", ["legal"])
    assert policy.allows(frozenset({"legal"}), "internal", ["legal"])
    assert policy.allows(frozenset({"platform-admins"}), "internal", ["legal"])


def test_a_written_grant_widens_one_class_for_one_group() -> None:
    policy = ReadPolicy.parse("internal=anyone,confidential=legal|compliance")
    assert policy.allows(frozenset({"legal"}), "confidential", [])
    assert not policy.allows(frozenset({"engineering"}), "confidential", [])
    assert not policy.allows(frozenset({"engineering"}), "regulated", [])
    assert policy.describe() == [
        {"class": "confidential", "groups": "compliance|legal"},
        {"class": "internal", "groups": "anyone"},
    ]


def test_a_value_that_cannot_be_read_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(AttributePolicyError):
        ReadPolicy.parse("internal=anyone;confidential=legal")
    with pytest.raises(AttributePolicyError):
        ReadPolicy.parse("confidential=")
    with pytest.raises(AttributePolicyError):
        ReadPolicy.parse("internal=anyone,internal=legal")
    with pytest.raises(AttributePolicyError):
        ReadPolicy.parse("  ")
    # An unreadable policy is not an open one.
    assert not ReadPolicy.denied().allows(frozenset({"platform-admins"}), "internal", [])


def test_p7_is_seeded_strict_and_a_typo_never_reaches_a_gate(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Pat Lead", ["program_lead"])
    assert P7_POLICY_ID in {item.id for item in POLICY_ITEMS}
    assert read_policy(platform.policy).grants == ReadPolicy.strict().grants
    before = len(platform.store.list_gates())
    with pytest.raises(ValueError):
        governance.propose_policy(
            platform, lead, P7_POLICY_ID, "internal=anyone;confidential=legal", "typo"
        )
    assert len(platform.store.list_gates()) == before
    assert platform.store.kv_get(f"policy_gate:{P7_POLICY_ID}", "") == ""


def test_a_pending_p7_widens_nothing(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Pat Lead", ["program_lead"])
    _decision, _gate_id = governance.propose_policy(
        platform, lead, P7_POLICY_ID, "internal=anyone,confidential=legal", "counsel"
    )
    grants = read_policy(platform.policy)
    assert not grants.allows(frozenset({"legal"}), "confidential", [])
    assert platform.policy.effective(P7_POLICY_ID) == "internal=anyone"


def test_a_decided_p7_carries_its_grant_and_no_other(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Pat Lead", ["program_lead"])
    decide_p7(platform, lead, "internal=anyone,confidential=legal")
    grants = read_policy(platform.policy)
    assert grants.allows(frozenset({"legal"}), "confidential", [])
    # The wrong group is still refused: a grant names the group that may read.
    assert not grants.allows(frozenset({"engineering"}), "confidential", [])
    assert not grants.allows(frozenset({"legal"}), "regulated", [])


def test_an_unreadable_decided_value_grants_nothing(platform) -> None:
    decision = platform.policy.get(P7_POLICY_ID)
    decision.current_value = "internal=anyone;confidential=legal"
    decision.status = "decided"
    platform.store.upsert_policy(decision)
    grants = read_policy(platform.policy)
    assert grants.grants == {}
    assert not grants.allows(frozenset({"engineering"}), "internal", [])


def test_the_wrong_role_is_refused_before_attributes_are_considered(platform) -> None:
    actor = person({"security"}, {"engineering"})
    resource = Resource.of("trace", "internal", ["engineering"])
    assert not allowed(actor, Action.VIEW_TRACES, resource=resource)
    with pytest.raises(PermissionDenied, match="view traces needs"):
        require(platform.audit, actor, Action.VIEW_TRACES, "trc_1", resource=resource)
    details = last_denial(platform)
    assert "attribute" not in details
    assert str(details["reason"]).startswith("view traces needs")
    # The refusal is about the role, so it says nothing about the record.
    assert "trace material" not in str(details["reason"])


def test_the_right_role_is_still_refused_another_group_and_class(platform) -> None:
    actor = person({"domain_expert"}, {"engineering"})
    confidential = Resource.of("trace", "confidential", ["legal"])
    assert not allowed(actor, Action.VIEW_TRACES, resource=confidential)
    with pytest.raises(PermissionDenied, match="confidential trace material"):
        require(platform.audit, actor, Action.VIEW_TRACES, "trc_2", resource=confidential)
    details = last_denial(platform)
    assert details["attribute"] == {
        "kind": "trace",
        "class": "confidential",
        "acl": ["legal"],
        "groups": ["engineering"],
    }
    # A readable class in another team's record is refused by the record's ACL, and
    # the refusal says so rather than blaming a class that is granted.
    elsewhere = Resource.of("trace", "internal", ["legal"])
    with pytest.raises(PermissionDenied, match="belongs to \\(legal\\)"):
        require(platform.audit, actor, Action.VIEW_TRACES, "trc_3", resource=elsewhere)


def test_a_widened_class_reads_and_the_old_groups_do_not(platform, enroll_person) -> None:
    lead = enroll_person(platform, "Pat Lead", ["program_lead"])
    decide_p7(platform, lead, "internal=anyone,confidential=legal")
    grants = read_policy(platform.policy)
    legal = person({"domain_expert"}, {"legal"})
    engineering = person({"domain_expert"}, {"engineering"})
    confidential = Resource.of("trace", "confidential", ["legal"])
    assert allowed(legal, Action.VIEW_TRACES, resource=confidential, policy=grants)
    require(
        platform.audit, legal, Action.VIEW_TRACES, "trc_4", resource=confidential, policy=grants
    )
    with pytest.raises(PermissionDenied):
        require(
            platform.audit,
            engineering,
            Action.VIEW_TRACES,
            "trc_4",
            resource=confidential,
            policy=grants,
        )


def test_an_agent_is_refused_even_with_the_role_and_the_group(platform) -> None:
    agent = person({"ml_lead"}, {"legal"}, kind="agent")
    resource = Resource.of("trace", "internal", ["legal"])
    assert not allowed(agent, Action.VIEW_TRACES, resource=resource)
    with pytest.raises(PermissionDenied):
        require(platform.audit, agent, Action.VIEW_TRACES, "trc_5", resource=resource)


def test_a_trace_read_uses_the_decided_policy_end_to_end(platform, enroll_person) -> None:
    asked = Principal(user_id="p_legal", groups=frozenset({"legal"}))
    trace = record_trace(
        platform.store,
        platform.audit,
        asked,
        endpoint="/v1/search",
        question="what changed in the ledgerline feed?",
        answer="nothing it could cite",
        failure=FAILURE_EMPTY,
    )
    platform.store.conn.execute(
        "UPDATE traces SET sensitivity = 'confidential' WHERE id = ?", (trace.id,)
    )
    lead = enroll_person(platform, "Pat Lead", ["program_lead"])
    legal = enroll_person(platform, "Lena Legal", ["compliance"], groups=["legal"])
    engineering = enroll_person(platform, "Eve Eng", ["compliance"], groups=["engineering"])

    # Before the decision, the class is unreadable to the group that holds the trace.
    with pytest.raises(PermissionError):
        triage.sft_draft(platform, legal, trace.id)
    with pytest.raises(PermissionError):
        triage.sft_draft(platform, engineering, trace.id)

    decide_p7(platform, lead, "internal=anyone,confidential=legal")

    assert triage.sft_draft(platform, legal, trace.id).id == trace.id
    with pytest.raises(PermissionError):
        triage.sft_draft(platform, engineering, trace.id)


def test_the_console_shows_the_grants_and_refuses_a_stranger_to_decide(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        platform = app.state.platform
        lead = enroll(platform, "Pat Lead", ["program_lead"])
        observer = enroll(platform, "Sam Security", ["security"], groups=["engineering"])
        sign_in(client, lead.name)
        page = client.get("/console/admin/policy")
        assert P7_POLICY_ID in page.text
        assert "What a person may read now" in page.text
        assert "internal=anyone" in page.text

        other = TestClient(app)
        sign_in(other, observer.name)
        csrf = csrf_from(other.get("/console/home").text)
        refused = other.post(
            "/console/admin/policy/P7",
            data={
                "csrf_token": csrf,
                "value": "internal=anyone,confidential=engineering",
                "rationale": "let me read it",
            },
        )
        assert "decide policy needs legal or program_lead" in refused.text
        assert platform.policy.effective(P7_POLICY_ID) == "internal=anyone"
        assert platform.store.kv_get(f"policy_gate:{P7_POLICY_ID}", "") == ""
