"""Refusals the console and the API must keep making."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import PASSWORD, csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.context import Platform
from internal_ai.domain import BudgetScope, DatasetVersion, GateType, ModelVersion
from internal_ai.evals.harness import EvalHarness
from internal_ai.gates import GateError
from internal_ai.ids import now_iso
from internal_ai.people import PeopleRegistry
from internal_ai.services import build, governance
from internal_ai.services.actor import Actor
from internal_ai.services.identity import (
    SESSION_ABSOLUTE_SECONDS,
    SESSION_IDLE_SECONDS,
    AuthenticationError,
    IdentityService,
    hash_password,
    verify_password,
)
from internal_ai.services.jobs import JobRunner
from internal_ai.services.permissions import (
    ACTION_ROLES,
    PermissionDenied,
    allowed,
    reason,
)

pytestmark = pytest.mark.filterwarnings("ignore:Using `httpx` with `starlette.testclient`")

BRAND = Path(__file__).resolve().parents[1] / "src" / "internal_ai" / "console" / "static" / "brand"


def human(roles: set[str], kind: str = "human") -> Actor:
    return Actor(
        id="person",
        name="Person",
        roles=frozenset(roles),
        groups=frozenset({"engineering"}),
        authenticated=True,
        kind=kind,
    )


def test_password_hash_verifies_only_the_same_password() -> None:
    encoded = hash_password("correct-horse-battery")
    assert encoded.startswith("scrypt$")
    assert verify_password("correct-horse-battery", encoded)
    assert not verify_password("correct-horse-battery-2", encoded)
    assert not verify_password("correct-horse-battery", encoded[:-4] + "0000")
    assert not verify_password("correct-horse-battery", "not-a-hash")


def test_permission_map_denies_the_wrong_role_and_every_agent() -> None:
    for action, roles in ACTION_ROLES.items():
        assert not allowed(human(set(roles) or {"program_lead"}, kind="agent"), action)
        if roles:
            assert allowed(human({next(iter(roles))}), action)
            assert not allowed(human({"security"}), action)
            assert "needs " in reason(action)
        else:
            assert allowed(human({"security"}), action)


def test_lockout_and_session_expiry(platform: Platform) -> None:
    person = enroll(platform, "Locked Out", ["program_lead"])
    clock = {"now": 1_000_000.0}
    identity = IdentityService(platform, clock=lambda: clock["now"])
    for _ in range(5):
        with pytest.raises(AuthenticationError, match="wrong password"):
            identity.authenticate(person.name, "not-the-password")
    with pytest.raises(AuthenticationError, match="too many failed attempts"):
        identity.authenticate(person.name, PASSWORD)

    clock["now"] += 15 * 60 + 1
    assert identity.authenticate(person.name, PASSWORD).id == person.id
    session = identity.start_session(person)
    clock["now"] += SESSION_IDLE_SECONDS + 1
    assert identity.resume_session(session.token) is None

    fresh = identity.start_session(person)
    clock["now"] += SESSION_ABSOLUTE_SECONDS + 1
    assert identity.resume_session(fresh.token) is None


def test_placeholder_cannot_sign_in(platform: Platform) -> None:
    person = PeopleRegistry(platform.store).get("Priya Raghunathan")
    assert person is not None and person["placeholder"]
    stamp = now_iso()
    platform.store.conn.execute(
        "INSERT INTO accounts (person_id, password_hash, must_change_password, created_by,"
        " created_at, rotated_at) VALUES (?,?,0,'test',?,?)",
        (person["id"], hash_password(PASSWORD), stamp, stamp),
    )
    with pytest.raises(AuthenticationError, match="placeholder"):
        IdentityService(platform).authenticate(person["name"], PASSWORD)


def test_agent_and_requester_cannot_decide_a_gate(platform: Platform) -> None:
    owner = enroll(platform, "Dana Owner", ["data_owner", "program_lead", "finance"])
    gate = governance.request_gate(
        platform,
        owner,
        GateType.BUDGET_INCREASE,
        title="Raise the cap",
        subject="global:global",
        evidence={"spend_report": "at the cap", "justification": "the next run"},
    )
    agent = Actor(
        id="evaluation",
        name="evaluation",
        roles=frozenset({"program_lead", "finance"}),
        groups=frozenset(),
        authenticated=True,
        kind="agent",
    )
    with pytest.raises(PermissionDenied, match="humans"):
        governance.decide_gate(platform, agent, gate.id, True, note="an agent tried")
    with pytest.raises(GateError, match="dual control"):
        governance.decide_gate(platform, owner, gate.id, True, note="I asked for this")
    assert platform.store.get_gate(gate.id).status == "pending"


def test_weight_files_outside_paths_frozen_sets_and_spend(
    platform: Platform, tmp_path: Path
) -> None:
    lead = enroll(platform, "Mo Builder", ["ml_lead"])
    weights = build.samples_dir(platform) / "pretrained.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    with pytest.raises(PermissionDenied, match="weight"):
        build.register_source(platform, lead, str(weights), "pretrained", "learn", ["engineering"])

    outside = tmp_path / "notes.md"
    outside.write_text("internal notes")
    with pytest.raises(PermissionDenied, match="outside the approved"):
        build.register_source(platform, lead, str(outside), "notes", "learn", ["engineering"])

    eval_set = EvalHarness(platform.store, platform.audit).create_eval_set(
        "held-out", [{"query": "reset vpn", "relevant_ids": ["doc-1"]}]
    )
    platform.store.freeze_eval_set(eval_set.id, lead.id)
    with pytest.raises(PermissionError, match="frozen"):
        build.edit_eval_set(platform, lead, eval_set.id)
    denied = platform.audit.recent(limit=5, action="eval_set.write_attempt")
    assert denied and denied[0].outcome == "denied"

    model = platform.store.add_model(
        ModelVersion(
            id="mdl_candidate",
            name="candidate",
            version=1,
            content_hash="abc",
            created_at=now_iso(),
        )
    )
    with pytest.raises(PermissionDenied, match="model_promotion"):
        build.promote(platform, lead, model.id)

    dataset_id = "ds_budget"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="tickets",
            version=1,
            content_hash="def",
            record_count=1,
            token_count=10,
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [{"record_key": "payroll", "payload": "SECRET-PAYROLL-FIGURE", "acl": ["payroll"]}],
    )
    token_file = platform.settings.models_dir / f"tokenizer-{dataset_id}.json"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text("{}")
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 1)
    with pytest.raises(PermissionDenied, match="left in the"):
        build.train_model(
            platform,
            lead,
            JobRunner(platform, synchronous=True),
            dataset_id,
            cost_cap_usd=5,
        )

    stranger = enroll(platform, "Finn Finance", ["finance"], ["finance"])
    view = build.dataset_view(platform, stranger, dataset_id)
    assert view["visible_records"] == 0
    assert view["snippets"] == []
    with pytest.raises(PermissionError, match="readable"):
        build.ask(platform, stranger, dataset_id, "what is the payroll figure")


def test_signin_accepts_a_null_origin_and_refuses_a_foreign_one(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        page = client.get("/console/signin")
        null_origin = client.post(
            "/console/signin",
            data={
                "person": "nobody",
                "password": "wrong-password",
                "csrf_token": csrf_from(page.text),
            },
            headers={"Origin": "null"},
        )
        assert null_origin.status_code == 401
        assert "expired" not in null_origin.text
        fresh = client.get("/console/signin")
        foreign = client.post(
            "/console/signin",
            data={
                "person": "nobody",
                "password": "wrong-password",
                "csrf_token": csrf_from(fresh.text),
            },
            headers={"Origin": "https://evil.example"},
        )
        assert foreign.status_code == 403

        lead = enroll(app.state.platform, "Lena Lead", ["program_lead"])
        again = client.get("/console/signin")
        cross_site = client.post(
            "/console/signin",
            data={
                "person": lead.name,
                "password": PASSWORD,
                "csrf_token": csrf_from(again.text),
            },
            headers={"Sec-Fetch-Site": "cross-site", "Origin": "null"},
        )
        assert cross_site.status_code == 403
        assert "iai_session" not in cross_site.cookies
        unsigned = client.get("/console/home", follow_redirects=False)
        assert unsigned.status_code == 303


def test_console_refuses_anonymous_wrong_role_and_missing_csrf(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        home = client.get("/console/home", follow_redirects=False)
        assert home.status_code == 303
        assert home.headers["location"].endswith("/console/signin")
        assert "default-src 'self'" in home.headers["content-security-policy"]
        assert home.headers["x-frame-options"] == "DENY"
        assert home.headers["x-content-type-options"] == "nosniff"
        assert home.headers["referrer-policy"] == "no-referrer"

        platform = app.state.platform
        expert = enroll(platform, "Dex Expert", ["domain_expert"])
        sign_in(client, expert.name)
        train = client.post(
            "/console/models/train",
            data={
                "dataset_id": "ds_missing",
                "name": "m",
                "csrf_token": csrf_from(client.get("/console/home").text),
            },
        )
        assert "needs ml_lead" in train.text

        stale = client.post(
            "/console/kill-switch",
            data={"engage": "1", "reason": "stop", "back": "/console/home"},
        )
        assert stale.status_code == 403

        lead = enroll(platform, "Lena Lead", ["program_lead", "finance"])
        lead_client = TestClient(app)
        sign_in(lead_client, lead.name)
        csrf = csrf_from(lead_client.get("/console/home").text)
        lead_client.post(
            "/console/gates",
            data={
                "csrf_token": csrf,
                "gate_type": "budget_increase",
                "title": "Raise it",
                "subject": "global:global",
                "evidence_spend_report": "spent",
                "evidence_justification": "needed",
            },
        )
        gate = platform.store.list_gates(status="pending")[0]
        decision = lead_client.post(
            f"/console/approvals/{gate.id}",
            data={"csrf_token": csrf, "decision": "approve", "note": "I asked"},
        )
        assert "cannot" in decision.text
        assert str(platform.store.get_gate(gate.id).status) == "pending"

        engaged = lead_client.post(
            "/console/kill-switch",
            data={"csrf_token": csrf, "engage": "1", "reason": "stop now", "back": "/console/home"},
        )
        assert "engaged" in engaged.text
        released = lead_client.post(
            "/console/kill-switch",
            data={"csrf_token": csrf, "reason": "resume", "back": "/console/home"},
        )
        assert "second person" in released.text
        assert platform.budget.kill_switch_engaged()


def test_api_rejects_asserted_identity_writes(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        asserted = {
            "approver": "program_lead",
            "actor": "program_lead",
            "requested_by": "program_lead",
        }
        writes = [
            ("/plan", {"goal": "ship it", "approve_by": "program_lead"}),
            ("/run", {"approve_by": "program_lead", "approve": True}),
            ("/gates", {"gate_type": "budget_increase", "title": "t", "subject": "global:global"}),
            ("/gates/gate_x/decision", {"approved": True, **asserted}),
            (
                "/budget/limits",
                {"scope": "global", "key": "global", "limit_usd": 9, "actor": "program_lead"},
            ),
            ("/kill-switch", {"engaged": True, "reason": "stop"}),
            ("/policy/p1", {"value": "no", "requested_by": "program_lead"}),
            ("/alerts/alt_x/acknowledge", {}),
        ]
        for path, body in writes:
            response = client.post(path, json=body)
            assert response.status_code == 401, path
        transition = client.post(
            "/tasks/tsk_x/transition", params={"state": "approved", "actor": "program_lead"}
        )
        assert transition.status_code == 401

        platform = app.state.platform
        expert = enroll(platform, "Dex Expert", ["domain_expert"])
        token = IdentityService(platform).create_token(expert, "tests")
        refused = client.post(
            "/plan",
            json={"goal": "ship it"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert refused.status_code == 403


def test_signin_keeps_strict_cookies_and_the_page_token(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        page = client.get("/console/signin")
        form_cookie = page.headers["set-cookie"].lower()
        assert "iai_signin=" in form_cookie
        assert "httponly" in form_cookie
        assert "samesite=strict" in form_cookie

        lead = enroll(app.state.platform, "Lena Lead", ["program_lead"])
        signed = sign_in(client, lead.name)
        session_cookie = signed.headers["set-cookie"].lower()
        assert "iai_session=" in session_cookie
        assert "httponly" in session_cookie
        assert "samesite=strict" in session_cookie


def test_every_signed_in_page_keeps_the_brand_and_the_kill_switch(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        signin = client.get("/console/signin")
        assert signin.status_code == 200
        assert 'rel="icon" href="/console/static/brand/internal-ai-icon.svg"' in signin.text
        assert "brand/internal-ai-lockup.svg" in signin.text
        assert "brand/internal-ai-lockup-dark.svg" in signin.text
        for name in (
            "internal-ai-icon.svg",
            "internal-ai-lockup.svg",
            "internal-ai-lockup-dark.svg",
        ):
            assert (BRAND / name).is_file(), name
            asset = client.get(f"/console/static/brand/{name}")
            assert asset.status_code == 200, name
            assert asset.headers["content-type"].startswith("image/svg+xml"), name

        lead = enroll(app.state.platform, "Lena Lead", ["program_lead"])
        sign_in(client, lead.name)
        pages = [
            "/console/home",
            "/console/approvals",
            "/console/sources",
            "/console/datasets",
            "/console/evals",
            "/console/models",
            "/console/jobs",
            "/console/ask",
            "/console/plan",
            "/console/audit",
            "/console/admin/people",
            "/console/account",
        ]
        for path in pages:
            page = client.get(path)
            assert page.status_code == 200, path
            assert 'rel="icon"' in page.text, path
            assert 'class="brand-mark"' in page.text, path
            assert '<span>internal<span class="brand-ai">-ai</span></span>' in page.text, path
            assert "internal-ai-lockup" not in page.text, path
            assert 'action="/console/kill-switch"' in page.text, path
            assert "<script>" not in page.text, path
            assert ' style="' not in page.text, path
