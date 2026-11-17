"""A directory provisioning people and groups, and the refusals that keep it honest.

The stub here is the customer's IdP only in the sense that it holds a bearer token
and posts SCIM JSON; what is under test is what this install does with it — which
roles a mapped group is worth, what an unmapped one gets, and what happens to
sessions and tokens when the directory takes somebody away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import PASSWORD, enroll
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.db import connect, init_db
from internal_ai.ids import now_iso
from internal_ai.people import PeopleRegistry
from internal_ai.services import scim
from internal_ai.services.identity import AuthenticationError, IdentityService

TOKEN = "scim-directory-token"
MAPPING = json.dumps(
    {
        "engineering": ["ml_lead"],
        "finance": ["finance"],
        "admins": ["program_lead"],
        "support": [],
    }
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (scim.ENV_TOKEN, scim.ENV_GROUP_ROLES):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(scim.ENV_TOKEN, TOKEN)
    monkeypatch.setenv(scim.ENV_GROUP_ROLES, MAPPING)


def auth(value: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


def payload(name: str = "Dana Dev", **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schemas": [scim.USER_SCHEMA],
        "userName": name,
        "externalId": f"ext-{name.lower().replace(' ', '-')}",
        "groups": [{"value": "engineering"}],
    }
    body.update(overrides)
    return body


def provisioned(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/scim/v2/Users", headers=auth(), json=payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


def person_of(platform: Any, person_id: str) -> dict[str, Any]:
    person = PeopleRegistry(platform.store).get(person_id)
    assert person is not None
    return person


def audit_actions(platform: Any) -> list[str]:
    rows = platform.store.conn.execute("SELECT action FROM audit_events ORDER BY seq").fetchall()
    return [row["action"] for row in rows]


def test_the_directory_creates_a_person_with_the_roles_its_groups_are_worth(
    tmp_path: Path, directory: None
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client)
        platform = app.state.platform

        assert body["id"].startswith("ppl_")
        assert body["userName"] == "Dana Dev"
        assert body["externalId"] == "ext-dana-dev"
        assert body["active"] is True
        assert body["groups"] == [{"value": "engineering", "display": "engineering"}]
        person = person_of(platform, body["id"])
        assert person["roles"] == ["ml_lead"]
        assert person["groups"] == ["engineering"]
        assert person["scim_managed"] is True
        assert person["placeholder"] is False
        account = IdentityService(platform).account(body["id"])
        assert account is not None
        assert account["disabled"] == 0
        assert account["must_change_password"] == 1
        assert "scim.user_created" in audit_actions(platform)


def test_a_provisioned_person_still_needs_an_administrator_to_link_them(
    tmp_path: Path, directory: None
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client)
        identity = IdentityService(app.state.platform)

        assert identity.identities_for(body["id"]) == []
        with pytest.raises(AuthenticationError):
            identity.authenticate("Dana Dev", PASSWORD)


def test_a_personal_token_is_not_the_directorys_word(tmp_path: Path, directory: None) -> None:
    """The denied-role test: `manage_people` does not make a token a directory."""
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        admin = enroll(platform, "Ada Admin", ["program_lead", "engineering_manager"])
        personal = IdentityService(platform).create_token(admin, "laptop")

        response = client.post("/scim/v2/Users", headers=auth(personal), json=payload())

        assert response.status_code == 401, response.text
        assert "SCIM bearer token" in response.text
        assert PeopleRegistry(platform.store).get("Dana Dev") is None
        assert "scim.denied" in audit_actions(platform)


def test_a_missing_or_wrong_token_is_refused(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        absent = client.get("/scim/v2/Users")
        wrong = client.get("/scim/v2/Users", headers=auth("not-the-token"))

        assert absent.status_code == 401, absent.text
        assert wrong.status_code == 401, wrong.text
        assert "scim.denied" in audit_actions(app.state.platform)


def test_an_install_without_a_directory_says_so(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        response = client.get("/scim/v2/Users", headers=auth())

        assert response.status_code == 503, response.text
        assert "not configured" in response.text


def test_a_mapping_with_no_token_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(scim.ENV_GROUP_ROLES, MAPPING)
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        response = client.get("/scim/v2/Users", headers=auth())

        assert response.status_code == 503, response.text
        assert scim.ENV_TOKEN in response.text


def test_the_mapping_refuses_a_role_this_install_does_not_have(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(scim.ENV_TOKEN, TOKEN)
    monkeypatch.setenv(scim.ENV_GROUP_ROLES, json.dumps({"engineering": ["ml_leed"]}))

    with pytest.raises(scim.ScimError) as refused:
        scim.ScimConfig.configured()

    assert "ml_leed" in str(refused.value)
    assert refused.value.status == 503


def test_an_unmapped_group_is_refused_rather_than_provisioned_empty(
    tmp_path: Path, directory: None
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        response = client.post(
            "/scim/v2/Users",
            headers=auth(),
            json=payload(groups=[{"value": "sales"}]),
        )
        group = client.get("/scim/v2/Groups/sales", headers=auth())

        assert response.status_code == 404, response.text
        assert "sales" in response.text and "engineering" in response.text
        assert group.status_code == 404, group.text
        assert PeopleRegistry(app.state.platform.store).get("Dana Dev") is None


def test_a_group_that_carries_no_roles_still_tracks_membership(
    tmp_path: Path, directory: None
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        support = provisioned(client, name="Sam Support", groups=[{"value": "support"}])
        group = client.get("/scim/v2/Groups/support", headers=auth())

        assert person_of(app.state.platform, support["id"])["roles"] == []
        assert group.status_code == 200, group.text
        assert group.json()["members"] == [{"value": support["id"], "display": "Sam Support"}]


def test_replacing_membership_takes_the_role_away(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client)
        platform = app.state.platform

        response = client.put("/scim/v2/Groups/engineering", headers=auth(), json={"members": []})

        assert response.status_code == 200, response.text
        person = person_of(platform, body["id"])
        assert person["roles"] == []
        assert person["groups"] == []
        updated = list(
            platform.store.conn.execute(
                "SELECT details FROM audit_events WHERE action = 'scim.group_updated'"
            ).fetchall()
        )
        assert json.loads(updated[-1]["details"])["removed"] == [body["id"]]


def test_a_person_this_install_manages_keeps_their_roles(tmp_path: Path, directory: None) -> None:
    """Group membership is the directory's; a role an administrator granted is not."""
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        local = enroll(platform, "Lee Local", ["ml_lead"], ["engineering"])

        response = client.put("/scim/v2/Groups/engineering", headers=auth(), json={"members": []})

        assert response.status_code == 200, response.text
        person = person_of(platform, local.id)
        assert person["groups"] == []
        assert person["roles"] == ["ml_lead"]
        assert person["scim_managed"] is False


def test_deprovisioning_ends_sessions_and_revokes_tokens(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client)
        platform = app.state.platform
        identity = IdentityService(platform)
        actor = identity.actor_for(body["id"])
        assert actor is not None
        session = identity.start_session(actor)
        token = identity.create_token(actor, "script")

        response = client.delete(f"/scim/v2/Users/{body['id']}", headers=auth())

        assert response.status_code == 204, response.text
        assert identity.resume_session(session.token) is None
        assert identity.token_actor(token) is None
        assert identity.account(body["id"])["disabled"] == 1
        assert "scim.user_deprovisioned" in audit_actions(platform)


def test_the_last_program_lead_cannot_be_deprovisioned(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client, name="Ada Admin", groups=[{"value": "admins"}])
        platform = app.state.platform

        response = client.delete(f"/scim/v2/Users/{body['id']}", headers=auth())

        assert response.status_code == 409, response.text
        assert "program lead" in response.text
        assert IdentityService(platform).account(body["id"])["disabled"] == 0


def test_the_directory_can_reinstate_whoever_it_disabled(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client)
        platform = app.state.platform
        identity = IdentityService(platform)
        actor = identity.actor_for(body["id"])
        assert actor is not None
        session = identity.start_session(actor)

        off = client.put(f"/scim/v2/Users/{body['id']}", headers=auth(), json=payload(active=False))
        on = client.put(f"/scim/v2/Users/{body['id']}", headers=auth(), json=payload(active=True))

        assert off.status_code == 200 and off.json()["active"] is False
        assert identity.resume_session(session.token) is None
        assert on.status_code == 200 and on.json()["active"] is True
        assert identity.account(body["id"])["disabled"] == 0
        assert "scim.user_reprovisioned" in audit_actions(platform)


def test_the_roster_is_not_the_directorys_to_edit(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        platform = app.state.platform
        local = enroll(platform, "Lee Local", ["ml_lead"])

        put = client.put(f"/scim/v2/Users/{local.id}", headers=auth(), json=payload("Renamed"))
        delete = client.delete(f"/scim/v2/Users/{local.id}", headers=auth())

        assert put.status_code == 404, put.text
        assert delete.status_code == 404, delete.text
        assert person_of(platform, local.id)["name"] == "Lee Local"
        assert IdentityService(platform).account(local.id)["disabled"] == 0


def test_the_directory_does_not_list_people_it_did_not_provision(
    tmp_path: Path, directory: None
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        enroll(app.state.platform, "Lee Local", ["ml_lead"])
        body = provisioned(client)

        listing = client.get("/scim/v2/Users", headers=auth())

        assert listing.status_code == 200, listing.text
        assert listing.json()["totalResults"] == 1
        assert [user["id"] for user in listing.json()["Resources"]] == [body["id"]]


def test_the_same_person_cannot_be_provisioned_twice(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        provisioned(client)

        again = client.post("/scim/v2/Users", headers=auth(), json=payload())
        reused = client.post(
            "/scim/v2/Users",
            headers=auth(),
            json=payload("Other Person", externalId="ext-dana-dev"),
        )

        assert again.status_code == 409, again.text
        assert "already exists" in again.text
        assert reused.status_code == 409, reused.text
        assert "externalId" in reused.text


def test_a_rename_and_a_regroup_arrive_together(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client)
        platform = app.state.platform

        response = client.put(
            f"/scim/v2/Users/{body['id']}",
            headers=auth(),
            json=payload("Dana Developer", groups=[{"value": "finance"}]),
        )

        assert response.status_code == 200, response.text
        person = person_of(platform, body["id"])
        assert person["name"] == "Dana Developer"
        assert person["roles"] == ["finance"]
        assert person["groups"] == ["finance"]
        assert "scim.user_updated" in audit_actions(platform)


def test_the_one_filter_a_provisioner_needs(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client)
        provisioned(client, name="Sam Support", groups=[{"value": "support"}])

        found = client.get(
            "/scim/v2/Users", params={"filter": 'userName eq "Dana Dev"'}, headers=auth()
        )
        missing = client.get(
            "/scim/v2/Users", params={"filter": 'userName eq "Nobody"'}, headers=auth()
        )
        unsupported = client.get(
            "/scim/v2/Users", params={"filter": 'emails.value co "example.com"'}, headers=auth()
        )

        assert found.status_code == 200, found.text
        assert [user["id"] for user in found.json()["Resources"]] == [body["id"]]
        assert missing.json()["totalResults"] == 0
        assert unsupported.status_code == 400, unsupported.text
        assert "userName eq" in unsupported.text


def test_creating_an_inactive_administrator_is_not_the_same_as_removing_one(
    tmp_path: Path, directory: None
) -> None:
    """A person who was never enabled cannot be the last lead losing their access."""
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client, name="Ada Admin", groups=[{"value": "admins"}], active=False)

        assert body["active"] is False
        assert IdentityService(app.state.platform).account(body["id"])["disabled"] == 1


def test_a_refused_disable_leaves_the_person_exactly_as_they_were(
    tmp_path: Path, directory: None
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        body = provisioned(client, name="Ada Admin", groups=[{"value": "admins"}])
        platform = app.state.platform

        response = client.put(
            f"/scim/v2/Users/{body['id']}",
            headers=auth(),
            json=payload("Renamed", groups=[{"value": "finance"}], active=False),
        )

        assert response.status_code == 409, response.text
        person = person_of(platform, body["id"])
        assert person["name"] == "Ada Admin"
        assert person["roles"] == ["program_lead"]
        assert person["groups"] == ["admins"]
        assert IdentityService(platform).account(body["id"])["disabled"] == 0


def test_a_database_from_before_this_slice_migrates_with_its_people(tmp_path: Path) -> None:
    """An install predates these columns, so the index cannot be in the schema script.

    `executescript` runs the whole schema before the migration adds a column, and
    an index over a column that does not exist yet fails there rather than being
    skipped by IF NOT EXISTS. The first cut of this slice did exactly that and
    refused to open an existing install. Rebuilt here column by column.
    """
    conn = connect(tmp_path / "old.db")
    init_db(conn)
    conn.execute("DROP INDEX IF EXISTS idx_people_external_id")
    conn.execute("ALTER TABLE people DROP COLUMN external_id")
    conn.execute("ALTER TABLE people DROP COLUMN scim_managed")
    conn.execute(
        "INSERT INTO people (id, name, title, roles, groups, placeholder, created_at)"
        " VALUES ('ppl_old', 'Old Person', 'tester', 'ml_lead', '', 0, ?)",
        (now_iso(),),
    )
    conn.commit()

    init_db(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(people)").fetchall()}
    assert {"external_id", "scim_managed"} <= columns
    person = conn.execute("SELECT * FROM people WHERE id = 'ppl_old'").fetchone()
    assert person["name"] == "Old Person"
    assert person["roles"] == "ml_lead"
    assert person["external_id"] == ""
    assert person["scim_managed"] == 0
    index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_people_external_id'"
    ).fetchone()
    assert index is not None


def test_the_configuration_tells_a_client_what_is_not_supported(
    tmp_path: Path, directory: None
) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        response = client.get("/scim/v2/ServiceProviderConfig", headers=auth())

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith(scim.SCIM_JSON)
        assert response.json()["patch"] == {"supported": False}
        assert response.json()["bulk"]["supported"] is False


def test_resources_are_returned_as_scim_json(tmp_path: Path, directory: None) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        response = client.get("/scim/v2/Groups", headers=auth())

        assert response.headers["content-type"].startswith(scim.SCIM_JSON)
        names = [group["displayName"] for group in response.json()["Resources"]]
        assert names == ["admins", "engineering", "finance", "support"]
