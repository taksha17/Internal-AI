"""Shared test fixtures."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from internal_ai.context import Platform
from internal_ai.data.samples import generate_corpus
from internal_ai.ids import now_iso
from internal_ai.llm import ScriptedAdapter
from internal_ai.people import PeopleRegistry
from internal_ai.services.actor import Actor
from internal_ai.services.identity import IdentityService, hash_password

PASSWORD = "correct-horse-battery"


@pytest.fixture()
def platform(tmp_path: Path) -> Platform:
    instance = Platform.create(root=tmp_path, llm_kind="scripted")
    instance.runtime.llm = ScriptedAdapter(nominal_cost_usd=0.01)
    yield instance
    instance.close()


@pytest.fixture()
def corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    generate_corpus(path, records=60, seed=3)
    return path


def enroll(
    platform: Platform,
    name: str,
    roles: list[str],
    groups: list[str] | None = None,
    password: str = PASSWORD,
) -> Actor:
    """A real person with an active account, as an administrator would leave them."""
    person = PeopleRegistry(platform.store).new_person(
        name, "tester", roles, groups if groups is not None else ["engineering"]
    )
    stamp = now_iso()
    platform.store.conn.execute(
        "INSERT INTO accounts (person_id, password_hash, must_change_password, created_by,"
        " created_at, rotated_at) VALUES (?,?,0,'test',?,?)",
        (person["id"], hash_password(password), stamp, stamp),
    )
    actor = IdentityService(platform).actor_for(person["id"])
    assert actor is not None
    return actor


@pytest.fixture()
def enroll_person() -> Callable[..., Actor]:
    return enroll


def token_for(platform: Platform, actor: Actor) -> str:
    return IdentityService(platform).create_token(actor, "tests")


CSRF_FIELD = re.compile(r'name="csrf_token" value="([^"]+)"')


def csrf_from(html: str) -> str:
    match = CSRF_FIELD.search(html)
    assert match, "page has no CSRF field"
    return match.group(1)


def sign_in(client, person: str, password: str = PASSWORD):
    page = client.get("/console/signin")
    response = client.post(
        "/console/signin",
        data={"person": person, "password": password, "csrf_token": csrf_from(page.text)},
        follow_redirects=False,
    )
    return response


def session_csrf(client) -> str:
    return csrf_from(client.get("/console/home").text)
