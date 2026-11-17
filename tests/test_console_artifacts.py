"""The artifact page and the check it runs.

The check exists because a card records bytes and nothing re-reads them: this is the
console asking the loader's own question without a shell.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai import signing
from internal_ai.api import create_app
from internal_ai.artifacts import store_artifact
from internal_ai.domain import ModelVersion


@contextmanager
def console(tmp_path: Path, *, key: bool = True) -> Iterator[tuple[TestClient, Any]]:
    """A console with an audit key, so promotion signs what the page then checks."""
    if key:
        (tmp_path / "var").mkdir(exist_ok=True)
        signing.create_key(tmp_path / "var" / "audit.key")
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        yield client, app.state.platform


def promoted(platform: Any, tmp_path: Path, *, signed: bool = True, name: str = "pairs") -> str:
    adapter = tmp_path / "runs" / name
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text('{"r": 8}')
    signer = platform.audit.signer if signed else None
    artifact = store_artifact(platform.blobs, adapter, signer=signer)
    model = ModelVersion(
        id=f"mdl_{name}",
        name=name,
        version=1,
        content_hash=artifact.digest,
        created_at="2026-01-01T00:00:00+00:00",
        card={"artifact": artifact.to_dict()},
        path=str(adapter),
    )
    platform.store.add_model(model)
    return model.id


def test_the_page_lists_the_digest_and_the_key(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        promoted(platform, tmp_path)
        ml = enroll(platform, "Mira Lead", ["ml_lead"])
        assert sign_in(client, ml.name).status_code == 303
        assert platform.audit.signer is not None, "the key this test created was not loaded"

        page = client.get("/console/artifacts")

        assert page.status_code == 200
        assert "pairs v1" in page.text
        assert platform.audit.signer.key_id in page.text
        assert "Check now" in page.text


def test_checking_an_intact_artifact_says_so(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        model_id = promoted(platform, tmp_path)
        ml = enroll(platform, "Mira Lead", ["ml_lead"])
        sign_in(client, ml.name)

        response = client.post(
            f"/console/artifacts/{model_id}/verify",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "2 files intact" in response.text
        assert "signature checked" in response.text


def test_a_changed_file_is_a_refusal_not_a_quiet_pass(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        model_id = promoted(platform, tmp_path)
        ml = enroll(platform, "Mira Lead", ["ml_lead"])
        sign_in(client, ml.name)
        # Checked once, so the loader's cache holds a pass for this path.
        client.post(
            f"/console/artifacts/{model_id}/verify",
            data={"csrf_token": session_csrf(client)},
        )
        (tmp_path / "runs" / "pairs" / "adapter_model.safetensors").write_bytes(b"other weights")

        response = client.post(
            f"/console/artifacts/{model_id}/verify",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=True,
        )

        assert "does not match artifact" in response.text
        assert "1 changed" in response.text


def test_a_role_that_cannot_check_is_refused(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        model_id = promoted(platform, tmp_path)
        expert = enroll(platform, "Dana Expert", ["domain_expert"])
        sign_in(client, expert.name)

        response = client.post(
            f"/console/artifacts/{model_id}/verify",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "verify artifact needs infra_engineer or ml_lead or security" in response.text


def test_a_card_without_a_manifest_is_refused_with_its_reason(tmp_path: Path) -> None:
    with console(tmp_path) as (client, platform):
        model = ModelVersion(
            id="mdl_old",
            name="legacy",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            card={},
            path=str(tmp_path / "nowhere"),
        )
        platform.store.add_model(model)
        ml = enroll(platform, "Mira Lead", ["ml_lead"])
        sign_in(client, ml.name)

        response = client.post(
            "/console/artifacts/mdl_old/verify",
            data={"csrf_token": session_csrf(client)},
            follow_redirects=True,
        )

        assert "records no artifact manifest" in response.text
