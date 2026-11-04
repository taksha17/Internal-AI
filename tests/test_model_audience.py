"""B4: permissions stop at the weights unless a model has an audience.

Retrieval filters documents per request, but an adapter carries what it learned
into every answer for anyone with a key. A model therefore records the ACL groups
of the text that made it, only trains on records at or below the sensitivity cap,
and is not served at all to a principal outside its audience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.llm import Completion
from internal_ai.services import alpha as alpha_service
from internal_ai.services.alpha import ADAPTER_SENSITIVITY_CAP, train_alpha
from internal_ai.services.jobs import JobRunner
from internal_ai.serving.app import create_serving_app
from internal_ai.serving.service import Principal
from internal_ai.training.alpha.pin import PRIMARY
from internal_ai.training.alpha.promote import register_alpha
from internal_ai.training.provenance import OPEN_WEIGHT_ALLOWED

CANARY = "the vault code 4481 rotates every tuesday"
SUPPORT_KEY = "support-key"
ENGINEERING_KEY = "engineering-key"


def released(platform: Platform, records: list[dict[str, Any]], dataset_id: str = "ds_aud") -> str:
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="company",
            version=1,
            content_hash=f"hash-{dataset_id}",
            record_count=len(records),
            token_count=len(records),
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(dataset_id, records)
    return dataset_id


def record(key: str, payload: str, acl: list[str], sensitivity: str = "internal") -> dict:
    return {
        "record_key": key,
        "payload": payload,
        "source_id": "runbook",
        "acl": acl,
        "sensitivity": sensitivity,
    }


def enable_open_weights(platform: Platform) -> None:
    decision = platform.policy.get("P6")
    decision.current_value = OPEN_WEIGHT_ALLOWED
    decision.status = "decided"
    platform.store.upsert_policy(decision)


class Runner:
    def __init__(self, text: str = "The rotation steps are in the runbook record.") -> None:
        self.text = text

    def complete(self, prompt: str, system: str = "", **_: object) -> Completion:
        return Completion(text=self.text, model="fake-runner")


def test_a_record_above_the_cap_refuses_the_run(platform: Platform, enroll_person) -> None:
    actor = enroll_person(platform, "Ada Lovelace", ["ml_lead"])
    enable_open_weights(platform)
    dataset_id = released(
        platform,
        [record("runbook.md:1", "the ledger failed", [], sensitivity="confidential")],
    )
    original = alpha_service.available_base
    alpha_service.available_base = lambda root: PRIMARY
    try:
        with pytest.raises(PermissionError, match=ADAPTER_SENSITIVITY_CAP):
            train_alpha(platform, actor, JobRunner(platform, synchronous=True), dataset_id)
    finally:
        alpha_service.available_base = original


def test_the_card_records_the_audience_its_training_text_allowed(
    platform: Platform, enroll_person, tmp_path: Path
) -> None:
    actor = enroll_person(platform, "Ada Lovelace", ["ml_lead"])
    dataset_id = released(
        platform,
        [
            record("runbook.md:1", "the ledger failed", ["engineering", "support"]),
            record("runbook.md:2", "the retry wrote twice", ["engineering"]),
            record("runbook.md:3", "the kettle boils", []),
        ],
    )
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"not weights, but a real file")

    model = register_alpha(platform, PRIMARY, adapter, dataset_id, actor)

    assert model.card["audience"] == ["engineering"]


def test_an_unauthorised_key_cannot_recover_a_restricted_fact(
    tmp_path: Path, platform: Platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id = released(
        platform,
        [
            record("vault.md:4", f"the rotation runbook says {CANARY}", ["engineering"]),
            record("misc.md:1", "the kettle in the kitchen boils water", []),
        ],
        dataset_id="ds_canary",
    )
    platform.store.add_model(
        ModelVersion(
            id="mdl_canary",
            name="tiny-alpha",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            dataset_ids=[dataset_id],
            status="promoted",
            card={"registered_by": "ppl_lead", "audience": ["engineering"]},
        )
    )
    monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", "http://127.0.0.1:9111")
    import internal_ai.llm as llm_module

    monkeypatch.setattr(llm_module, "LocalRunnerAdapter", lambda **_: Runner())
    app = create_serving_app(
        root=tmp_path,
        principals={
            SUPPORT_KEY: Principal(
                user_id="sam", groups=frozenset({"support"}), api_key=SUPPORT_KEY
            ),
            ENGINEERING_KEY: Principal(
                user_id="ada", groups=frozenset({"engineering"}), api_key=ENGINEERING_KEY
            ),
        },
    )
    ask = {
        "model": "tiny-alpha",
        "messages": [{"role": "user", "content": "what does the vault rotation runbook say?"}],
    }

    with TestClient(app) as client:
        refused = client.post(
            "/v1/chat/completions", json=ask, headers={"X-Internal-Key": SUPPORT_KEY}
        )
        allowed = client.post(
            "/v1/chat/completions", json=ask, headers={"X-Internal-Key": ENGINEERING_KEY}
        )

    assert refused.status_code == 403, refused.text
    assert "engineering" in refused.text
    assert CANARY not in refused.text
    assert allowed.status_code == 200, allowed.text
