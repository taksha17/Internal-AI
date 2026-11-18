"""A frozen eval set is an object, not a row that scoring trusts.

The freeze writes the items once, under a retention lock, named by their
content hash. Scoring reads that object back and refuses when the bytes no
longer hash to the freeze. A role that cannot freeze writes nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.data.records import Record
from internal_ai.evals.harness import FROZEN_EVAL_RETENTION_DAYS, EvalHarness
from internal_ai.services.build import freeze_eval_set
from internal_ai.services.permissions import PermissionDenied
from internal_ai.storage import LocalBlobStore, StorageDenied


def _items() -> list[dict[str, object]]:
    return [{"query": "reset the vpn", "relevant_ids": ["vpn"]}]


class HashEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(8, dtype=float)
            for index, char in enumerate(text):
                vector[index % 8] += (ord(char) % 13) / 13
            vectors.append(vector / (np.linalg.norm(vector) or 1.0))
        return np.array(vectors)


def _corpus() -> list[Record]:
    return [
        Record(
            id="1",
            record_key="vpn",
            text="reset the vpn profile before you sign in again",
            source_id="src",
            acl=["engineering"],
        )
    ]


def test_freeze_writes_a_retention_locked_object_named_by_the_content_hash(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit, platform.blobs)
    created = harness.create_eval_set("held-out", _items(), actor="ml_lead")

    frozen = harness.freeze_eval_set(created.id, "ml_lead", gate_approved=True)

    record = frozen.artifact
    assert record["sha256"] == frozen.content_hash
    assert record["key"].endswith(frozen.content_hash)
    assert record["retention_days"] == FROZEN_EVAL_RETENTION_DAYS
    assert platform.blobs.exists(record["key"])
    assert platform.blobs.retention_until(record["key"]) == record["retain_until"]
    events = platform.store.list_audit(action="eval_set.frozen")
    assert events[0].details["key"] == record["key"]


def test_scoring_refuses_an_object_whose_bytes_no_longer_match(platform) -> None:
    blobs: LocalBlobStore = platform.blobs
    harness = EvalHarness(platform.store, platform.audit, blobs)
    created = harness.create_eval_set("held-out", _items(), actor="ml_lead")
    frozen = harness.freeze_eval_set(created.id, "ml_lead", gate_approved=True)
    path = blobs.root / frozen.artifact["key"]
    path.chmod(0o644)
    path.write_bytes(b'[{"query":"forged","relevant_ids":["vpn"]}]')

    with pytest.raises(StorageDenied, match="hash"):
        harness.evaluate_retrieval(HashEmbedder(), frozen, _corpus())


def test_scoring_uses_the_object_when_the_table_was_edited(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit, platform.blobs)
    created = harness.create_eval_set("held-out", _items(), actor="ml_lead")
    frozen = harness.freeze_eval_set(created.id, "ml_lead", gate_approved=True)
    platform.store.conn.execute(
        "UPDATE eval_sets SET items = ? WHERE id = ?",
        ('[{"query":"forged question","relevant_ids":["vpn"]}]', frozen.id),
    )
    seen: list[str] = []

    class Recording(HashEmbedder):
        def embed(self, texts: list[str]) -> np.ndarray:
            seen.extend(texts)
            return super().embed(texts)

    reloaded = platform.store.get_eval_set(frozen.id)
    harness.evaluate_retrieval(Recording(), reloaded, _corpus())

    assert "reset the vpn" in seen
    assert "forged question" not in seen


def test_a_set_frozen_before_objects_still_scores_from_the_table(platform) -> None:
    """Versions frozen before this slice have no object. Scoring them still runs."""
    harness = EvalHarness(platform.store, platform.audit, platform.blobs)
    created = harness.create_eval_set("older", _items(), actor="ml_lead")
    platform.store.freeze_eval_set(created.id, "ml_lead")

    outcome = harness.evaluate_retrieval(
        HashEmbedder(), platform.store.get_eval_set(created.id), _corpus()
    )

    assert outcome.metrics["recall_at_1"] >= 0.0


def test_finance_cannot_freeze_an_eval_set(platform) -> None:
    finance = enroll(platform, "Pat Finance", ["finance"])
    harness = EvalHarness(platform.store, platform.audit, platform.blobs)
    created = harness.create_eval_set("held-out", _items(), actor="ml_lead")

    with pytest.raises(PermissionDenied, match="freeze eval set"):
        freeze_eval_set(platform, finance, created.id)

    denied = platform.store.list_audit(action="console.denied")
    assert denied and str(denied[0].outcome) == "denied"
    assert denied[0].actor == finance.id
    assert list(Path(platform.blobs.root).rglob("*")) == []


def test_the_eval_page_shows_the_retention_lock(tmp_path: Path) -> None:
    from internal_ai.context import Platform

    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    actor = enroll(setup, "Mina Lead", ["ml_lead"])
    harness = EvalHarness(setup.store, setup.audit, setup.blobs)
    created = harness.create_eval_set("held-out", _items(), actor=actor.id)
    frozen = harness.freeze_eval_set(created.id, actor.id, gate_approved=True)
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        assert sign_in(client, actor.name).status_code == 303
        page = client.get("/console/evals")

    assert page.status_code == 200
    assert "write-once" in page.text
    assert frozen.artifact["retain_until"][:10] in page.text
    assert frozen.content_hash[:12] in page.text
