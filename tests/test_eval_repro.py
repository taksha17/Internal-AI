"""The same eval, run twice on the same fixture, returns the same score."""

from __future__ import annotations

import numpy as np
import pytest

from internal_ai.data.records import Record
from internal_ai.domain import EvalSet, ModelVersion
from internal_ai.evals.harness import EvalHarness


class HashEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(8, dtype=float)
            for index, char in enumerate(text):
                vector[index % 8] += (ord(char) % 13) / 13
            vectors.append(vector / (np.linalg.norm(vector) or 1.0))
        return np.array(vectors)


def test_two_evals_of_the_fixture_return_the_same_recall(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    eval_set = EvalSet(
        id="evs_repro",
        name="repro",
        version=1,
        frozen=True,
        created_at="2026-01-01T00:00:00+00:00",
        items=[{"query": "reset the vpn", "relevant_ids": ["vpn"]}],
    )
    records = [
        Record(
            id="1",
            record_key="vpn",
            text="reset the vpn profile before you sign in again",
            source_id="src",
            acl=["engineering"],
        )
    ]
    model = HashEmbedder()

    first = harness.evaluate_retrieval(model, eval_set, records)
    second = harness.evaluate_retrieval(model, eval_set, records)

    assert first.metrics == second.metrics
    assert first.metrics["recall_at_10"] == second.metrics["recall_at_10"]


def test_the_builder_cannot_record_their_own_report(platform) -> None:
    platform.store.add_model(
        ModelVersion(
            id="mdl_builder",
            name="tiny",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            card={"registered_by": "ppl_mo"},
        )
    )
    harness = EvalHarness(platform.store, platform.audit)
    outcome = harness.evaluate_retrieval(
        HashEmbedder(),
        EvalSet(
            id="evs_self",
            name="self",
            version=1,
            frozen=True,
            created_at="2026-01-01T00:00:00+00:00",
            items=[{"query": "vpn", "relevant_ids": ["vpn"]}],
        ),
        [
            Record(
                id="1",
                record_key="vpn",
                text="reset the vpn",
                source_id="src",
                acl=["engineering"],
            )
        ],
    )

    with pytest.raises(PermissionError, match="builder cannot record"):
        harness.record_report("mdl_builder", "evs_self", outcome, judged_by="ppl_mo")
