"""A citation that landed is not an answer that is right; both are reported."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from internal_ai.domain import DatasetVersion
from internal_ai.evals.metrics import token_f1
from internal_ai.ids import now_iso
from internal_ai.training.alpha.pin import PRIMARY
from internal_ai.training.alpha.promote import evaluate_alpha, register_alpha

SPAN = "A server generating a 401 response MUST send a WWW-Authenticate header field."
HIT = {"path": "rfc/rfc7235.txt", "line": 312, "text": SPAN}
ITEM = {
    "query": "What must a server generating a 401 response do?",
    "hits": [HIT],
    "answer": SPAN,
    "relevant_ids": ["rfc/rfc7235.txt:312"],
}


def dataset(platform, dataset_id: str) -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="company",
            version=1,
            content_hash=f"hash-{dataset_id}",
            record_count=1,
            token_count=1,
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": "rfc/rfc7235.txt:311",
                "payload": "The request requires user authentication.",
                "source_id": "rfc",
                "acl": [],
                "split": "train",
            }
        ],
    )


def frozen_set(platform, items: list[dict]) -> str:
    from internal_ai.evals.harness import EvalHarness

    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set("alpha", items)
    harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=True)
    return eval_set.id


def alpha_model(platform, enroll_person, tmp_path: Path, dataset_id: str):
    person = enroll_person(platform, "Elena Alpha", ["ml_lead"])
    dataset(platform, dataset_id)
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    (tmp_path / "reproducibility.json").write_text(
        json.dumps({"adapter_backend": "peft"}), encoding="utf-8"
    )
    return register_alpha(platform, PRIMARY, adapter, dataset_id, person)


def test_extractive_f1_scores_the_overlap_of_an_answer_with_its_reference() -> None:
    assert token_f1(SPAN, SPAN) == 1.0
    assert token_f1(SPAN.upper(), SPAN) == 1.0
    assert token_f1("a server must send a response", SPAN) < 1.0
    assert token_f1("a server must send a response", SPAN) > 0.0
    assert token_f1("the nightly job failed", SPAN) == 0.0
    assert token_f1("", SPAN) == 0.0


def test_the_report_carries_answer_correctness_beside_citation_validity(
    platform, enroll_person, tmp_path: Path
) -> None:
    model = alpha_model(platform, enroll_person, tmp_path, "ds_alpha")
    eval_set_id = frozen_set(platform, [ITEM])
    verbatim = f"{SPAN} (rfc/rfc7235.txt:312)"
    paraphrased = "A server sending a 401 response must include a WWW-Authenticate field."

    report = evaluate_alpha(platform, model.id, eval_set_id, [verbatim], [paraphrased])

    assert report.metrics["citation_pass_rate"] == 1.0
    assert report.metrics["answer_f1"] == 1.0
    assert report.baseline_metrics["answer_f1"] < 1.0
    assert set(report.per_case["answer_f1"]) == {"candidate", "baseline", "keyword"}
    assert len(report.per_case["answer_f1"]["candidate"]) == 1


def test_a_citation_marker_does_not_count_as_answering(
    platform, enroll_person, tmp_path: Path
) -> None:
    """The line number is what the citation check reads, so it is not answer text."""
    model = alpha_model(platform, enroll_person, tmp_path, "ds_alpha")
    eval_set_id = frozen_set(platform, [ITEM])

    bare = evaluate_alpha(platform, model.id, eval_set_id, [SPAN], [SPAN])
    with_marker = evaluate_alpha(
        platform,
        model.id,
        frozen_set(platform, [ITEM]),
        [f"line 312 rfc/rfc7235.txt:312 {SPAN}"],
        [SPAN],
    )

    assert with_marker.metrics["answer_f1"] == bare.metrics["answer_f1"] == 1.0


def test_the_notes_state_both_numbers(platform, enroll_person, tmp_path: Path) -> None:
    """A gate holder reads the prose, so the prose carries the second number too."""
    model = alpha_model(platform, enroll_person, tmp_path, "ds_alpha")
    report = evaluate_alpha(
        platform,
        model.id,
        frozen_set(platform, [ITEM]),
        [SPAN],
        ["the nightly reconciliation job failed on the warehouse ledgerline feed"],
    )

    assert "citation" in report.notes
    assert "answer" in report.notes.lower()
