"""An adapter is an improvement only when it beats the model it sits on."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from internal_ai.domain import DatasetVersion, EvalReport, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.training.alpha.citations import AskHit
from internal_ai.training.alpha.pin import PRIMARY
from internal_ai.training.alpha.promote import (
    evaluate_alpha,
    register_alpha,
    request_alpha_promotion,
    serve_alpha,
)

HITS = [
    AskHit("Acmy_Website/next.config.ts", 36),
    AskHit("Acmy_Website/src/app/api/payments/razorpay/create-order/route.ts", 8),
]
ITEMS = [
    {
        "query": "which host is allowed in production?",
        "hits": [
            {
                "path": "Acmy_Website/next.config.ts",
                "line": 36,
                "text": "const allowedHost = 'app.example.com';",
            }
        ],
    },
    {
        "query": "which key signs the create-order call?",
        "hits": [
            {
                "path": "Acmy_Website/src/app/api/payments/razorpay/create-order/route.ts",
                "line": 8,
                "text": "const key = process.env.RAZORPAY_KEY_ID;",
            }
        ],
    },
]
GOOD_HOST = "See Acmy_Website/next.config.ts at line 36."
GOOD_KEY = (
    "The key is read in Acmy_Website/src/app/api/payments/razorpay/create-order/route.ts at line 8."
)
MOVED = "See Acmy_Website/next.config.ts at line 99."
TRAINING_TEXT = "the nightly reconciliation job failed on the warehouse ledgerline feed"


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
                "record_key": "train-1",
                "payload": TRAINING_TEXT,
                "source_id": "s",
                "acl": [],
                "split": "train",
            }
        ],
    )


def many_items(count: int) -> list[dict[str, Any]]:
    """Enough questions that a lead can clear two standard errors.

    Two questions cannot: one flipped answer is half the score, and promotion
    refuses a difference that one answer could have produced.
    """
    return [
        {**ITEMS[index % len(ITEMS)], "query": f"{ITEMS[index % len(ITEMS)]['query']} #{index}"}
        for index in range(count)
    ]


def alpha_eval_set(platform, items: list[dict[str, Any]] | None = None) -> str:
    from internal_ai.evals.harness import EvalHarness

    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set("alpha", ITEMS if items is None else items)
    harness.freeze_eval_set(eval_set.id, "ml_lead", gate_approved=True)
    return eval_set.id


def approve_gate(platform, gate_id: str) -> None:
    stored = platform.store.get_gate(gate_id)
    assert stored is not None
    stored.status = type(stored.status)("approved")
    platform.store.update_gate(stored)


def alpha_model(platform, enroll_person, tmp_path: Path) -> tuple[Any, ModelVersion]:
    person = enroll_person(platform, "Elena Alpha", ["ml_lead"])
    dataset(platform, "ds_alpha")
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    record = tmp_path / "reproducibility.json"
    record.write_text(json.dumps({"adapter_backend": "peft"}), encoding="utf-8")
    return person, register_alpha(platform, PRIMARY, adapter, "ds_alpha", person)


def test_an_adapter_that_loses_to_its_base_is_refused_with_both_numbers(
    platform, enroll_person, tmp_path: Path
) -> None:
    _, model = alpha_model(platform, enroll_person, tmp_path)
    eval_set_id = alpha_eval_set(platform)

    report = evaluate_alpha(platform, model.id, eval_set_id, [MOVED, MOVED], [GOOD_HOST, GOOD_KEY])

    assert report.passed is False
    check = report.checks["beats_base"]
    assert check["passed"] is False
    assert check["adapter_citation_pass_rate"] == 0.0
    assert check["base_citation_pass_rate"] == 1.0
    assert "loses to its base" in check["note"]
    assert report.metrics["citation_pass_rate"] == 0.0
    assert report.baseline_metrics == {"citation_pass_rate": 1.0}


def test_matching_the_base_is_not_beating_it(platform, enroll_person, tmp_path: Path) -> None:
    _, model = alpha_model(platform, enroll_person, tmp_path)
    eval_set_id = alpha_eval_set(platform)

    report = evaluate_alpha(platform, model.id, eval_set_id, [GOOD_HOST, MOVED], [MOVED, GOOD_KEY])

    assert report.metrics["citation_pass_rate"] == report.baseline_metrics["citation_pass_rate"]
    assert report.checks["beats_base"]["passed"] is False
    assert "cannot tell the two apart" in report.checks["beats_base"]["note"]
    assert report.passed is False


def test_identical_answers_from_both_runs_name_the_likely_cause(
    platform, enroll_person, tmp_path: Path
) -> None:
    _, model = alpha_model(platform, enroll_person, tmp_path)
    eval_set_id = alpha_eval_set(platform)

    report = evaluate_alpha(
        platform, model.id, eval_set_id, [GOOD_HOST, GOOD_KEY], [GOOD_HOST, GOOD_KEY]
    )

    assert report.checks["beats_base"]["passed"] is False
    assert "adapter was not loaded" in report.checks["beats_base"]["note"]


def test_contamination_is_measured_against_the_training_split(
    platform, enroll_person, tmp_path: Path, monkeypatch
) -> None:
    """Reading every record made the check impossible to pass for a held-out split."""
    from internal_ai.training.alpha.promote import _alpha_contamination

    _, model = alpha_model(platform, enroll_person, tmp_path)
    eval_set = platform.store.get_eval_set(alpha_eval_set(platform))
    asked: dict[str, str] = {}

    def fake(dataset_id: str, split: str) -> list[dict]:
        asked["dataset_id"] = dataset_id
        asked["split"] = split
        return []

    monkeypatch.setattr(platform.store, "dataset_records_by_split", fake)
    check = _alpha_contamination(platform, model, eval_set)

    assert asked == {"dataset_id": "ds_alpha", "split": "train"}
    assert check["passed"] is False


LEDGER = "the warehouse ledgerline reconciliation failed on the nightly feed."


def test_the_keyword_baseline_searches_the_lines_the_answers_live_on(
    platform, enroll_person, tmp_path: Path
) -> None:
    """Its index was the training split, which held none of the 32 answers.

    A comparator that cannot reach any answer scores zero against every model, so
    the gate that asks whether the adapter beats it passes on any kept citation.
    """
    from internal_ai.training.alpha.promote import keyword_baseline_answers

    alpha_model(platform, enroll_person, tmp_path)
    platform.store.add_dataset_records(
        "ds_alpha",
        [
            {
                "record_key": "notes.txt:7",
                "payload": LEDGER,
                "source_id": "s",
                "acl": [],
                "split": "heldout",
            }
        ],
    )
    eval_set = platform.store.get_eval_set(
        alpha_eval_set(
            platform,
            [
                {
                    "query": "warehouse ledgerline reconciliation nightly",
                    "hits": [{"path": "notes.txt", "line": 7, "text": LEDGER}],
                }
            ],
        )
    )

    answers = keyword_baseline_answers(platform, eval_set, "ds_alpha")

    assert answers == [f"{LEDGER} (notes.txt:7)"]


def test_an_answer_that_cites_nothing_has_kept_nothing(
    platform, enroll_person, tmp_path: Path
) -> None:
    """Silence is not a kept citation, or the model that says least wins."""
    _, model = alpha_model(platform, enroll_person, tmp_path)
    eval_set_id = alpha_eval_set(platform)

    report = evaluate_alpha(
        platform,
        model.id,
        eval_set_id,
        ["I could not find that in the corpus.", "No passage found."],
        [GOOD_HOST, GOOD_KEY],
    )

    assert report.metrics["citation_pass_rate"] == 0.0
    assert report.checks["beats_base"]["passed"] is False


def test_an_evaluation_without_the_base_answers_is_refused(
    platform, enroll_person, tmp_path: Path
) -> None:
    _, model = alpha_model(platform, enroll_person, tmp_path)
    eval_set_id = alpha_eval_set(platform)

    with pytest.raises(ValueError, match="baseline answers"):
        evaluate_alpha(platform, model.id, eval_set_id, [GOOD_HOST, GOOD_KEY], [])


def test_a_passing_report_from_before_the_base_check_cannot_serve(
    platform, enroll_person, tmp_path: Path
) -> None:
    person, model = alpha_model(platform, enroll_person, tmp_path)
    gate = request_alpha_promotion(platform, person, model.id, GOOD_HOST, HITS)
    approve_gate(platform, gate.id)
    platform.store.add_eval_report(
        EvalReport(
            id="rep_uncompared",
            model_id=model.id,
            eval_set_id=alpha_eval_set(platform),
            created_at=now_iso(),
            metrics={"citation_pass_rate": 1.0},
            passed=True,
            notes="scored without asking the base",
        )
    )

    with pytest.raises(PermissionError, match="beat the un-adapted base"):
        serve_alpha(platform, model.id, tmp_path / "runner")
    assert platform.store.get_model(model.id).status == "candidate"


def test_beating_the_base_promotes_and_keeps_both_numbers_on_the_card(
    platform, enroll_person, tmp_path: Path
) -> None:
    person, model = alpha_model(platform, enroll_person, tmp_path)
    gate = request_alpha_promotion(platform, person, model.id, GOOD_HOST, HITS)
    approve_gate(platform, gate.id)
    eval_set_id = alpha_eval_set(platform, many_items(40))

    report = evaluate_alpha(
        platform,
        model.id,
        eval_set_id,
        [GOOD_HOST, GOOD_KEY] * 20,
        [GOOD_HOST, GOOD_KEY] * 10 + [MOVED, GOOD_KEY] * 10,
    )
    assert report.passed is True, report.checks
    assert report.checks["beats_base"]["adapter_citation_pass_rate"] == 1.0
    assert report.checks["beats_base"]["base_citation_pass_rate"] == 0.75

    serve_alpha(platform, model.id, tmp_path / "runner")
    promoted = platform.store.get_model(model.id)
    assert promoted.status == "promoted"
    assert promoted.card["eval_results"]["citation_pass_rate"] == 1.0
    assert promoted.card["baseline_metrics"]["citation_pass_rate"] == 0.75
