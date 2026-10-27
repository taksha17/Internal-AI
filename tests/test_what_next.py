"""What a failed run says next, and that it never blames the wrong cause.

A safety check can fail while the score is fine, and a small corpus can be the
whole story. These rules must tell those apart, name the number they rest on,
and give the same answer every time.
"""

from __future__ import annotations

from pathlib import Path

from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.data.suggest import GENERATE_RECORDS
from internal_ai.evals.advice import (
    classifier_advice,
    failed_checks,
    retrieval_advice,
)

CLEAN_CHECKS = {
    "contamination": {"passed": True},
    "pii_regurgitation": {"passed": True},
    "permission_leak": {"passed": True},
    "baseline_informativeness": {"passed": True, "baseline_mean_bm25_score": 2.5},
}


def retrieval_report(**over: object) -> dict:
    report = {
        "passed": False,
        "metrics": {"recall_at_10": 0.06, "mrr": 0.06},
        "baseline_metrics": {"recall_at_10": 0.18, "mrr": 0.18},
        "checks": dict(CLEAN_CHECKS),
        "notes": "model does not beat the keyword-search baseline by the required margin",
    }
    report.update(over)
    return report


def classifier_report(**over: object) -> dict:
    report = {
        "passed": False,
        "accuracy": 0.5,
        "baseline_accuracy": 0.5,
        "min_accuracy": 0.6,
        "label_distribution": {"billing": 40, "refunds": 2},
        "notes": "classifier does not beat the majority-class baseline or misses the floor",
    }
    report.update(over)
    return report


def story(lines: list[str]) -> str:
    return " ".join(lines)


def test_a_small_corpus_is_told_keyword_search_wins_at_this_size() -> None:
    lines = retrieval_advice(retrieval_report(), records=98, tokens=25_000)

    text = story(lines)
    assert "98 records" in text
    assert "25,000 tokens" in text
    assert "keyword search is expected to win" in text
    assert f"{GENERATE_RECORDS:,} usable records" in text
    assert "0.060" in text and "0.180" in text


def test_a_large_corpus_is_not_told_to_add_more_data() -> None:
    lines = retrieval_advice(retrieval_report(), records=9_000, tokens=4_000_000)

    text = story(lines)
    assert "past the volume bar" in text
    assert "dedup and quality" in text
    assert "about 500 usable records is where" not in text
    assert "keep using it until the corpus grows" not in text


def test_a_passing_run_is_given_no_advice() -> None:
    assert retrieval_advice(retrieval_report(passed=True), records=98, tokens=25_000) == []
    assert classifier_advice(classifier_report(passed=True)) == []
    assert retrieval_advice(None, records=0, tokens=0) == []
    assert classifier_advice(None) == []


def test_a_failed_safety_check_is_read_before_the_score() -> None:
    checks = dict(CLEAN_CHECKS)
    checks["contamination"] = {"passed": False, "overlap": 3}

    lines = retrieval_advice(retrieval_report(checks=checks), records=98, tokens=25_000)

    text = story(lines)
    assert "contamination check did not pass" in text
    assert "Rebuild the dataset's split" in text
    assert "keyword search is expected to win" not in text
    assert "usable records is where" not in text


def test_a_pii_failure_says_scrub_rather_than_more_data() -> None:
    checks = dict(CLEAN_CHECKS)
    checks["pii_regurgitation"] = {"passed": False, "hits": [{"probe": "ssn"}]}

    lines = retrieval_advice(retrieval_report(checks=checks), records=98, tokens=25_000)

    text = story(lines)
    assert "PII check did not pass" in text
    assert "Scrub the dataset" in text


def test_the_informational_check_is_never_a_reason_to_refuse() -> None:
    checks = dict(CLEAN_CHECKS)
    checks["baseline_informativeness"] = {"passed": False, "baseline_mean_bm25_score": 0.0}

    assert failed_checks(retrieval_report(checks=checks)) == []
    lines = retrieval_advice(retrieval_report(checks=checks), records=98, tokens=25_000)
    assert "baseline_informativeness" not in story(lines)


def test_a_classifier_that_ties_the_majority_class_is_told_so() -> None:
    lines = classifier_advice(classifier_report())

    text = story(lines)
    assert "no better than always saying the most common label" in text
    assert "accuracy 0.500 against a majority-class baseline of 0.500" in text
    assert "More answers per label" in text
    assert "The least-represented label, refunds, has 2 of 42 rows" in text


def test_a_classifier_under_the_floor_is_told_which() -> None:
    lines = classifier_advice(classifier_report(accuracy=0.7, baseline_accuracy=0.5))

    text = story(lines)
    assert "under the floor" in text
    assert "accuracy 0.700" in text
    assert "0.600" in text
    assert "no better than always saying" not in text


def test_a_single_label_says_there_is_nothing_to_separate() -> None:
    lines = classifier_advice(classifier_report(label_distribution={"billing": 40}))

    assert "Only one label is present" in story(lines)


def test_the_same_report_always_reads_the_same_way() -> None:
    first = retrieval_advice(retrieval_report(), records=98, tokens=25_000)
    second = retrieval_advice(retrieval_report(), records=98, tokens=25_000)
    third = classifier_advice(classifier_report())

    assert first == second
    assert third == classifier_advice(classifier_report())


def test_the_models_page_explains_a_run_that_failed(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        builder = enroll(app.state.platform, "Mo Builder", ["ml_lead"])
        assert sign_in(client, builder.name).status_code == 303

        app.state.jobs.submit(
            builder,
            "train_classifier",
            "Train classifier router-v1",
            {"dataset_id": "ds_x", "name": "router-v1"},
            lambda: {
                "run_id": "run_x",
                "name": "router-v1",
                "status": "completed",
                "accuracy": 0.5,
                "baseline_accuracy": 0.5,
                "dataset_id": "ds_x",
            },
            agent="ml_engineer",
        )
        app.state.jobs.submit(
            builder,
            "evaluate_classifier",
            "Evaluate classifier router-v1 on the test split",
            {"run_id": "run_x", "job_id": "cjob_x", "dataset_id": "ds_x", "split": "test"},
            lambda: classifier_report(),
            agent="evaluation",
        )

        page = client.get("/console/models")

        assert page.status_code == 200
        assert "advice-list" in page.text
        assert "no better than always saying the most common label" in page.text
        assert "The least-represented label, refunds" in page.text
