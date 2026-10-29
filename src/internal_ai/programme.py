"""The twelve stages a customer's data walks. A stage closes only on its artifact."""

from __future__ import annotations

from typing import Any

from internal_ai.brief import programme_gaps
from internal_ai.context import Platform


def customer_stages(platform: Platform) -> list[dict[str, Any]]:
    """Each stage, whether its artifact exists, and the one thing still missing."""
    datasets = platform.store.list_datasets()
    models = platform.store.list_models()
    scratch = [model for model in models if model.card.get("track") != "alpha"]
    reports = platform.store.list_eval_reports()
    gaps = programme_gaps(platform, release=False)
    return [
        _stage(1, "Target", "/console/brief", not gaps, "; ".join(gaps) or "record the brief"),
        _stage(
            2,
            "Legal clearance",
            "/console/sources",
            _legal_cleared(platform),
            "approve a source that has a legal status",
        ),
        _stage(
            3,
            "Compute",
            "/console/jobs",
            _compute_known(platform),
            "record what this corpus needs",
        ),
        _stage(
            4,
            "Data",
            "/console/datasets",
            any(item.released for item in datasets),
            "release a dataset",
        ),
        _stage(
            5,
            "Tokenizer",
            "/console/datasets",
            _tokenizer_measured(platform, datasets),
            "train a tokenizer and measure the token count",
        ),
        _stage(
            6,
            "Evals",
            "/console/evals",
            any(item.frozen for item in platform.store.list_eval_sets()),
            "freeze an eval set before training",
        ),
        _stage(
            7,
            "Ablations",
            "/console/datasets",
            any(item.stats.get("ablations") for item in datasets),
            "record an ablation and leave the choice open",
        ),
        _stage(
            8,
            "Pretraining",
            "/console/models",
            bool(scratch),
            "train a model from random weights",
        ),
        _stage(
            9,
            "Post-training",
            "/console/datasets",
            _supervision_ingested(platform),
            "ingest pairs a person wrote",
        ),
        _stage(
            10,
            "Red team",
            "/console/models",
            _safety_recorded(reports),
            "record a safety probe",
        ),
        _stage(
            11,
            "Serving",
            "/console/models",
            _serving_ready(scratch, reports),
            "promote a from-scratch model whose frozen report passed",
        ),
        _stage(
            12,
            "The loop",
            "/console/datasets",
            _feedback_exported(datasets),
            "export filed misses as the next dataset version",
        ),
    ]


def _stage(number: int, name: str, href: str, done: bool, waiting: str) -> dict[str, Any]:
    return {
        "n": number,
        "name": name,
        "href": href,
        "done": done,
        "waiting": "" if done else waiting,
    }


def _legal_cleared(platform: Platform) -> bool:
    from internal_ai.services.build import list_sources

    allowed = {"internal", "licensed", "public"}
    return any(
        item.get("legal_status") in allowed and item.get("gate_status") == "approved"
        for item in list_sources(platform)
    )


def _compute_known(platform: Platform) -> bool:
    advised = platform.store.conn.execute(
        "SELECT 1 FROM kv WHERE key LIKE 'compute_advice:%' LIMIT 1"
    ).fetchone()
    if advised:
        return True
    if platform.store.list_audit(action="compute.advised", limit=1):
        return True
    if platform.store.list_audit(action="train.compute", limit=1):
        return True
    reserved = platform.store.conn.execute(
        "SELECT 1 FROM evidence WHERE kind = 'compute_reservation' LIMIT 1"
    ).fetchone()
    return reserved is not None


def _tokenizer_measured(platform: Platform, datasets: list[Any]) -> bool:
    from internal_ai.services.build import tokenizer_path

    for dataset in datasets:
        if dataset.stats.get("token_count_kind") != "measured":
            continue
        if tokenizer_path(platform, dataset.id) is not None:
            return True
    return False


def _supervision_ingested(platform: Platform) -> bool:
    events = platform.store.list_audit(action="supervision.ingested", limit=20)
    return any(int(event.details.get("authored_by_model") or 0) == 0 for event in events)


def _safety_recorded(reports: list[Any]) -> bool:
    return any("safety" in report.checks or "red_team" in report.checks for report in reports)


def _feedback_exported(datasets: list[Any]) -> bool:
    return any(
        item.stats.get("kind") == "feedback" or item.lineage.get("kind") == "feedback"
        for item in datasets
    )


def _serving_ready(scratch: list[Any], reports: list[Any]) -> bool:
    by_model: dict[str, Any] = {}
    for report in reports:
        by_model[report.model_id] = report
    for model in scratch:
        report = by_model.get(model.id)
        if model.status == "promoted" and report is not None and report.passed:
            return True
    return False
