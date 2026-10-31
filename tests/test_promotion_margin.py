"""A promotion has to show a lead the plan would believe.

The gate holder approves a decision; they do not re-run the evaluation. So the
report they sign has to carry the comparison, and the platform has to refuse
the small lead itself rather than trusting whoever is holding the gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from internal_ai.domain import EvalReport, EvalSet, ModelVersion
from internal_ai.workflows import PROMOTION_MARGIN, margin_refusal, promote_model


def candidate(platform, card: dict | None = None, model_id: str = "mdl_margin") -> ModelVersion:
    return platform.store.add_model(
        ModelVersion(
            id=model_id,
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            card=card or {"registered_by": "ppl_builder"},
        )
    )


def frozen(platform) -> EvalSet:
    return platform.store.add_eval_set(
        EvalSet(
            id="evs_margin",
            name="held-out",
            version=1,
            frozen=True,
            created_at="2026-01-01T00:00:00+00:00",
            content_hash="c",
        )
    )


def report_for(
    platform, metrics: dict[str, float], baseline: dict[str, float], model_id: str = "mdl_margin"
) -> None:
    platform.store.add_eval_report(
        EvalReport(
            id="rep_margin",
            model_id=model_id,
            eval_set_id="evs_margin",
            created_at="2026-01-02T00:00:00+00:00",
            metrics=metrics,
            baseline_metrics=baseline,
            passed=True,
            judged_by="ppl_judge",
        )
    )


def test_the_margin_is_the_plans_five_points() -> None:
    assert PROMOTION_MARGIN == 0.05


def test_a_keyword_lead_under_the_margin_is_refused_with_both_figures(platform) -> None:
    candidate(platform)
    frozen(platform)
    report_for(platform, {"recall_at_10": 0.41}, {"recall_at_10": 0.39})

    with pytest.raises(PermissionError) as refused:
        promote_model(platform, "mdl_margin", gate_approved=True, approver="ppl_lead")

    assert "candidate 0.41 vs keyword 0.39" in str(refused.value)
    assert "below the +0.05 margin" in str(refused.value)
    assert platform.store.get_model("mdl_margin").status != "promoted"


def test_a_recall_report_without_a_baseline_cannot_promote(platform) -> None:
    candidate(platform)
    frozen(platform)
    report_for(platform, {"recall_at_10": 0.9}, {})

    with pytest.raises(PermissionError, match="never run"):
        promote_model(platform, "mdl_margin", gate_approved=True, approver="ppl_lead")


def test_an_adapters_lead_over_its_base_is_held_to_the_margin(platform) -> None:
    candidate(
        platform,
        card={
            "registered_by": "ppl_builder",
            "base_repo_id": "Qwen/Qwen3-4B",
            "adapter_backend": "peft",
        },
    )
    frozen(platform)
    report_for(platform, {"citation_pass_rate": 0.60}, {"citation_pass_rate": 0.58})

    with pytest.raises(PermissionError, match="un-adapted base"):
        promote_model(platform, "mdl_margin", gate_approved=True, approver="ppl_lead")


def test_a_report_that_compared_nothing_is_refused(platform) -> None:
    candidate(platform)
    frozen(platform)
    report_for(platform, {}, {})

    with pytest.raises(PermissionError, match="nothing was compared"):
        promote_model(platform, "mdl_margin", gate_approved=True, approver="ppl_lead")


def test_a_lead_past_the_margin_promotes(platform) -> None:
    candidate(platform)
    frozen(platform)
    report_for(platform, {"recall_at_10": 0.72}, {"recall_at_10": 0.40})

    promoted = promote_model(platform, "mdl_margin", gate_approved=True, approver="ppl_lead")

    assert promoted.status == "promoted"
    assert promoted.card["baseline_metrics"] == {"recall_at_10": 0.40}


def test_a_tie_is_not_a_gain() -> None:
    report = SimpleNamespace(
        metrics={"recall_at_10": 0.55}, baseline_metrics={"recall_at_10": 0.55}
    )

    assert margin_refusal(report) == (
        "candidate 0.55 vs keyword 0.55 on recall_at_10: +0.00 is below the +0.05 margin"
    )
    clear = SimpleNamespace(
        metrics={"recall_at_10": 0.60, "citation_pass_rate": 0.70},
        baseline_metrics={"recall_at_10": 0.55, "citation_pass_rate": 0.60},
    )
    assert margin_refusal(clear) is None
