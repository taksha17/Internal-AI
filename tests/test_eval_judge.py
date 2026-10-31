"""A model is judged by someone other than the person who built it.

The gate needs two approvers, but the report those approvers read used to come
from the builder's own hands. The report now names the judge, the product
refuses a self-run, and promotion refuses a self-judged report.
"""

from __future__ import annotations

from conftest import enroll

from internal_ai.domain import EvalReport, EvalSet, ModelVersion
from internal_ai.evals.harness import EvalHarness, EvalOutcome
from internal_ai.services import build
from internal_ai.services.permissions import PermissionDenied
from internal_ai.workflows import promote_model


def model_built_by(platform, builder_id: str, model_id: str = "mdl_self") -> ModelVersion:
    return platform.store.add_model(
        ModelVersion(
            id=model_id,
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            card={"registered_by": builder_id, "tokenizer_path": "/missing/tokenizer.json"},
        )
    )


def frozen_set(platform) -> EvalSet:
    return platform.store.add_eval_set(
        EvalSet(
            id="evs_judge",
            name="held-out",
            version=1,
            frozen=True,
            created_at="2026-01-01T00:00:00+00:00",
            content_hash="c",
        )
    )


def add_report(platform, model_id: str, judged_by: str, passed: bool = True) -> None:
    platform.store.add_eval_report(
        EvalReport(
            id=f"rep_{judged_by}",
            model_id=model_id,
            eval_set_id="evs_judge",
            created_at="2026-01-02T00:00:00+00:00",
            metrics={"recall_at_10": 0.8},
            baseline_metrics={"recall_at_10": 0.5},
            passed=passed,
            judged_by=judged_by,
        )
    )


def test_a_builder_cannot_run_the_evaluation_of_their_own_model(platform) -> None:
    builder = enroll(platform, "Mo Builder", ["ml_lead"])
    model_built_by(platform, builder.id)
    frozen_set(platform)

    try:
        build.evaluate_model(platform, builder, None, "mdl_self", "evs_judge")
    except PermissionDenied as error:
        assert "cannot judge their own model" in str(error)
    else:
        raise AssertionError("a builder evaluated their own model")


def test_the_report_records_who_judged_it(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    outcome = EvalOutcome(
        passed=True, metrics={"recall_at_10": 0.8}, baseline_metrics={}, checks={}, notes=""
    )

    named = harness.record_report("mdl_x", "evs_x", outcome, actor="evaluation", judged_by="alice")
    default = harness.record_report("mdl_y", "evs_y", outcome, actor="evaluation")

    assert named.judged_by == "alice"
    assert default.judged_by == "evaluation"
    assert platform.store.get_eval_report(named.id).judged_by == "alice"


def test_a_report_the_builder_judged_cannot_promote(platform) -> None:
    builder = enroll(platform, "Mo Builder", ["ml_lead"])
    model_built_by(platform, builder.id)
    frozen_set(platform)
    add_report(platform, "mdl_self", judged_by=builder.id)

    try:
        promote_model(platform, "mdl_self", gate_approved=True, approver="mia")
    except PermissionError as error:
        assert "different ml_lead" in str(error)
    else:
        raise AssertionError("a self-judged report promoted a model")
    assert platform.store.get_model("mdl_self").status != "promoted"


def test_a_report_from_another_judge_promotes(platform) -> None:
    builder = enroll(platform, "Mo Builder", ["ml_lead"])
    model_built_by(platform, builder.id)
    frozen_set(platform)
    add_report(platform, "mdl_self", judged_by="mia")

    promoted = promote_model(platform, "mdl_self", gate_approved=True, approver="lena")

    assert promoted.status == "promoted"
