"""A recipe is what a rebase repeats; four columns decide whether it counts."""

from __future__ import annotations

import pytest
from conftest import enroll

from internal_ai.brief import Metric, ProjectBrief
from internal_ai.brief import save as save_brief
from internal_ai.domain import DatasetVersion
from internal_ai.evals.harness import EvalHarness
from internal_ai.ids import now_iso
from internal_ai.services.permissions import PermissionDenied
from internal_ai.training.alpha.pin import PRIMARY, PinnedBase
from internal_ai.training.rebase import build_recipe, compare, load_recipe, propose

RESEARCH = PinnedBase(
    repo_id="acme/base-3b", revision="deadbeef", license="qwen-research", shards=()
)
GOOD = {
    "old": {"recall_at_10": 0.70},
    "new": {"recall_at_10": 0.80},
    "base_only": {"recall_at_10": 0.75},
    "rented": {"recall_at_10": 0.85},
}


def lead(platform, name: str = "Mo Lead"):
    return enroll(platform, name, ["ml_lead"], ["engineering"])


def released_dataset(platform, *, released: bool = True) -> str:
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_rebase",
            name="company",
            version=1,
            content_hash="hash-rebase",
            record_count=1,
            token_count=1,
            created_at=now_iso(),
            released=released,
        )
    )
    platform.store.add_dataset_records(
        "ds_rebase",
        [{"record_key": "r1", "payload": "the ledger feed failed overnight", "acl": []}],
    )
    return "ds_rebase"


def frozen_eval_set(platform, actor) -> str:
    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set("rebase", [{"query": "ledger", "relevant_ids": ["d1"]}])
    harness.freeze_eval_set(eval_set.id, actor.id, gate_approved=True)
    return eval_set.id


def record_bar(platform, actor, threshold: str = "0.7") -> None:
    save_brief(
        platform.store,
        platform.audit,
        actor,
        ProjectBrief(metrics=[Metric(name="recall@10", threshold=threshold)]),
    )


def setup(platform):
    mo = lead(platform)
    record_bar(platform, mo)
    return mo, build_recipe(
        platform,
        mo,
        name="quarterly-rebase",
        dataset_id=released_dataset(platform),
        base=PRIMARY,
        eval_set_ids=[frozen_eval_set(platform, mo)],
    )


def report(recipe, **columns):
    values = {key: dict(value) for key, value in GOOD.items()}
    values.update(columns)
    return compare(recipe, **values)


def test_a_recipe_records_what_a_rebase_repeats(platform) -> None:
    mo, recipe = setup(platform)

    assert recipe.dataset_hash
    assert recipe.base_repo_id == PRIMARY.repo_id
    assert recipe.base_revision == PRIMARY.revision
    assert recipe.license_class == "permissive"
    assert len(recipe.eval_set_ids) == 1
    assert [metric.name for metric in recipe.metrics] == ["recall@10"]
    assert recipe.created_by == mo.id

    loaded = load_recipe(platform, recipe.id)
    assert loaded == recipe


def test_building_a_recipe_needs_the_role(platform) -> None:
    owner = enroll(platform, "Omar Owner", ["data_owner"], ["engineering"])

    with pytest.raises(PermissionDenied):
        build_recipe(
            platform,
            owner,
            name="sneaky",
            dataset_id="ds_missing",
            base=PRIMARY,
            eval_set_ids=["evs_missing"],
        )


def test_an_unreleased_dataset_is_refused(platform) -> None:
    mo = lead(platform)
    record_bar(platform, mo)

    with pytest.raises(PermissionError, match="released dataset"):
        build_recipe(
            platform,
            mo,
            name="unreleased",
            dataset_id=released_dataset(platform, released=False),
            base=PRIMARY,
            eval_set_ids=[frozen_eval_set(platform, mo)],
        )


def test_an_unfrozen_eval_set_is_refused(platform) -> None:
    mo = lead(platform)
    record_bar(platform, mo)
    harness = EvalHarness(platform.store, platform.audit)
    eval_set = harness.create_eval_set("loose", [{"query": "ledger", "relevant_ids": ["d1"]}])

    with pytest.raises(PermissionError, match="not frozen"):
        build_recipe(
            platform,
            mo,
            name="loose",
            dataset_id=released_dataset(platform),
            base=PRIMARY,
            eval_set_ids=[eval_set.id],
        )


def test_a_research_licensed_base_is_refused(platform) -> None:
    mo = lead(platform)
    record_bar(platform, mo)

    with pytest.raises(PermissionError, match="research_only"):
        build_recipe(
            platform,
            mo,
            name="research",
            dataset_id=released_dataset(platform),
            base=RESEARCH,
            eval_set_ids=[frozen_eval_set(platform, mo)],
        )


def test_a_brief_without_a_metric_is_refused(platform) -> None:
    mo = lead(platform)

    with pytest.raises(ValueError, match="nothing to be judged by"):
        build_recipe(
            platform,
            mo,
            name="no-bar",
            dataset_id=released_dataset(platform),
            base=PRIMARY,
            eval_set_ids=[frozen_eval_set(platform, mo)],
        )


def test_the_report_carries_four_columns(platform) -> None:
    _, recipe = setup(platform)

    result = report(recipe)

    assert result.proposes_promotion is True
    assert sorted(result.columns) == ["base_only", "new", "old", "rented"]
    assert result.rows[0]["new"] == pytest.approx(0.80)
    assert "rented" in result.text()


def test_a_missed_bar_is_refused(platform) -> None:
    mo, recipe = setup(platform)
    result = report(recipe, new={"recall_at_10": 0.60})

    assert result.proposes_promotion is False
    with pytest.raises(PermissionError, match="does not earn a promotion"):
        propose(platform, mo, result)
    events = platform.store.list_audit(action="rebase.promotion_refused", limit=5)
    assert events and events[-1].subject == recipe.id


def test_an_adapter_that_ties_its_base_is_not_a_gain(platform) -> None:
    _, recipe = setup(platform)

    result = report(recipe, base_only={"recall_at_10": 0.80})

    assert result.proposes_promotion is False
    assert any("not a gain" in breach for breach in result.breaches)


def test_losing_ground_against_the_old_adapter_is_refused(platform) -> None:
    _, recipe = setup(platform)

    result = report(recipe, old={"recall_at_10": 0.90})

    assert result.proposes_promotion is False
    assert any("lost ground" in breach for breach in result.breaches)


def test_a_column_that_measures_nothing_is_refused(platform) -> None:
    _, recipe = setup(platform)

    with pytest.raises(ValueError, match="no measurements"):
        report(recipe, rented={})


def test_a_column_missing_the_brief_metric_is_a_breach_not_a_blank(platform) -> None:
    _, recipe = setup(platform)

    result = report(recipe, rented={"precision_at_10": 0.9})

    assert result.proposes_promotion is False
    assert any("rented API" in breach for breach in result.breaches)


def test_a_passing_rebase_is_proposed_and_leaves_serving_alone(platform) -> None:
    mo, recipe = setup(platform)

    propose(platform, mo, report(recipe))

    events = platform.store.list_audit(action="rebase.promotion_proposed", limit=5)
    assert events and events[-1].subject == recipe.id
    assert platform.store.list_audit(action="model.activated", limit=5) == []
