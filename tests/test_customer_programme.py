"""The twelve stages stay open until the artifact for that stage exists."""

from __future__ import annotations

from pathlib import Path

from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.domain import DatasetVersion, EvalReport, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.programme import customer_stages


def _by_name(platform) -> dict:
    return {stage["name"]: stage for stage in customer_stages(platform)}


def test_a_fresh_install_has_twelve_open_stages(platform) -> None:
    stages = customer_stages(platform)
    assert [stage["n"] for stage in stages] == list(range(1, 13))
    assert all(stage["done"] is False for stage in stages)
    assert "target" in stages[0]["waiting"]
    assert stages[2]["lines"] == []


def test_a_released_dataset_closes_only_the_data_stage(platform) -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_one",
            name="corpus",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=10,
            created_at=now_iso(),
            released=True,
        )
    )
    stages = _by_name(platform)
    assert stages["Data"]["done"] is True
    assert stages["Tokenizer"]["done"] is False
    assert stages["Pretraining"]["done"] is False


def test_an_alpha_model_does_not_close_pretraining_or_serving(platform) -> None:
    platform.store.add_model(
        ModelVersion(
            id="mdl_alpha",
            name="qwen",
            version=1,
            content_hash="abc",
            created_at=now_iso(),
            status="promoted",
            card={"track": "alpha"},
        )
    )
    platform.store.add_eval_report(
        EvalReport(
            id="rep_alpha",
            model_id="mdl_alpha",
            eval_set_id="evs_one",
            created_at=now_iso(),
            passed=True,
            checks={"safety": {"passed": True}},
        )
    )
    stages = _by_name(platform)
    assert stages["Pretraining"]["done"] is False
    assert stages["Serving"]["done"] is False
    assert stages["Red team"]["done"] is True


def test_a_passing_from_scratch_model_can_be_served(platform) -> None:
    platform.store.add_model(
        ModelVersion(
            id="mdl_small",
            name="small",
            version=1,
            content_hash="abc",
            created_at=now_iso(),
            status="promoted",
            card={},
        )
    )
    platform.store.add_eval_report(
        EvalReport(
            id="rep_small",
            model_id="mdl_small",
            eval_set_id="evs_one",
            created_at=now_iso(),
            passed=True,
        )
    )
    stages = _by_name(platform)
    assert stages["Pretraining"]["done"] is True
    assert stages["Serving"]["done"] is True


def test_a_failed_report_leaves_serving_open(platform) -> None:
    platform.store.add_model(
        ModelVersion(
            id="mdl_small",
            name="small",
            version=1,
            content_hash="abc",
            created_at=now_iso(),
            status="promoted",
            card={},
        )
    )
    platform.store.add_eval_report(
        EvalReport(
            id="rep_fail",
            model_id="mdl_small",
            eval_set_id="evs_one",
            created_at=now_iso(),
            passed=False,
        )
    )
    assert _by_name(platform)["Serving"]["done"] is False


def test_feedback_closes_the_loop(platform) -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_fb",
            name="ask-feedback",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=4,
            created_at=now_iso(),
            lineage={"kind": "feedback"},
            released=False,
        )
    )
    assert _by_name(platform)["The loop"]["done"] is True


def test_the_board_prints_hours_and_leaves_a_large_size_waiting(platform) -> None:
    from internal_ai.compute.advise import advise_dataset

    platform.store.add_dataset(
        DatasetVersion(
            id="ds_hours",
            name="corpus",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=200_000_000,
            created_at=now_iso(),
            stats={"token_count_kind": "measured"},
        )
    )
    advise_dataset(platform, "ds_hours")
    stage = _by_name(platform)["Compute"]
    assert stage["done"] is True
    text = " ".join(stage["lines"])
    assert "small" in text and "runnable" in text
    assert "7b" in text and "waiting" in text


def test_the_home_page_shows_the_hours_under_compute(tmp_path: Path) -> None:
    from internal_ai.compute.advise import advise_dataset
    from internal_ai.context import Platform

    setup = Platform.create(root=tmp_path, llm_kind="scripted")
    enroll(setup, "Ada Admin", ["program_lead"])
    setup.store.add_dataset(
        DatasetVersion(
            id="ds_hours",
            name="corpus",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=200_000_000,
            created_at=now_iso(),
            stats={"token_count_kind": "measured"},
        )
    )
    advise_dataset(setup, "ds_hours")
    setup.close()

    app = create_app(root=tmp_path, llm_kind="scripted")
    client = TestClient(app)
    with client:
        assert sign_in(client, "Ada Admin").status_code == 303
        page = client.get("/console/home").text
    assert "3. Compute" in page
    assert "small " in page and "GPU-hours" in page and "runnable" in page
    assert "7b " in page and "waiting" in page
    assert page.count("<p class=\"muted small\">") >= 3
