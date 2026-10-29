"""A model that failed its newest evaluation stops answering requests.

The promotion gate approves one report, but serving answers every request after
it. A later failing re-run is evidence about that model now, so the newest
frozen report decides whether it may still be reached.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from internal_ai.context import Platform
from internal_ai.domain import EvalReport, EvalSet, ModelVersion
from internal_ai.serving.service import ModelRouter, Principal, out_of_service, served_model


def add_model(platform: Platform, model_id: str, version: int, status: str = "promoted") -> None:
    platform.store.add_model(
        ModelVersion(
            id=model_id,
            name="embedding-v1",
            version=version,
            content_hash=f"h{version}",
            created_at="now",
            status=status,
        )
    )


def add_frozen_set(platform: Platform, eval_set_id: str = "evs_held_out") -> None:
    platform.store.add_eval_set(
        EvalSet(
            id=eval_set_id,
            name="held-out",
            version=1,
            frozen=True,
            created_at="2026-01-01T00:00:00+00:00",
            content_hash="c",
        )
    )


def add_report(
    platform: Platform,
    model_id: str,
    created_at: str,
    recall: float,
    *,
    passed: bool,
    eval_set_id: str = "evs_held_out",
) -> None:
    platform.store.add_eval_report(
        EvalReport(
            id=f"rep_{model_id}_{created_at}",
            model_id=model_id,
            eval_set_id=eval_set_id,
            created_at=created_at,
            metrics={"recall_at_10": recall},
            baseline_metrics={"recall_at_10": 0.375},
            passed=passed,
        )
    )


def test_a_later_failing_report_takes_a_promoted_model_out_of_service(platform) -> None:
    add_frozen_set(platform)
    add_model(platform, "mdl_309a38d221fd", 1)
    add_report(platform, "mdl_309a38d221fd", "2026-01-02T00:00:00+00:00", 0.61, passed=True)
    assert served_model(platform.store, ModelRouter(platform.store, platform.audit)) is not None

    add_report(platform, "mdl_309a38d221fd", "2026-01-03T00:00:00+00:00", 0.083, passed=False)

    refusal = out_of_service(platform.store, platform.store.get_model("mdl_309a38d221fd"))
    assert refusal is not None
    assert "0.083" in refusal and "0.375" in refusal
    assert served_model(platform.store, ModelRouter(platform.store, platform.audit)) is None


def test_a_refused_version_falls_back_to_the_last_one_that_passed(platform) -> None:
    add_frozen_set(platform)
    add_model(platform, "mdl_old", 1)
    add_report(platform, "mdl_old", "2026-01-02T00:00:00+00:00", 0.64, passed=True)
    add_model(platform, "mdl_new", 2)
    add_report(platform, "mdl_new", "2026-01-03T00:00:00+00:00", 0.42, passed=False)

    router = ModelRouter(platform.store, platform.audit)

    assert router.resolve("embedding-v1").id == "mdl_old"


def test_activating_a_failed_model_is_refused(platform) -> None:
    add_frozen_set(platform)
    add_model(platform, "mdl_bad", 1)
    add_report(platform, "mdl_bad", "2026-01-03T00:00:00+00:00", 0.083, passed=False)
    router = ModelRouter(platform.store, platform.audit)

    try:
        router.set_active("embedding-v1", "mdl_bad", actor="admin", reason="deploy")
    except PermissionError as error:
        assert "failed its evaluation" in str(error)
    else:
        raise AssertionError("a failed model was activated")
    assert not platform.store.list_audit(action="serving.version_activated")


def test_a_report_on_an_unfrozen_set_is_not_evidence(platform) -> None:
    platform.store.add_eval_set(
        EvalSet(
            id="evs_open",
            name="open",
            version=1,
            frozen=False,
            created_at="2026-01-01T00:00:00+00:00",
            content_hash="c",
        )
    )
    add_model(platform, "mdl_open", 1)
    add_report(
        platform,
        "mdl_open",
        "2026-01-03T00:00:00+00:00",
        0.2,
        passed=False,
        eval_set_id="evs_open",
    )

    assert out_of_service(platform.store, platform.store.get_model("mdl_open")) is None


def test_the_search_endpoint_says_why_nothing_is_in_service(tmp_path: Path) -> None:
    from internal_ai.serving.app import create_serving_app

    platform = Platform.create(root=tmp_path)
    add_frozen_set(platform)
    add_model(platform, "mdl_failed", 1)
    add_report(platform, "mdl_failed", "2026-01-03T00:00:00+00:00", 0.083, passed=False)
    platform.close()

    app = create_serving_app(
        root=tmp_path,
        principals={
            "k": Principal(user_id="alice", groups=frozenset({"engineering"}), api_key="k")
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/search", json={"query": "how do I reset my vpn"}, headers={"X-Internal-Key": "k"}
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "no model is in service"
    assert "mdl_failed" in " ".join(detail["reasons"])
