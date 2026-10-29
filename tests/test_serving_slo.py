"""Serving reports its latency, stays up under a burst, and does not log the question."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from internal_ai.domain import AuditEvent, AuditOutcome
from internal_ai.ids import new_id
from internal_ai.serving.app import create_serving_app
from internal_ai.serving.report import serve_report
from internal_ai.serving.service import Principal, log_request


def _event(platform, latency: float, cost: float, created_at: str, citation: bool = True) -> None:
    platform.store.append_audit(
        AuditEvent(
            id=new_id("aud"),
            seq=0,
            actor="ada",
            actor_kind="human",
            action="serving.request",
            subject="/v1/answer",
            outcome=AuditOutcome.SUCCESS,
            details={"latency_ms": latency, "citation_passed": citation, "hits": 1, "model": "m"},
            cost_usd=cost,
            created_at=created_at,
        )
    )


def test_the_report_prints_percentiles_and_cost(platform) -> None:
    platform.store.conn.execute(
        "INSERT INTO project_brief (id, payload, updated_at, updated_by) VALUES (?,?,?,?)",
        (
            "brief",
            json.dumps({"latency_ms": 50, "timeline": "30 days"}),
            "2026-10-01T00:00:00+00:00",
            "test",
        ),
    )
    now = datetime.now(UTC)
    _event(platform, 10, 0.001, now.isoformat())
    _event(platform, 20, 0.003, (now - timedelta(minutes=1)).isoformat())
    _event(platform, 40, 0.002, (now - timedelta(days=40)).isoformat())

    dry = serve_report(platform, apply=False)
    assert dry["p50_ms"] == 20
    assert dry["p95_ms"] == 40
    assert dry["cost_per_1000"] == 2.0
    assert dry["within_brief"] is True
    assert dry["past_retention"] == 1

    applied = serve_report(platform, apply=True)
    assert applied["past_retention"] == 0
    assert applied["requests"] == 2
    again = serve_report(platform, apply=False)
    assert again["requests"] == 2
    assert again["p95_ms"] == 20


def test_a_burst_of_health_and_models_has_no_500_and_one_429(tmp_path) -> None:
    app = create_serving_app(
        root=tmp_path,
        rate_limit_per_minute=60,
        principals={"k": Principal(user_id="ada", groups=frozenset({"engineering"}), api_key="k")},
    )
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=40) as pool:
            health = list(pool.map(lambda _: client.get("/health"), range(40)))
        codes = []
        for _ in range(70):
            codes.append(client.get("/v1/models", headers={"X-Internal-Key": "k"}).status_code)

    assert {item.status_code for item in health} == {200}
    assert 500 not in codes
    assert 429 in codes


def test_the_audit_row_does_not_contain_the_question(platform) -> None:
    question = "where is the razorpay host configured"
    log_request(
        platform.audit,
        Principal(user_id="ada", groups=frozenset(), api_key="k"),
        "/v1/answer",
        200,
        12.0,
        {"hits": 1, "question": question, "note": f"asked {question}"},
        question=question,
    )
    recorded = platform.store.list_audit(action="serving.request", limit=5)[0]
    blob = json.dumps(recorded.details)
    assert question not in blob
    assert "question" not in recorded.details
