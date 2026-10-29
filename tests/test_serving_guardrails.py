"""The answer endpoint keeps its output checked and its load bounded."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from internal_ai.config import ENV_GUARDRAIL_POLICY
from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.llm import Completion
from internal_ai.serving.app import create_serving_app, validate_output_guardrails
from internal_ai.serving.service import Principal

HIT = {"path": "src/a.txt", "line": 1, "text": "the password rotation lives here"}
ANSWER = "The password rotation is in src/a.txt:1."


class Runner:
    def __init__(self, text: str = ANSWER, delay: float = 0.0, tracker: Tracker | None = None):
        self.text = text
        self.delay = delay
        self.tracker = tracker

    def complete(self, prompt: str, system: str = "", **_: object) -> Completion:
        if self.tracker is not None:
            self.tracker.enter()
        try:
            if self.delay:
                time.sleep(self.delay)
            return Completion(text=self.text, model="fake-runner")
        finally:
            if self.tracker is not None:
                self.tracker.exit()


class Tracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.most_at_once = 0

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            self.most_at_once = max(self.most_at_once, self.current)

    def exit(self) -> None:
        with self.lock:
            self.current -= 1


def _app(
    tmp_path, monkeypatch, runner: Runner, config: dict | None = None, max_concurrent: int = 5
):
    platform = Platform.create(root=tmp_path)
    platform.store.add_model(
        ModelVersion(
            id="mdl_served",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            status="promoted",
            config=config or {},
        )
    )
    platform.close()
    monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", "http://127.0.0.1:9111")
    monkeypatch.setenv("INTERNAL_AI_MAX_CONCURRENT", str(max_concurrent))
    import internal_ai.llm as llm_module

    monkeypatch.setattr(llm_module, "LocalRunnerAdapter", lambda **_: runner)
    return create_serving_app(
        root=tmp_path,
        principals={"k": Principal(user_id="ada", groups=frozenset({"engineering"}), api_key="k")},
    )


def _ask(client: TestClient) -> object:
    return client.post(
        "/v1/answer",
        json={"question": "where is the rotation", "hits": [HIT]},
        headers={"X-Internal-Key": "k"},
    )


def test_the_guardrail_scan_flags_text_that_names_a_secret() -> None:
    assert validate_output_guardrails("the password is in the vault") == [
        "potential password disclosure"
    ]
    assert validate_output_guardrails("rotate the API-key tonight") == [
        "potential API key disclosure"
    ]
    assert validate_output_guardrails("this file is confidential") == ["confidential term used"]
    assert validate_output_guardrails("the reset steps are in src/a.txt:1") == []


def test_an_answer_that_names_a_secret_keeps_the_violation_on_record(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner())
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 200, response.text
    assert response.json()["guardrail_violations"] == ["potential password disclosure"]

    reopened = Platform.create(root=tmp_path)
    try:
        event = reopened.store.list_audit(action="serving.request", limit=5)[0]
    finally:
        reopened.close()
    assert event.details["guardrail_violations"] == ["potential password disclosure"]


def test_a_refusing_model_withholds_a_flagged_answer(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner(), config={"guardrail_policy": "refuse"})
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "answer withheld by guardrail policy"
    assert detail["violations"] == ["potential password disclosure"]
    assert "text" not in response.json()

    reopened = Platform.create(root=tmp_path)
    try:
        event = reopened.store.list_audit(action="serving.request", limit=5)[0]
    finally:
        reopened.close()
    assert event.details["status_code"] == 422
    assert event.details["guardrail_policy"] == "refuse"
    assert event.details["guardrail_violations"] == ["potential password disclosure"]


def test_a_refusing_model_still_answers_clean_text(tmp_path, monkeypatch) -> None:
    app = _app(
        tmp_path,
        monkeypatch,
        Runner(text="The reset steps are in src/a.txt:1."),
        config={"guardrail_policy": "refuse"},
    )
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 200, response.text
    assert response.json()["guardrail_violations"] == []


def test_an_unrecognised_policy_fails_closed(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner(), config={"guardrail_policy": "Refuse"})
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 500, response.text
    assert "unknown guardrail_policy" in response.json()["detail"]
    assert "text" not in response.json()


def test_the_install_default_applies_when_a_model_does_not_choose(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_GUARDRAIL_POLICY, "refuse")
    app = _app(tmp_path, monkeypatch, Runner())
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["violations"] == ["potential password disclosure"]
    assert "text" not in response.json()


def test_a_model_can_still_choose_its_own_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_GUARDRAIL_POLICY, "refuse")
    app = _app(tmp_path, monkeypatch, Runner(), config={"guardrail_policy": "warn"})
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 200, response.text
    assert response.json()["guardrail_violations"] == ["potential password disclosure"]


def test_an_unrecognised_install_default_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_GUARDRAIL_POLICY, "block")
    app = _app(tmp_path, monkeypatch, Runner())
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 500, response.text
    assert "unknown guardrail_policy" in response.json()["detail"]


def test_an_answer_slower_than_the_target_is_refused(tmp_path, monkeypatch) -> None:
    app = _app(
        tmp_path,
        monkeypatch,
        Runner(delay=0.02),
        config={"latency_target_ms": 1},
    )
    with TestClient(app) as client:
        response = _ask(client)

    assert response.status_code == 408, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "latency target exceeded"
    assert detail["latency_target_ms"] == 1


def test_the_load_ceiling_keeps_answers_from_overlapping(tmp_path, monkeypatch) -> None:
    tracker = Tracker()
    app = _app(tmp_path, monkeypatch, Runner(delay=0.05, tracker=tracker), max_concurrent=1)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda _: _ask(client), range(3)))

    assert [item.status_code for item in responses] == [200, 200, 200]
    assert tracker.most_at_once == 1
