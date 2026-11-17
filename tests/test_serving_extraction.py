"""The serving guardrail for extraction-shaped traffic: bursts refuse, rephrases pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.config import ENV_EXTRACTION_POLICY
from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.llm import Completion
from internal_ai.serving.app import create_serving_app
from internal_ai.serving.extraction import (
    INSTALL_EXTRACTION_POLICY_KEY,
    ExtractionMonitor,
    check_extraction_policy,
    configured_install_extraction_policy,
    content_words,
    jaccard,
)
from internal_ai.serving.service import Principal, RetrievalHit
from internal_ai.serving.trace import FAILURE_EXTRACTION, list_traces

HIT = {"path": "src/a.txt", "line": 1, "text": "the rotation lives here"}
QUESTION = "where is the rotation"
INSTALL_URL = "/console/models/extraction-default"


def test_content_words_keep_what_carries_the_question() -> None:
    words = content_words("Where is the rotation?")
    assert "rotation" in words
    assert "the" not in words
    assert "is" not in words


def test_jaccard_is_one_for_the_same_question_and_zero_for_disjoint_ones() -> None:
    assert jaccard(content_words(QUESTION), content_words(QUESTION)) == 1.0
    assert jaccard(content_words(QUESTION), content_words("how do refunds work")) == 0.0
    assert jaccard(content_words("what is it"), content_words("what is it")) == 0.0


def test_rephrasings_count_as_similar_and_other_questions_do_not() -> None:
    base = content_words(QUESTION)
    assert jaccard(base, content_words("where is the rotation please")) >= 0.5
    assert jaccard(base, content_words("tell me where the rotation is")) >= 0.5
    assert jaccard(base, content_words("how do refunds work")) < 0.5


def test_the_eighth_near_duplicate_trips_and_a_few_rephrases_do_not() -> None:
    monitor = ExtractionMonitor()
    findings = [monitor.observe("ada", QUESTION) for _ in range(7)]
    assert not any(item["suspect"] for item in findings)

    tripped = monitor.observe("ada", QUESTION)
    assert tripped["suspect"] is True
    assert tripped["similar"] == 8
    assert tripped["window"] == 8

    assert monitor.observe("ada", "how do refunds work")["suspect"] is False


def test_each_user_gets_their_own_window() -> None:
    monitor = ExtractionMonitor()
    for _ in range(8):
        monitor.observe("ada", QUESTION)
    assert monitor.observe("bob", QUESTION)["suspect"] is False


def test_questions_older_than_the_window_fall_out() -> None:
    monitor = ExtractionMonitor(window_seconds=300)
    for second in range(7):
        assert monitor.observe("ada", QUESTION, now=float(second))["suspect"] is False
    fresh = monitor.observe("ada", QUESTION, now=400.0)
    assert fresh["suspect"] is False
    assert fresh["window"] == 1


def test_observe_reports_counts_and_a_flag_and_nothing_else() -> None:
    assert ExtractionMonitor().observe("ada", QUESTION) == {
        "suspect": False,
        "similar": 1,
        "window": 1,
    }


def test_an_unknown_policy_fails_closed() -> None:
    assert check_extraction_policy("warn") == "warn"
    assert check_extraction_policy("refuse") == "refuse"
    assert check_extraction_policy("") == "warn"
    with pytest.raises(ValueError, match="unknown extraction_policy"):
        check_extraction_policy("block")


def test_the_install_policy_comes_from_the_store_first_and_the_environment_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = Platform.create(root=tmp_path)
    try:
        assert configured_install_extraction_policy(platform.store) == "warn"
        monkeypatch.setenv(ENV_EXTRACTION_POLICY, "refuse")
        assert configured_install_extraction_policy(platform.store) == "refuse"
        platform.store.kv_set(INSTALL_EXTRACTION_POLICY_KEY, "warn")
        assert configured_install_extraction_policy(platform.store) == "warn"
    finally:
        platform.close()


class Runner:
    def __init__(self, text: str = "The rotation is in src/a.txt:1.") -> None:
        self.text = text

    def complete(self, prompt: str, system: str = "", **_: object) -> Completion:
        return Completion(text=self.text, model="fake-runner")


def _app(tmp_path, monkeypatch, runner: Runner):
    platform = Platform.create(root=tmp_path)
    platform.store.add_model(
        ModelVersion(
            id="mdl_served",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            status="promoted",
            config={},
        )
    )
    platform.close()
    monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", "http://127.0.0.1:9111")
    import internal_ai.llm as llm_module

    monkeypatch.setattr(llm_module, "LocalRunnerAdapter", lambda **_: runner)
    return create_serving_app(
        root=tmp_path,
        principals={"k": Principal(user_id="ada", groups=frozenset({"engineering"}), api_key="k")},
    )


def _ask(client: TestClient, question: str = QUESTION):
    return client.post(
        "/v1/answer",
        json={"question": question, "hits": [HIT]},
        headers={"X-Internal-Key": "k"},
    )


def _principal() -> Principal:
    return Principal(user_id="ada", groups=frozenset({"engineering"}), api_key="k")


def test_warn_answers_a_burst_and_says_so(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner())
    with TestClient(app) as client:
        responses = [_ask(client) for _ in range(8)]

    assert [item.status_code for item in responses] == [200] * 8
    first, *_, last = (item.json()["extraction_warning"] for item in responses)
    assert first == {"suspect": False, "similar_questions": 1}
    assert last == {"suspect": True, "similar_questions": 8}

    traces = list_traces(app.state.platform.store, _principal(), limit=1)
    assert traces[0].failure == ""
    events = app.state.platform.store.list_audit(action="serving.request", limit=1)
    assert events[0].details["extraction_suspect"] is True
    assert events[0].details["extraction_similar"] == 8


def test_refuse_withholds_a_burst_and_names_it(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner())
    with TestClient(app) as client:
        app.state.platform.store.kv_set(INSTALL_EXTRACTION_POLICY_KEY, "refuse")
        responses = [_ask(client) for _ in range(8)]

    assert [item.status_code for item in responses] == [200] * 7 + [422]
    detail = responses[-1].json()["detail"]
    assert detail["error"] == "answer withheld by extraction guardrail"
    assert detail["similar_questions"] == 8
    assert "text" not in responses[-1].json()

    refused = [
        trace
        for trace in list_traces(app.state.platform.store, _principal(), limit=8)
        if trace.failure == FAILURE_EXTRACTION
    ]
    assert len(refused) == 1
    assert refused[0].answer == ""
    events = app.state.platform.store.list_audit(action="serving.request", limit=8)
    refused_event = next(event for event in events if event.details.get("status_code") == 422)
    assert refused_event.details["extraction_suspect"] is True


def test_distinct_questions_never_trip_the_guardrail(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner())
    questions = [
        "where is the rotation",
        "how do refunds work",
        "who approves a budget increase",
        "what ends a training run early",
        "when does the eval set freeze",
        "which base may this install load",
        "why did promotion refuse",
        "how is spend recorded",
    ]
    with TestClient(app) as client:
        app.state.platform.store.kv_set(INSTALL_EXTRACTION_POLICY_KEY, "refuse")
        responses = [_ask(client, question) for question in questions]

    assert [item.status_code for item in responses] == [200] * 8


def test_an_unrecognised_stored_policy_fails_closed(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner())
    with TestClient(app) as client:
        app.state.platform.store.kv_set(INSTALL_EXTRACTION_POLICY_KEY, "block")
        response = _ask(client)

    assert response.status_code == 500, response.text
    assert "unknown extraction_policy" in response.json()["detail"]


def test_the_audit_event_carries_counts_and_never_the_question(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, Runner())
    question = "where is the zebra rotation"
    with TestClient(app) as client:
        app.state.platform.store.kv_set(INSTALL_EXTRACTION_POLICY_KEY, "refuse")
        for _ in range(8):
            _ask(client, question)

    reopened = Platform.create(root=tmp_path)
    try:
        events = reopened.store.list_audit(action="serving.request", limit=8)
        event = next(item for item in events if item.details.get("status_code") == 422)
    finally:
        reopened.close()
    blob = json.dumps(event.details, sort_keys=True)
    assert question not in blob
    assert "question" not in event.details
    assert event.details["extraction_similar"] == 8


class ChatRetrieval:
    def corpus(self, dataset_id: str):
        return [{"acl": ["engineering"], "sensitivity": "internal"}]

    def keyword_search(self, dataset_id, query, principal, top_k=5):
        return [
            RetrievalHit(
                document_id="runbook#1",
                score=0.9,
                snippet="rotate the creds",
                source_id="runbook",
                path="runbook.md",
                line=3,
            )
        ]


def _chat_app(tmp_path, monkeypatch):
    platform = Platform.create(root=tmp_path)
    platform.store.add_model(
        ModelVersion(
            id="mdl_chat",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            dataset_ids=["ds_chat"],
            status="promoted",
        )
    )
    platform.close()
    monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", "http://127.0.0.1:9111")
    import internal_ai.llm as llm_module

    monkeypatch.setattr(llm_module, "LocalRunnerAdapter", lambda **_: Runner())
    return create_serving_app(
        root=tmp_path,
        principals={"k": Principal(user_id="ada", groups=frozenset({"engineering"}), api_key="k")},
    )


def _chat(client: TestClient, question: str = "rotate the creds"):
    return client.post(
        "/v1/chat/completions",
        json={"model": "embedding-v1", "messages": [{"role": "user", "content": question}]},
        headers={"X-Internal-Key": "k"},
    )


def test_chat_completions_refuses_a_burst(tmp_path, monkeypatch) -> None:
    app = _chat_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        app.state.retrieval = ChatRetrieval()
        app.state.platform.store.kv_set(INSTALL_EXTRACTION_POLICY_KEY, "refuse")
        responses = [_chat(client) for _ in range(8)]

    assert [item.status_code for item in responses] == [200] * 7 + [422]
    assert responses[-1].json()["detail"]["error"] == "answer withheld by extraction guardrail"


def _console_app(tmp_path: Path):
    platform = Platform.create(root=tmp_path, llm_kind="scripted")
    enroll(platform, "Mo Builder", ["ml_lead"])
    enroll(platform, "Sam Sec", ["security"])
    enroll(platform, "Dex Expert", ["domain_expert"])
    platform.close()
    return create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)


def test_security_and_ml_lead_set_the_policy_and_others_cannot(tmp_path: Path) -> None:
    app = _console_app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Sam Sec")
        page = client.post(
            INSTALL_URL, data={"policy": "refuse", "csrf_token": session_csrf(client)}
        )
        assert page.status_code == 200
        assert "extraction-shaped questions now refuse across the install" in page.text
        assert app.state.platform.store.kv_get(INSTALL_EXTRACTION_POLICY_KEY) == "refuse"

    app = _console_app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Mo Builder")
        page = client.post(INSTALL_URL, data={"policy": "warn", "csrf_token": session_csrf(client)})
        assert "extraction-shaped questions now warn across the install" in page.text
        events = app.state.platform.store.list_audit(
            action="serving.extraction_default_set", limit=5
        )
        assert [event.details["after"] for event in reversed(events)] == ["refuse", "warn"]

    app = _console_app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Dex Expert")
        page = client.post(
            INSTALL_URL, data={"policy": "refuse", "csrf_token": session_csrf(client)}
        )
        assert "needs ml_lead or security" in page.text
        store = app.state.platform.store
        assert store.kv_get(INSTALL_EXTRACTION_POLICY_KEY) == "warn"
        denied = store.list_audit(action="console.denied", limit=5)
        assert denied[0].details["action"] == "set_extraction_policy"


def test_an_unknown_extraction_policy_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    app = _console_app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Sam Sec")
        page = client.post(
            INSTALL_URL, data={"policy": "block", "csrf_token": session_csrf(client)}
        )
        assert "unknown extraction_policy" in page.text
        assert app.state.platform.store.kv_get(INSTALL_EXTRACTION_POLICY_KEY) is None


def test_the_form_shows_for_security_and_not_for_domain_experts(tmp_path: Path) -> None:
    app = _console_app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Sam Sec")
        assert f'action="{INSTALL_URL}"' in client.get("/console/models").text

    app = _console_app(tmp_path)
    with TestClient(app) as client:
        sign_in(client, "Dex Expert")
        assert f'action="{INSTALL_URL}"' not in client.get("/console/models").text
