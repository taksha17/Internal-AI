"""An OpenAI-shaped request is answered under the same rules, streamed or whole."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.llm import Completion
from internal_ai.serving.app import create_serving_app
from internal_ai.serving.service import Principal, RetrievalHit

GROUNDED = "The rotation steps are in the runbook record."
MISCITED = "The rotation steps are in src/b.txt:9."
ASK = {"model": "embedding-v1", "messages": [{"role": "user", "content": "rotate the creds"}]}


class Runner:
    def __init__(self, text: str = GROUNDED) -> None:
        self.text = text

    def complete(self, prompt: str, system: str = "", **_: object) -> Completion:
        return Completion(text=self.text, model="fake-runner")


class Retrieval:
    """A retrieval stand-in; the ACL rules themselves are tested elsewhere."""

    def __init__(self, hits: list[RetrievalHit], rows: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits
        self.rows = (
            rows if rows is not None else [{"acl": ["engineering"], "sensitivity": "internal"}]
        )

    def corpus(self, dataset_id: str) -> list[dict[str, Any]]:
        return self.rows

    def search(
        self,
        embedder: object,
        dataset_id: str,
        query: str,
        principal: Principal,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        return self.hits


def _app(tmp_path: Any, monkeypatch: Any, text: str = GROUNDED) -> tuple[Any, Retrieval]:
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

    monkeypatch.setattr(llm_module, "LocalRunnerAdapter", lambda **_: Runner(text))
    app = create_serving_app(
        root=tmp_path,
        principals={"k": Principal(user_id="ada", groups=frozenset({"engineering"}), api_key="k")},
    )
    hits = [
        RetrievalHit(
            document_id="ds_chat#1", score=0.9, snippet="rotate the creds", source_id="runbook"
        )
    ]
    return app, Retrieval(hits)


@contextmanager
def _open(app: Any, retrieval: Retrieval) -> Iterator[TestClient]:
    with TestClient(app) as client:
        app.state.retrieval = retrieval
        app.state.embedders[""] = object()
        yield client


def _frames(body: str) -> list[str]:
    return [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]


def test_a_chat_completion_has_the_openai_shape(tmp_path, monkeypatch) -> None:
    app, retrieval = _app(tmp_path, monkeypatch)
    with _open(app, retrieval) as client:
        response = client.post("/v1/chat/completions", json=ASK, headers={"X-Internal-Key": "k"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "embedding-v1"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": GROUNDED},
            "finish_reason": "stop",
        }
    ]
    assert body["usage"]["total_tokens"] >= 1
    assert body["citation_passed"] is True


def test_a_streamed_completion_ends_with_done_and_adds_up_to_the_answer(
    tmp_path, monkeypatch
) -> None:
    app, retrieval = _app(tmp_path, monkeypatch)
    with _open(app, retrieval) as client:
        response = client.post(
            "/v1/chat/completions", json={**ASK, "stream": True}, headers={"X-Internal-Key": "k"}
        )

    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _frames(response.text)
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    content = "".join(item["choices"][0]["delta"].get("content", "") for item in chunks)
    assert content == GROUNDED


def test_a_citation_failure_closes_the_stream_with_an_error_frame(tmp_path, monkeypatch) -> None:
    app, retrieval = _app(tmp_path, monkeypatch, text=MISCITED)
    with _open(app, retrieval) as client:
        response = client.post(
            "/v1/chat/completions", json={**ASK, "stream": True}, headers={"X-Internal-Key": "k"}
        )

    frames = _frames(response.text)
    assert "[DONE]" not in frames
    assert json.loads(frames[-1])["error"]["type"] == "citation_failed"

    reopened = Platform.create(root=tmp_path)
    try:
        event = reopened.store.list_audit(action="serving.request", limit=5)[0]
    finally:
        reopened.close()
    assert event.details["citation_passed"] is False
    assert event.details["stream_outcome"] == "refused"
    assert event.details["status_code"] == 200


def test_a_dataset_this_key_cannot_read_is_refused(tmp_path, monkeypatch) -> None:
    app, _retrieval = _app(tmp_path, monkeypatch)
    blocked = Retrieval([], rows=[{"acl": ["legal"], "sensitivity": "internal"}])
    with _open(app, blocked) as client:
        response = client.post("/v1/chat/completions", json=ASK, headers={"X-Internal-Key": "k"})

    assert response.status_code == 403
    assert "may be read with this key" in response.json()["detail"]


def test_a_named_model_is_never_swapped_for_the_served_one(tmp_path, monkeypatch) -> None:
    app, retrieval = _app(tmp_path, monkeypatch)
    with _open(app, retrieval) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**ASK, "model": "not-a-model"},
            headers={"X-Internal-Key": "k"},
        )

    assert response.status_code == 404
    assert "not-a-model" in response.json()["detail"]


def test_a_client_that_walks_away_mid_stream_still_leaves_an_event(tmp_path, monkeypatch) -> None:
    app, retrieval = _app(tmp_path, monkeypatch)
    with (
        _open(app, retrieval) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={**ASK, "stream": True},
            headers={"X-Internal-Key": "k"},
        ) as response,
    ):
        assert response.status_code == 200
        lines = response.iter_lines()
        next(lines)

    reopened = Platform.create(root=tmp_path)
    try:
        event = reopened.store.list_audit(action="serving.request", limit=5)[0]
    finally:
        reopened.close()
    assert event.details["status_code"] == 499
    assert event.details["completed"] is False
