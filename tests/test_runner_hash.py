"""B1, the runner half: the process answering must report the adapter it loaded.

`serve_alpha()` writes a manifest and nothing else, so the supported route to a
promoted adapter is an external runner behind `INTERNAL_AI_LOCAL_RUNNER` — and
nothing used to check what that process was actually running. A runner now
answers `/health` with the adapter hash it loaded, and the platform refuses to
route when the report is missing or names a different adapter than the card.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

from internal_ai.domain import ModelVersion
from internal_ai.ids import now_iso
from internal_ai.training.alpha.local import answer_adapter

DIGEST = "a" * 64


def card_model(model_id: str = "mdl_runner", **card: Any) -> ModelVersion:
    values: dict[str, Any] = {"base_repo_id": "Example/tiny", "base_revision": "rev1"}
    values.update(card)
    return ModelVersion(
        id=model_id,
        name="tiny-alpha",
        version=1,
        content_hash="h",
        created_at=now_iso(),
        status="candidate",
        config={},
        card=values,
        path="",
    )


class _Runner(BaseHTTPRequestHandler):
    health: ClassVar[dict[str, Any]] = {}
    answer_text: ClassVar[str] = "the ledger failed (runner)"

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._json(200, self.health)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._json(
            200,
            {
                "model": "runner",
                "choices": [{"message": {"content": self.answer_text}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
        )

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch):
    def serve(health: dict[str, Any] | None):
        class Handler(_Runner):
            pass

        Handler.health = health if health is not None else {}
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", url)
        return server

    started: list[ThreadingHTTPServer] = []

    def start(health: dict[str, Any] | None):
        started.append(serve(health))
        return started[-1]

    yield start
    for server in started:
        server.shutdown()


def test_a_runner_that_reports_the_promoted_hash_is_used(platform, runner) -> None:
    runner({"status": "ok", "adapter_sha256": DIGEST})
    adapter = answer_adapter(platform, card_model(adapter_sha256=DIGEST))

    assert adapter.name == "local-runner"
    completion = adapter.complete("Question: the ledger failed\nAnswer:")
    assert completion.text == "the ledger failed (runner)"


def test_a_runner_reporting_a_different_hash_is_refused(platform, runner) -> None:
    runner({"status": "ok", "adapter_sha256": "b" * 64})

    with pytest.raises(PermissionError, match="different adapter") as refused:
        answer_adapter(platform, card_model(adapter_sha256=DIGEST))

    assert DIGEST[:16] in str(refused.value)


def test_a_runner_that_reports_no_adapter_is_refused(platform, runner) -> None:
    runner({"status": "ok"})

    with pytest.raises(PermissionError, match="does not report"):
        answer_adapter(platform, card_model(adapter_sha256=DIGEST))


def test_a_runner_that_cannot_be_asked_is_refused(platform, runner) -> None:
    runner(None)  # /health answers 404

    with pytest.raises(PermissionError, match="does not report"):
        answer_adapter(platform, card_model(adapter_sha256=DIGEST))


def test_a_model_with_no_promoted_adapter_keeps_the_plain_runner_path(platform, runner) -> None:
    runner({"status": "ok", "adapter_sha256": DIGEST})

    adapter = answer_adapter(platform, card_model())

    assert adapter.name == "local-runner"
