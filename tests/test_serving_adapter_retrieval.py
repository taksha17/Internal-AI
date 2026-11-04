"""B2: a generator and a retriever are different things, and serving must say so.

All three retrieval-shaped routes built their embedder from `model.path`, so an
adapter model — whose path is a PEFT directory — crashed instead of answering,
and the unit tests never saw it because they stubbed the embedder with a bare
object. These tests run a real adapter directory, no embedder stand-in, and the
real retrieval service.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from test_alpha_adapt import tiny_snapshot

from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion, ModelVersion
from internal_ai.ids import now_iso, tree_hash
from internal_ai.serving.app import create_serving_app
from internal_ai.serving.service import Principal

pytest.importorskip("torch", reason="the adapter directory is a PEFT tree")
pytest.importorskip("transformers", reason="the adapter directory is a PEFT tree")
pytest.importorskip("peft", reason="the adapter directory is a PEFT tree")


class _Runner(BaseHTTPRequestHandler):
    health: ClassVar[dict[str, Any]] = {}
    answer_text: ClassVar[str] = "The rotation steps are in the runbook record."

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        body = json.dumps(self.health).encode("utf-8")
        self._send(body)

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send(
            json.dumps(
                {
                    "model": "runner",
                    "choices": [{"message": {"content": self.answer_text}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 9},
                }
            ).encode("utf-8")
        )

    def _send(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    adapter_holder: dict[str, str] = {}

    class Handler(_Runner):
        pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "INTERNAL_AI_LOCAL_RUNNER", f"http://127.0.0.1:{server.server_address[1]}/v1"
    )
    yield Handler, adapter_holder
    server.shutdown()


def adapter_directory(tmp_path: Path) -> Path:
    """What promotion saves: a real PEFT tree beside a real pinned snapshot."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    tiny_snapshot(tmp_path)
    snapshot = tmp_path / "Example--tiny" / "rev1"
    adapted = get_peft_model(
        AutoModelForCausalLM.from_pretrained(snapshot),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    directory = tmp_path / "adapter"
    adapted.save_pretrained(directory)
    return directory


def served(tmp_path: Path, adapter: Path) -> str:
    platform = Platform.create(root=tmp_path)
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_ret",
            name="runbooks",
            version=1,
            content_hash="hash-ret",
            record_count=2,
            token_count=8,
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(
        "ds_ret",
        [
            {
                "record_key": "runbook.md:3",
                "payload": "the rotation steps for the ledger feed are in the runbook",
                "source_id": "runbook",
                "acl": ["engineering"],
            },
            {
                "record_key": "misc.md:1",
                "payload": "the kettle in the kitchen boils water",
                "source_id": "misc",
                "acl": ["engineering"],
            },
        ],
    )
    platform.store.add_model(
        ModelVersion(
            id="mdl_alpha",
            name="tiny-alpha",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            dataset_ids=["ds_ret"],
            status="promoted",
            path=str(adapter),
            card={
                "registered_by": "ppl_lead",
                "base_repo_id": "Example/tiny",
                "base_revision": "rev1",
                "adapter_backend": "peft",
                "adapter_sha256": tree_hash(adapter),
                "trained_from_scratch": False,
            },
        )
    )
    platform.close()
    return tree_hash(adapter)


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_serving_app(
            root=tmp_path,
            principals={
                "k": Principal(user_id="ada", groups=frozenset({"engineering"}), api_key="k")
            },
        )
    )


def test_adapter_search_answers_with_keywords_and_no_embedder(tmp_path: Path, runner) -> None:
    handler, _ = runner
    handler.health = {"status": "ok"}
    adapter = adapter_directory(tmp_path)
    served(tmp_path, adapter)

    with _client(tmp_path) as client:
        response = client.post(
            "/v1/search",
            json={"query": "rotation steps ledger", "dataset_id": "ds_ret", "top_k": 2},
            headers={"X-Internal-Key": "k"},
        )

    assert response.status_code == 200, response.text
    hits = response.json()["hits"]
    assert hits, "keyword search found no document"
    assert hits[0]["document_id"] == "runbook.md:3"


def test_adapter_chat_cites_from_keyword_retrieval(tmp_path: Path, runner) -> None:
    handler, _ = runner
    adapter = adapter_directory(tmp_path)
    digest = served(tmp_path, adapter)
    handler.health = {"status": "ok", "adapter_sha256": digest}

    with _client(tmp_path) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny-alpha",
                "messages": [{"role": "user", "content": "where are the rotation steps?"}],
            },
            headers={"X-Internal-Key": "k"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"]
    assert body["citation_passed"] is True


def test_embed_refuses_a_generator_instead_of_crashing(tmp_path: Path, runner) -> None:
    handler, _ = runner
    handler.health = {"status": "ok"}
    adapter = adapter_directory(tmp_path)
    served(tmp_path, adapter)

    with _client(tmp_path) as client:
        response = client.post(
            "/v1/embed",
            json={"query": "rotation steps", "dataset_id": "ds_ret"},
            headers={"X-Internal-Key": "k"},
        )

    assert response.status_code == 409, response.text
    assert "no embedder" in response.json()["detail"]
