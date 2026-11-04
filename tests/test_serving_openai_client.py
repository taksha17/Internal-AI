"""B8: an OpenAI-shaped client gets what it asked for, or is told it may not.

The client here is the official ``openai`` package talking HTTP to a running
server, because the complaint was that the endpoint is narrower than its shape
suggests: parameters were dropped silently, only the last message was read, and
the usage numbers were word counts.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openai")
from openai import BadRequestError, OpenAI

from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.llm import Completion
from internal_ai.serving.app import create_serving_app
from internal_ai.serving.service import Principal

GROUNDED = "The rotation steps are in the runbook record."
KEY = "openai-test-key"
PAYLOAD = "The rotation steps are written down in the runbook record; rotate the creds weekly."


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, system: str = "", **kwargs: Any) -> Completion:
        self.calls.append({"prompt": prompt, "system": system, **kwargs})
        return Completion(text=GROUNDED, model="fake-runner")


def _app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str = ""
) -> tuple[Any, RecordingRunner]:
    platform = Platform.create(root=tmp_path)
    platform.store.add_model(
        ModelVersion(
            id="mdl_openai",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="2026-01-01T00:00:00+00:00",
            dataset_ids=["ds_openai"],
            path=path,
            status="promoted",
        )
    )
    platform.store.add_dataset(
        DatasetVersion(
            id="ds_openai",
            name="runbook",
            version=1,
            content_hash="hash-openai",
            record_count=1,
            token_count=12,
            created_at=now_iso(),
            released=True,
        )
    )
    platform.store.add_dataset_records(
        "ds_openai",
        [
            {
                "record_key": "runbook.md:1",
                "payload": PAYLOAD,
                "source_id": "runbook",
                "acl": [],
                "sensitivity": "internal",
            }
        ],
    )
    platform.close()
    monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", "http://127.0.0.1:9111")
    import internal_ai.llm as llm_module

    runner = RecordingRunner()
    monkeypatch.setattr(llm_module, "LocalRunnerAdapter", lambda **_: runner)
    app = create_serving_app(
        root=tmp_path,
        principals={KEY: Principal(user_id="ada", groups=frozenset({"engineering"}), api_key=KEY)},
    )
    return app, runner


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def running(app: Any) -> Iterator[OpenAI]:
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline and thread.is_alive():
        time.sleep(0.02)
    if not server.started:
        raise AssertionError("the serving app did not start")
    try:
        yield OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=KEY, max_retries=0)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def ask(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"model": "embedding-v1", "messages": messages}


def test_the_client_gets_a_system_prompt_and_a_two_turn_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, runner = _app(tmp_path, monkeypatch)
    with running(app) as client:
        client.chat.completions.create(
            **ask(
                [
                    {"role": "system", "content": "You answer for the platform on-call desk."},
                    {"role": "user", "content": "how do we rotate the creds?"},
                    {"role": "assistant", "content": "the runbook says to rotate them weekly"},
                    {"role": "user", "content": "and where are the steps written down?"},
                ]
            )
        )

    prompt = runner.calls[0]["prompt"]
    assert "You answer for the platform on-call desk." in prompt
    assert "how do we rotate the creds?" in prompt
    assert "the runbook says to rotate them weekly" in prompt
    assert "and where are the steps written down?" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_max_tokens_and_temperature_reach_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, runner = _app(tmp_path, monkeypatch)
    with running(app) as client:
        client.chat.completions.create(
            **ask([{"role": "user", "content": "rotate the creds"}]),
            max_tokens=77,
            temperature=0.3,
        )

    assert runner.calls[0]["max_tokens"] == 77
    assert runner.calls[0]["temperature"] == 0.3


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        ({"stop": ["END"]}, "stop"),
        ({"seed": 7}, "seed"),
        ({"n": 2}, "n"),
    ],
)
def test_an_unsupported_parameter_is_rejected_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any], named: str
) -> None:
    app, runner = _app(tmp_path, monkeypatch)
    with running(app) as client, pytest.raises(BadRequestError) as refused:
        client.chat.completions.create(
            **ask([{"role": "user", "content": "rotate the creds"}]), **extra
        )

    assert named in json.dumps(refused.value.response.json())
    assert not runner.calls


def test_usage_counts_come_from_the_tokenizer_that_sits_beside_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    words = ["[UNK]", *sorted(set(f"{GROUNDED} {PAYLOAD}".lower().split()))]
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocab={word: index for index, word in enumerate(words)}, unk_token="[UNK]"
        )
    )
    from tokenizers import pre_tokenizers

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    model_dir = tmp_path / "served"
    model_dir.mkdir()
    tokenizer.save(str(model_dir / "tokenizer.json"))

    app, runner = _app(tmp_path, monkeypatch, path=str(model_dir))
    with running(app) as client:
        answer = client.chat.completions.create(
            **ask([{"role": "user", "content": "rotate the creds"}])
        )

    prompt = runner.calls[0]["prompt"]
    assert answer.usage.prompt_tokens == len(tokenizer.encode(prompt).ids)
    assert answer.usage.completion_tokens == len(tokenizer.encode(GROUNDED).ids)
    assert answer.usage.total_tokens == answer.usage.prompt_tokens + answer.usage.completion_tokens
