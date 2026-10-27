"""Policy P2: raw internal data never reaches a third-party LLM API.

The external adapter redacts secret- and PII-shaped spans from every prompt
and system message before the request leaves the process.
"""

from __future__ import annotations

import httpx

from internal_ai.llm import AnthropicAdapter, redact_for_external


def test_redaction_labels_each_hit() -> None:
    text, count = redact_for_external(
        "key AKIAIOSFODNN7EXAMPLE and person@example.com called 415-555-0134"
    )
    assert count == 3
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "person@example.com" not in text
    assert "415-555-0134" not in text
    assert "[redacted:aws_access_key]" in text
    assert "[redacted:email]" in text
    assert "[redacted:phone]" in text


def test_clean_code_passes_through() -> None:
    prompt = "Write a Python function that retries with exponential backoff."
    text, count = redact_for_external(prompt)
    assert text == prompt
    assert count == 0


def test_the_wire_payload_is_redacted(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    sent: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"content": [{"type": "text", "text": "{}"}], "usage": {}}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
            return None

        def post(self, url: str, json: dict, headers: dict) -> FakeResponse:
            sent.update(json)
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    adapter = AnthropicAdapter(model="x", api_key="test-key")
    adapter.complete("debug this: password = 's3cret-value-here' from ticket 7")

    body = sent["messages"][0]["content"]
    assert "s3cret-value-here" not in body
    assert "[redacted:password_assignment]" in body
