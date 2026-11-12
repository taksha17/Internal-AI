"""Serving answers through the harness only when the install asks for it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from internal_ai.domain import BudgetScope
from internal_ai.serving.harnessed import (
    HarnessedAnswer,
    harness_enabled,
    harnessed_answer,
    signature_for,
)
from internal_ai.training.alpha.citations import AskHit

MODEL = SimpleNamespace(name="alpha-1", id="mdl_alpha")


def test_the_switch_is_off_unless_a_lab_turns_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERNAL_AI_HARNESS", raising=False)
    assert harness_enabled() is False
    monkeypatch.setenv("INTERNAL_AI_HARNESS", "1")
    assert harness_enabled() is True
    monkeypatch.setenv("INTERNAL_AI_HARNESS", "off")
    assert harness_enabled() is False


def test_a_signature_reads_the_window_from_the_installed_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERNAL_AI_HARNESS_WINDOW", raising=False)
    assert signature_for(MODEL).window_tokens == 8192
    monkeypatch.setenv("INTERNAL_AI_HARNESS_WINDOW", "32768")
    assert signature_for(MODEL).window_tokens == 32768


def test_a_served_answer_leaves_a_digest_a_spend_record_and_an_event(platform) -> None:  # type: ignore[no-untyped-def]
    answer = harnessed_answer(
        platform,
        platform.runtime.llm,
        MODEL,
        "where is ticket 42?",
        [AskHit("a.txt", 3)],
        ["the thing"],
        actor="alice",
    )
    assert isinstance(answer, HarnessedAnswer)
    assert len(answer.prompt_sha256) == 64
    assert answer.prompt_tokens > 0
    assert answer.cost_usd > 0
    assert platform.store.spend_by(BudgetScope.AGENT, "alice") == pytest.approx(answer.cost_usd)
    events = platform.store.list_audit(action="harness.answer", actor="alice")
    assert events and events[0].details["harness"]


def test_a_cap_is_reported_as_too_many_requests_not_an_answer(platform) -> None:  # type: ignore[no-untyped-def]
    platform.budget.engage_kill_switch()
    with pytest.raises(HTTPException) as raised:
        harnessed_answer(
            platform, platform.runtime.llm, MODEL, "where?", [AskHit("a.txt", 3)], ["the thing"]
        )
    assert raised.value.status_code == 429


def test_a_document_posed_as_a_question_is_a_bad_request(platform) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(HTTPException) as raised:
        harnessed_answer(
            platform,
            platform.runtime.llm,
            MODEL,
            "find the clause " * 500,
            [AskHit("a.txt", 3)],
            ["the thing"],
        )
    assert raised.value.status_code == 400
