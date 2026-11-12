"""Answer through the harness, when an install has asked for it.

Serving answered through ``cited_answer`` before the harness existed, and an eval
report measures that path. Turning the harness on is therefore an install
decision (``INTERNAL_AI_HARNESS=1``) rather than a default that silently changes
what a score means. When it is on, a request takes the same conditioned prompt
and one attempt — serving never repaired, so enabling the harness does not add a
second generation — and leaves a digest, a latency, a spend record and a tool
count beside the answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import HTTPException

from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.harness import (
    BackendChoice,
    BudgetRefused,
    Harness,
    HarnessDeadlineExceeded,
    HarnessRequest,
    ModelSignature,
    PromptTooLong,
    RequestRefused,
)
from internal_ai.harness.backends import is_external
from internal_ai.llm import LLMAdapter
from internal_ai.training.alpha.citations import AskHit

ENV_HARNESS = "INTERNAL_AI_HARNESS"
ENV_WINDOW = "INTERNAL_AI_HARNESS_WINDOW"
DEFAULT_WINDOW_TOKENS = 8192
"""What a served model is assumed to carry when its config does not say."""

_ON = frozenset({"1", "true", "on", "yes"})


def harness_enabled(source: dict[str, str] | None = None) -> bool:
    """Off unless a lab turns it on, because it changes what a report measures."""
    env = source if source is not None else os.environ
    return env.get(ENV_HARNESS, "0").strip().lower() in _ON


def signature_for(model: ModelVersion) -> ModelSignature:
    raw = os.environ.get(ENV_WINDOW, "").strip()
    window = int(raw) if raw.isdigit() else DEFAULT_WINDOW_TOKENS
    return ModelSignature(name=model.name or model.id, window_tokens=window)


@dataclass(frozen=True, slots=True)
class HarnessedAnswer:
    """What serving needs from an answer, plus what the harness measured."""

    text: str
    passed: bool
    model: str
    mismatches: tuple[str, ...]
    prompt_tokens: int = 0
    cost_usd: float = 0.0
    prompt_sha256: str = ""
    latency_ms: int = 0
    attempts: int = 0


def harnessed_answer(
    platform: Platform,
    adapter: LLMAdapter,
    model: ModelVersion,
    question: str,
    hits: list[AskHit],
    excerpts: list[str],
    *,
    training_texts: tuple[str, ...] = (),
    system: str = "",
    history: tuple[tuple[str, str], ...] = (),
    max_tokens: int = 400,
    temperature: float = 0.0,
    actor: str = "serving",
) -> HarnessedAnswer:
    """Answer once, through the harness, with the refusals mapped to HTTP."""
    harness = Harness(
        BackendChoice(adapter.name, adapter, is_external(adapter)),
        signature_for(model),
        max_attempts=1,
        training_texts=training_texts,
        audit=platform.audit,
        budget=platform.budget,
        actor=actor,
        actor_kind="human",
    )
    request = HarnessRequest(
        question=question,
        hits=tuple(hits),
        excerpts=tuple(excerpts),
        system=system,
        history=history,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        answer = harness.answer(request)
    except BudgetRefused as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except HarnessDeadlineExceeded as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (PromptTooLong, RequestRefused) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return HarnessedAnswer(
        text=answer.text,
        passed=answer.passed,
        model=answer.model,
        mismatches=answer.problems,
        prompt_tokens=answer.prompt_tokens,
        cost_usd=answer.cost_usd,
        prompt_sha256=answer.prompt_sha256,
        latency_ms=answer.latency_ms,
        attempts=answer.attempts,
    )
