"""Prompt conditioning: one job, sized to the model that has to do it.

The prompt text is not this module's to invent. ``pairs`` holds the one wording
the adapter was trained and scored in, so a second wording is a new measured
artifact rather than a setting. What arrives here is the client's context; what
leaves is either a prompt that fits the model's window or a refusal that says how
far over it is. Shrinking the question quietly would make the answer look
successful while measuring a different one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from internal_ai.training.alpha.answer import with_context
from internal_ai.training.alpha.pairs import (
    CANDIDATE_LIMIT,
    prompt_for,
    prompt_for_candidates,
)

ANSWER_RESERVE_TOKENS = 512
"""Room kept for the answer, so a prompt that just fits cannot crowd it out."""

SMALL_WINDOW = 8192

TokenCounter = Callable[[str], int]


def estimated_tokens(text: str) -> int:
    """A conservative count for an install with no tokenizer for the model."""
    return int(len(text.split()) * 1.4)


@dataclass(frozen=True, slots=True)
class ModelSignature:
    """What the harness needs to know about the model, and nothing more."""

    name: str
    window_tokens: int
    family: str = "open-weights"

    def __post_init__(self) -> None:
        if self.window_tokens < 512:
            raise ValueError(
                f"{self.name} declares a {self.window_tokens}-token window; a model that "
                "small cannot hold the prompt this install asks in"
            )


def preamble_budget(signature: ModelSignature) -> int:
    """How much client context a model of this window may be asked to carry.

    A 7B model handed a flagship's preamble spends its window on instructions and
    has none left for the answer, so the budget follows the window instead of
    being one number that suits the largest model.
    """
    if signature.window_tokens <= SMALL_WINDOW:
        return 300
    if signature.window_tokens <= 32768:
        return 600
    return 1200


@dataclass(frozen=True, slots=True)
class PromptVariant:
    """A prompt this install may ask in, and where its text lives."""

    name: str
    source: str


VARIANTS: dict[str, PromptVariant] = {
    "selection-v1": PromptVariant("selection-v1", "pairs.prompt_for_candidates"),
}
"""One entry on purpose. A second variant is a new frozen run, not a config value."""


@dataclass(frozen=True, slots=True)
class ConditionedPrompt:
    text: str
    variant: str
    prompt_tokens: int
    preamble_tokens: int


class PromptTooLong(ValueError):
    """The request does not fit, and trimming it would change the question."""


def variant_for(name: str) -> PromptVariant:
    variant = VARIANTS.get(name)
    if variant is None:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError(
            f"unknown prompt variant {name!r}: this install asks in {known}, and a new "
            "variant is a measured artifact before it is a setting"
        )
    return variant


def candidates_prompt(question: str, candidates: Sequence[tuple[str, str]]) -> str:
    """The measured prompt for these candidates, refusing a list it never saw."""
    if len(candidates) > CANDIDATE_LIMIT:
        raise ValueError(
            f"{len(candidates)} candidates were offered; the prompt was measured with at "
            f"most {CANDIDATE_LIMIT}, and a longer list is a different question"
        )
    if candidates:
        return prompt_for_candidates(question, list(candidates))
    return prompt_for(question)


def condition(
    question: str,
    candidates: Sequence[tuple[str, str]],
    signature: ModelSignature,
    *,
    system: str = "",
    history: Sequence[tuple[str, str]] = (),
    variant: str = "selection-v1",
    counter: TokenCounter | None = None,
) -> ConditionedPrompt:
    """Compose the prompt for this model, or refuse when it does not fit."""
    chosen = variant_for(variant)
    count = counter or estimated_tokens
    base = candidates_prompt(question.strip(), candidates)
    preamble = with_context("", system, history)
    preamble_tokens = count(preamble) if preamble else 0
    allowed = preamble_budget(signature)
    if preamble_tokens > allowed:
        raise PromptTooLong(
            f"the client context is {preamble_tokens} tokens; a "
            f"{signature.window_tokens}-token window carries {allowed} of context "
            "beside the answer"
        )
    text = with_context(base, system, history)
    prompt_tokens = count(text)
    ceiling = signature.window_tokens - ANSWER_RESERVE_TOKENS
    if prompt_tokens > ceiling:
        raise PromptTooLong(
            f"the prompt is {prompt_tokens} tokens and a {signature.window_tokens}-token "
            f"window leaves {ceiling} once the answer is reserved"
        )
    return ConditionedPrompt(text, chosen.name, prompt_tokens, preamble_tokens)
