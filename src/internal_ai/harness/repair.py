"""Bounded repair: one error, one correction prompt, a fixed number of tries.

A small model that writes a malformed citation has not failed the task; it has
failed one attempt. The repair is deterministic — the error the validator
returned becomes the correction — so a repair cannot invent a second question,
and the cap is checked before the first call so no configuration can turn a bad
answer into an open loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

MAX_ATTEMPTS = 3
"""Three is the ceiling: each retry costs a generation and buys one chance."""


@dataclass(frozen=True, slots=True)
class Attempt:
    index: int
    prompt: str
    text: str
    error: str


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    text: str
    attempts: tuple[Attempt, ...]
    exhausted: bool

    @property
    def used(self) -> int:
        return len(self.attempts)


def correction_prompt(prompt: str, error: str) -> str:
    """The original prompt plus the rejection, so a repair cannot nest prompts."""
    return (
        f"{prompt}\n\nYour previous answer was rejected: {error}. "
        "Answer again in exactly the same format, changing only what the rejection names."
    )


def run_with_repair(
    generate: Callable[[str], str],
    prompt: str,
    validate: Callable[[str], str | None],
    max_attempts: int = 2,
) -> RepairOutcome:
    """Generate, validate, and repair at most ``max_attempts`` times."""
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError(
            f"max_attempts must be between 1 and {MAX_ATTEMPTS}; {max_attempts} would "
            "either skip the check or leave the loop unbounded"
        )
    attempts: list[Attempt] = []
    current = prompt
    for index in range(1, max_attempts + 1):
        text = generate(current)
        error = validate(text) or ""
        attempts.append(Attempt(index, current, text, error))
        if not error:
            return RepairOutcome(text, tuple(attempts), False)
        current = correction_prompt(prompt, error)
    return RepairOutcome(attempts[-1].text, tuple(attempts), True)
