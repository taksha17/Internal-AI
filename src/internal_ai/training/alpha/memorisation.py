"""Words the answer repeats from its training data, unchanged.

Citation checking asks whether an answer named a source; it never asks whether
the answer is reciting one. The probe answers that: a run of words from a
training record that comes back out of the answer in the same order is a
memorised span. Eight words is the short end of a sentence, so a match is not a
coincidence of common phrasing - and the refusal names the span, so a person can
judge a boilerplate false positive.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

WORD = re.compile(r"[a-z0-9]+(?:['.-][a-z0-9]+)*")
SPAN_WORDS = 8
SOURCE_LIMIT = 200


@dataclass(frozen=True, slots=True)
class MemorisationReport:
    passed: bool
    repeated: tuple[str, ...]
    probed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "repeated": list(self.repeated),
            "probed": self.probed,
        }


def normalised_words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def answer_spans(answer: str, length: int = SPAN_WORDS) -> set[tuple[str, ...]]:
    """Every run of `length` words in the answer, as a set of word tuples."""
    words = normalised_words(answer)
    return {tuple(words[start : start + length]) for start in range(len(words) - length + 1)}


def probe(
    answer: str,
    sources: Sequence[str],
    length: int = SPAN_WORDS,
    limit: int = SOURCE_LIMIT,
) -> MemorisationReport:
    """Runs of words the answer took from its sources without rewording them."""
    if length < 2:
        raise ValueError("a memorised span needs at least two words to mean anything")
    spans = answer_spans(answer, length)
    repeated: list[str] = []
    probed = 0
    if not spans:
        return MemorisationReport(passed=True, repeated=(), probed=0)
    for text in sources[:limit]:
        probed += 1
        words = normalised_words(text)
        for start in range(0, max(len(words) - length + 1, 0), length):
            window = tuple(words[start : start + length])
            if window in spans:
                span = " ".join(window)
                if span not in repeated:
                    repeated.append(span)
    return MemorisationReport(passed=not repeated, repeated=tuple(repeated), probed=probed)
