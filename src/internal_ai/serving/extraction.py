"""Spotting extraction-shaped traffic before an answer is generated.

An extraction attempt is not one question but many: the same information need
asked again and again with small rewordings. A served model that answers each
one leaks a little more every time, so this guardrail watches the shape of a
user's recent questions, never their content.

The comparison is deliberately model-free: Jaccard overlap over content words,
where a content word is what the training pairs call one. There is no query
vector to reuse on most routes — adapters are served through keyword retrieval,
and `/v1/answer` never embeds the question — so a vector signal would need a
new model call per request. That would be a new spend path for a guardrail, and
a new store of vectors to protect. Token overlap needs neither: it runs on
every endpoint, costs nothing, and the window lives only in memory. What
leaves the process is counts and a flag, in the audit event and the trace.

A burst of near-duplicate questions trips the guardrail; a few rephrasings do
not. Rephrasing your own question is normal, and the default action is warn.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from internal_ai.config import installed_extraction_policy
from internal_ai.store import Store
from internal_ai.training.alpha.memorisation import normalised_words
from internal_ai.training.alpha.pairs import STOPWORDS

EXTRACTION_WARN = "warn"
EXTRACTION_REFUSE = "refuse"
EXTRACTION_POLICIES = (EXTRACTION_WARN, EXTRACTION_REFUSE)
INSTALL_EXTRACTION_POLICY_KEY = "serving.extraction_policy"

WINDOW_SECONDS = 300
BURST_COUNT = 8
SIMILARITY_THRESHOLD = 0.5


def content_words(text: str) -> frozenset[str]:
    """The words that carry the question's information need, by the pairs' own rule."""
    return frozenset(word for word in normalised_words(text) if word not in STOPWORDS)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Overlap of two content-word sets; a question with no content words matches nothing."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class ExtractionMonitor:
    """One sliding window of content-word sets per user, in memory only.

    The window is the whole state: no vectors, no text and no word sets are
    persisted anywhere. A caller gets counts and a flag, which is everything
    the audit event and the trace are allowed to carry.
    """

    def __init__(
        self,
        window_seconds: int = WINDOW_SECONDS,
        burst_count: int = BURST_COUNT,
        similarity: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.window_seconds = window_seconds
        self.burst_count = burst_count
        self.similarity = similarity
        self._windows: dict[str, deque[tuple[float, frozenset[str]]]] = {}
        self._lock = threading.Lock()

    def observe(self, user_id: str, question: str, now: float | None = None) -> dict[str, Any]:
        """File this question and say whether the window looks like a burst.

        The incoming question counts toward its own burst: the eighth
        near-duplicate is what trips, not the ninth.
        """
        moment = now if now is not None else time.monotonic()
        words = content_words(question)
        with self._lock:
            window = self._windows.setdefault(user_id, deque())
            while window and moment - window[0][0] > self.window_seconds:
                window.popleft()
            window.append((moment, words))
            similar = sum(1 for _, other in window if jaccard(words, other) >= self.similarity)
            return {
                "suspect": similar >= self.burst_count,
                "similar": similar,
                "window": len(window),
            }


def check_extraction_policy(value: str) -> str:
    """Validate an install extraction policy; an unknown one fails closed."""
    policy = str(value or EXTRACTION_WARN)
    if policy not in EXTRACTION_POLICIES:
        raise ValueError(
            f"unknown extraction_policy {policy!r}; use one of {', '.join(EXTRACTION_POLICIES)}"
        )
    return policy


def configured_install_extraction_policy(store: Store) -> str:
    """What the install does with extraction-shaped traffic: warn, unless told to refuse.

    The console's setting wins, so an operator's choice holds from the next
    request and needs no restart. An install that never chose falls back to
    ``INTERNAL_AI_EXTRACTION_POLICY``.
    """
    return store.kv_get(INSTALL_EXTRACTION_POLICY_KEY, "") or installed_extraction_policy()
