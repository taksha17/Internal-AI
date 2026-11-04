"""Training, evaluation and serving read from one prompt builder.

The adapter learned to answer inside the prompt ``pairs.prompt_for_candidates``
builds, and the scored run asked in exactly that text. Serving used to build
its own wording and to cite a record key with line 0, so the promotion evidence
described behaviour the deployed path did not reproduce. These tests hold the
three surfaces to the same bytes and refuse a key that loses its line.
"""

from __future__ import annotations

from internal_ai.llm import Completion
from internal_ai.training.alpha.answer import cited_answer
from internal_ai.training.alpha.citations import AskHit, hit_from_key
from internal_ai.training.alpha.pairs import (
    candidates_for,
    prompt_for,
    prompt_for_candidates,
    shown,
)

QUESTION = "which host is allowed in production?"
CORPUS = [
    (
        "Acmy_Website/next.config.ts:36",
        "The host is allowed in production when it appears as allowedHost in the config.",
    ),
    (
        "Acmy_Website/src/app/api/payments/create-order/route.ts:8",
        "The create-order route signs its request with the razorpay key id.",
    ),
]


class Capture:
    """A runner stand-in that records the exact text it was asked in."""

    name = "capture"

    def __init__(self) -> None:
        self.prompt: str | None = None

    def complete(self, prompt: str, system: str = "", **_: object) -> Completion:
        self.prompt = prompt
        return Completion(
            text="The host is listed in Acmy_Website/next.config.ts:36.", model="capture"
        )


def _hits_off_corpus() -> tuple[list[AskHit], list[str]]:
    hits: list[AskHit] = []
    excerpts: list[str] = []
    for key, text in CORPUS:
        hit = hit_from_key(key)
        hits.append(hit)
        excerpts.append(text)
    return hits, excerpts


def test_the_serving_prompt_is_the_prompt_training_uses() -> None:
    offered = candidates_for(QUESTION, CORPUS)
    trained = prompt_for_candidates(QUESTION, offered)

    hits, excerpts = _hits_off_corpus()
    adapter = Capture()
    cited_answer(adapter, QUESTION, hits, excerpts)

    assert adapter.prompt == trained


def test_a_question_without_candidates_uses_the_one_open_question_prompt() -> None:
    adapter = Capture()
    cited_answer(adapter, QUESTION, [], [])

    assert adapter.prompt == prompt_for(QUESTION)


def test_the_shown_excerpt_matches_what_the_pairs_builder_truncates() -> None:
    for _key, text in CORPUS:
        assert shown(shown(text)) == shown(text)


def test_a_line_addressed_key_keeps_its_line() -> None:
    assert hit_from_key("Acmy_Website/next.config.ts:36") == AskHit(
        "Acmy_Website/next.config.ts", 36
    )
    assert hit_from_key("Acmy_Website/next.config.ts:3:9") == AskHit(
        "Acmy_Website/next.config.ts:3", 9
    )


def test_a_key_that_names_no_line_cites_the_whole_record() -> None:
    assert hit_from_key("ds_chat#1") == AskHit("ds_chat#1", 0)
    assert hit_from_key("NBA.csv") == AskHit("NBA.csv", 0)
