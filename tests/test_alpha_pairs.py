"""The pairs adaptation learns from, and the citation each one promises."""

from __future__ import annotations

from internal_ai.training.alpha.pairs import (
    INSTRUCTION,
    citation_pairs,
    keywords_for,
    prompt_for,
)

PAYLOAD = (
    "A server generating a 401 response MUST send a WWW-Authenticate header field. "
    "A server MAY send it on other responses."
)


def test_a_pair_ends_with_the_line_it_came_from() -> None:
    [pair] = citation_pairs([{"record_key": "rfc9110.txt:5026", "payload": PAYLOAD}])

    assert pair["text"].startswith(INSTRUCTION)
    assert pair["text"].endswith("(rfc9110.txt:5026)")
    assert "MAY send it on other responses" not in pair["text"]


def test_a_row_without_a_line_is_not_trained_on() -> None:
    """A pair whose citation is not a citation teaches confident nonsense."""
    rows = [
        {"record_key": "notes", "payload": PAYLOAD},
        {"record_key": "rfc1.txt:12", "payload": "   "},
    ]

    assert citation_pairs(rows) == []


def test_keywords_drop_the_words_that_carry_no_topic() -> None:
    assert keywords_for("The server MUST send the header field") == "server, send, header, field"


def test_the_same_rows_always_produce_the_same_pairs() -> None:
    rows = [{"record_key": "rfc9110.txt:5026", "payload": PAYLOAD}]

    assert citation_pairs(rows) == citation_pairs(list(rows))


def test_generation_asks_in_the_same_shape_training_answered() -> None:
    assert prompt_for("Which header?") == (f"{INSTRUCTION}\nQuestion: Which header?\nAnswer:")
