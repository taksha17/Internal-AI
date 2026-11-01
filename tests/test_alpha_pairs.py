"""The pairs adaptation learns from, and the citation each one promises."""

from __future__ import annotations

from internal_ai.training.alpha.pairs import (
    INSTRUCTION,
    SELECTION,
    candidates_for,
    citation_pairs,
    keywords_for,
    prompt_for,
    prompt_for_candidates,
    retrieval_pairs,
)

PAYLOAD = (
    "A server generating a 401 response MUST send a WWW-Authenticate header field. "
    "A server MAY send it on other responses."
)
# A passage is one sentence, so the second one is not part of the answer.
FIRST = "A server generating a 401 response MUST send a WWW-Authenticate header field."


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


def test_a_wrapped_sentence_is_joined_and_cited_from_its_first_line() -> None:
    """A record is one wrapped line, so half a clause is not an answer."""
    rows = [
        {
            "record_key": "rfc9110.txt:10",
            "payload": "A server generating a 401 response MUST send a WWW-Authenticate header",
        },
        {"record_key": "rfc9110.txt:11", "payload": "field containing at least one challenge."},
        {"record_key": "rfc9110.txt:12", "payload": "Unrelated next sentence that follows."},
    ]

    [pair] = citation_pairs(rows)

    assert "at least one challenge" in pair["text"]
    assert "Unrelated next sentence" not in pair["text"]
    assert pair["text"].endswith("(rfc9110.txt:10)")


def test_page_furniture_is_not_an_answer() -> None:
    rows = [
        {"record_key": "rfc9110.txt:1", "payload": "Network Working Group"},
        {"record_key": "rfc9110.txt:2", "payload": "RFC 9110 HTTP Semantics June 2022"},
    ]

    assert citation_pairs(rows) == []


def test_a_passage_that_never_finishes_its_sentence_is_skipped() -> None:
    """Half a clause with a citation on it is a broken answer, not a cheap one."""
    rows = [
        {
            "record_key": "rfc9114.txt:210",
            "payload": "Multiplexing of requests is performed using the QUIC stream abstraction, which is",
        }
    ]

    assert citation_pairs(rows) == []


def test_a_page_footer_inside_a_passage_is_dropped() -> None:
    rows = [
        {
            "record_key": "rfc8446.txt:3421",
            "payload": "Rescorla Standards Track [Page 60] RFC 8446 TLS August 2018",
        }
    ]

    assert citation_pairs(rows) == []


def test_keywords_drop_the_words_that_carry_no_topic() -> None:
    assert keywords_for("The server MUST send the header field") == "server, send, header, field"


def test_the_same_rows_always_produce_the_same_pairs() -> None:
    rows = [{"record_key": "rfc9110.txt:5026", "payload": PAYLOAD}]

    assert citation_pairs(rows) == citation_pairs(list(rows))


def test_generation_asks_in_the_same_shape_training_answered() -> None:
    assert prompt_for("Which header?") == (f"{INSTRUCTION}\nQuestion: Which header?\nAnswer:")


CORPUS = [
    ("rfc9110.txt:5026", FIRST),
    ("rfc9111.txt:142", "HTTP is a stateless application-level request and response protocol."),
]


def test_the_candidates_are_the_ranked_search_the_baseline_runs() -> None:
    offered = candidates_for("401 WWW-Authenticate header", CORPUS, limit=1)

    assert offered == [("rfc9110.txt:5026", FIRST)]


def test_the_same_question_always_gets_the_same_candidates() -> None:
    assert candidates_for("401", CORPUS) == candidates_for("401", list(CORPUS))


def test_a_selection_pair_offers_the_candidates_and_ends_with_the_copy() -> None:
    [pair] = retrieval_pairs([{"record_key": "rfc9110.txt:5026", "payload": PAYLOAD}], CORPUS)

    assert pair["text"].startswith(SELECTION)
    assert "[rfc9110.txt:5026]" in pair["text"]
    assert pair["text"].endswith(f"{FIRST} (rfc9110.txt:5026)")


def test_a_passage_the_search_never_offers_is_not_padded_into_the_list() -> None:
    """Appending the answer would teach the model to pick whichever looks appended."""
    elsewhere = [("rfc9111.txt:142", "HTTP is a stateless application-level request protocol.")]

    assert (
        retrieval_pairs([{"record_key": "rfc9110.txt:5026", "payload": PAYLOAD}], elsewhere) == []
    )


def test_the_prompt_lists_every_candidate_it_must_choose_from() -> None:
    prompt = prompt_for_candidates("Which header?", CORPUS)

    assert prompt == (
        f"{SELECTION}\nQuestion: Which header?\nCandidates:\n"
        f"[rfc9110.txt:5026] {FIRST}\n"
        "[rfc9111.txt:142] HTTP is a stateless application-level request and response protocol.\n"
        "Answer:"
    )
