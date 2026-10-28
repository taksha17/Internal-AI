"""A document must not collapse onto its own opening line.

When the vector for a record's first line sits almost exactly on the vector for
the whole record, the encoder is reproducing the string rather than its meaning.
The probe flags that only when the document is long enough for the body to have
moved the vector, and only when the opening is closer to its own record than
records are to each other.
"""

from __future__ import annotations

import hashlib

import numpy as np

from internal_ai.data.records import Record
from internal_ai.evals.harness import EvalHarness
from internal_ai.evals.memorisation import (
    DEFAULT_MARGIN,
    DEFAULT_THRESHOLD,
    opening_line,
    probe,
    probe_items,
)

HEAD_A = " ".join(f"title-a{index}" for index in range(24))
HEAD_B = " ".join(f"title-b{index}" for index in range(24))
BODY_A = " ".join(f"body-a{index}" for index in range(260))
BODY_B = " ".join(f"body-b{index}" for index in range(260))
LONG_A = f"{HEAD_A}\n{BODY_A}"
LONG_B = f"{HEAD_B}\n{BODY_B}"
SHORT = "Reset the vpn tunnel by rotating the gateway certificate."

DIMS = 32


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _line_vector(text: str) -> np.ndarray:
    return np.random.default_rng(_seed(text.splitlines()[0])).normal(size=DIMS)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def record(key: str, text: str) -> Record:
    return Record(id=key, record_key=key, text=text, source_id="s", acl=["engineering"])


class Memoriser:
    """An encoder that keys a document off its opening line alone."""

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([_line_vector(text) for text in texts])


class Summariser:
    """An encoder that reads the body: the opening sits near it, not on top."""

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [
                0.85 * _unit(_line_vector(text))
                + 0.15 * _unit(np.random.default_rng(_seed(text)).normal(size=DIMS))
                for text in texts
            ]
        )


class Collapsed:
    """An encoder whose vectors all point the same way. Nothing is reproduced."""

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0] + [0.0] * (DIMS - 1) for _text in texts])


def test_the_opening_line_is_the_first_non_empty_line_trimmed_to_a_summary() -> None:
    assert opening_line("\n\nReset the vpn tunnel. Then reconnect.\nmore") == (
        "Reset the vpn tunnel. Then reconnect."
    )
    words = " ".join(f"word{index}" for index in range(40))
    assert len(opening_line(words).split()) == 24


def test_a_document_mapped_onto_its_opening_line_is_flagged() -> None:
    report = probe(Memoriser(), [record("doc_a", LONG_A), record("doc_b", LONG_B)])

    assert report.passed is False
    assert report.probed == 2
    assert report.flagged[0]["document"] == "doc_a"
    assert report.flagged[0]["similarity"] == 1.0
    assert report.flagged[0]["background"] < 1.0
    assert report.threshold == DEFAULT_THRESHOLD
    assert report.margin == DEFAULT_MARGIN


def test_an_encoder_that_reads_the_body_passes() -> None:
    report = probe(Summariser(), [record("doc_a", LONG_A), record("doc_b", LONG_B)])

    assert report.passed is True
    assert report.flagged == ()
    assert report.probed == 2


def test_a_short_document_is_not_probed() -> None:
    report = probe(Memoriser(), [record("doc_a", SHORT)])

    assert report.passed is True
    assert report.probed == 0
    assert report.skipped == 1


def test_an_encoder_that_scores_everything_alike_is_not_a_memoriser() -> None:
    report = probe(Collapsed(), [record("doc_a", LONG_A), record("doc_b", LONG_B)])

    assert report.passed is True
    assert report.flagged == ()


def test_the_threshold_is_the_line_that_decides() -> None:
    records = [record("doc_a", LONG_A)]
    flagged = probe(Summariser(), records, threshold=0.0)
    clear = probe(Memoriser(), records, threshold=1.01)

    assert flagged.flagged and flagged.passed is False
    assert clear.passed is True


def test_an_evaluation_fails_on_a_memorising_encoder(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    corpus = [record("doc_a", LONG_A)]
    eval_set = harness.create_eval_set(
        "memorised", [{"query": "reset the vpn", "relevant_ids": ["doc_a"]}]
    )

    outcome = harness.evaluate_retrieval(
        Memoriser(), eval_set, corpus, training_records=[], margin=0.0
    )

    check = outcome.checks["retrieval_memorisation"]
    assert check["passed"] is False
    assert check["flagged"][0]["document"] == "doc_a"
    assert outcome.passed is False


def test_an_evaluation_records_the_probe_even_when_it_passes(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    corpus = [record("doc_a", LONG_A)]
    eval_set = harness.create_eval_set(
        "clean", [{"query": "reset the vpn", "relevant_ids": ["doc_a"]}]
    )

    outcome = harness.evaluate_retrieval(
        Summariser(), eval_set, corpus, training_records=[], margin=0.0
    )

    assert outcome.checks["retrieval_memorisation"]["passed"] is True
    assert outcome.checks["retrieval_memorisation"]["probed"] == 1


def test_an_evaluation_says_when_nothing_was_long_enough_to_probe(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    corpus = [record("doc_a", SHORT)]
    eval_set = harness.create_eval_set(
        "tickets", [{"query": "reset the vpn", "relevant_ids": ["doc_a"]}]
    )

    outcome = harness.evaluate_retrieval(
        Memoriser(), eval_set, corpus, training_records=[], margin=0.0
    )

    check = outcome.checks["retrieval_memorisation"]
    assert check["passed"] is True
    assert check["probed"] == 0
    assert check["skipped"] == 1
    assert "long enough" in check["note"]


def test_the_advice_names_the_flagged_documents() -> None:
    from internal_ai.evals.advice import retrieval_advice

    report = {
        "passed": False,
        "checks": {
            "retrieval_memorisation": {
                "passed": False,
                "flagged": [{"document": "doc_a", "similarity": 0.9971}],
            }
        },
        "metrics": {},
        "baseline_metrics": {},
    }

    lines = retrieval_advice(report, records=10, tokens=1000)

    assert any("doc_a at 0.997" in line for line in lines)


class Lookup:
    """An encoder with fixed answers: some texts deliberately land together."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, texts: list[str]) -> np.ndarray:
        fallback = _unit(np.random.default_rng(_seed("other")).normal(size=DIMS))
        return np.array([self.mapping.get(text, fallback) for text in texts])


def _item(paraphrase: str | None, key: str = "doc_a") -> dict[str, object]:
    item: dict[str, object] = {"query": "how do I reset the vpn", "relevant_ids": [key]}
    if paraphrase is not None:
        item["paraphrase"] = paraphrase
    return item


def test_a_rewrite_scored_as_the_record_itself_is_flagged() -> None:
    same = [1.0] + [0.0] * (DIMS - 1)
    model = Lookup({LONG_A: same, "how do I reset the vpn": same})

    report = probe_items(model, [_item("how do I reset the vpn")], [record("doc_a", LONG_A)])

    assert report.passed is False
    assert report.probed == 1
    assert report.flagged[0]["document"] == "doc_a"
    assert report.flagged[0]["similarity"] == 1.0
    assert report.flagged[0]["paraphrase"] == "how do I reset the vpn"


def test_an_item_without_a_paraphrase_is_skipped() -> None:
    report = probe_items(Lookup({}), [_item(None)], [record("doc_a", LONG_A)])

    assert report.passed is True
    assert report.probed == 0
    assert report.skipped == 1


def test_an_item_pointing_at_an_absent_record_is_skipped() -> None:
    report = probe_items(Lookup({}), [_item("how do I reset the vpn", key="doc_z")], [])

    assert report.probed == 0
    assert report.skipped == 1


def test_an_evaluation_fails_on_a_rewrite_scored_as_the_record(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    same = [1.0] + [0.0] * (DIMS - 1)
    model = Lookup({LONG_A: same, "how do I reset the vpn": same})
    corpus = [record("doc_a", LONG_A)]
    eval_set = harness.create_eval_set(
        "rewrites",
        [
            {
                "query": "how do I reset the vpn",
                "paraphrase": "how do I reset the vpn",
                "relevant_ids": ["doc_a"],
            }
        ],
    )

    outcome = harness.evaluate_retrieval(model, eval_set, corpus, training_records=[], margin=0.0)

    check = outcome.checks["paraphrase_memorisation"]
    assert check["passed"] is False
    assert check["probed"] == 1
    assert outcome.passed is False


def test_an_evaluation_says_when_no_item_carried_a_paraphrase(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)
    model = Lookup({})
    corpus = [record("doc_a", LONG_A)]
    eval_set = harness.create_eval_set("plain", [_item(None)])

    outcome = harness.evaluate_retrieval(model, eval_set, corpus, training_records=[], margin=0.0)

    check = outcome.checks["paraphrase_memorisation"]
    assert check["passed"] is True
    assert check["probed"] == 0
    assert "paraphrase" in check["note"]


def test_the_rewrite_is_part_of_the_frozen_exam(platform) -> None:
    harness = EvalHarness(platform.store, platform.audit)

    first = harness.create_eval_set("a", [_item("how do I reset the vpn")])
    second = harness.create_eval_set("b", [_item("vpn tunnel troubleshooting")])

    assert first.items[0]["paraphrase"] == "how do I reset the vpn"
    assert first.content_hash != second.content_hash


def test_the_advice_names_records_flagged_for_a_rewrite() -> None:
    from internal_ai.evals.advice import retrieval_advice

    report = {
        "passed": False,
        "checks": {
            "paraphrase_memorisation": {
                "passed": False,
                "flagged": [{"document": "doc_a", "similarity": 1.0}],
            }
        },
        "metrics": {},
        "baseline_metrics": {},
    }

    lines = retrieval_advice(report, records=10, tokens=1000)

    assert any("doc_a at 1.000" in line for line in lines)
