"""The instruments that decide an alpha report, checked on their own.

Two of them failed the first real T4 run for reasons that had nothing to do with
the model, so each rule now has a test that states what it is allowed to mean.
"""

from __future__ import annotations

from internal_ai.data.transforms import PII_PATTERNS
from internal_ai.training.alpha.citations import AskHit, score_answer


def test_a_dotted_section_number_is_not_an_address() -> None:
    """The corpus is full of Section 8.1.2.1, and it is not a host."""
    pattern = PII_PATTERNS["ip_address"]

    assert pattern.search("Section 8.1.2.1 defines the field order.") is None
    assert pattern.search("section 4.2.3.1 applies here") is None


def test_two_lines_of_one_file_are_each_a_valid_citation() -> None:
    """An item may cite the same file twice, and neither line is a move."""
    hits = [AskHit("rfc9110.txt", 10), AskHit("rfc9110.txt", 20)]

    assert score_answer("See rfc9110.txt:10.", hits).passed
    assert score_answer("See rfc9110.txt:20.", hits).passed


def test_a_line_the_item_does_not_hold_is_still_a_mismatch() -> None:
    """Fixing the message must not make a borrowed line number acceptable."""
    hits = [AskHit("rfc9110.txt", 10), AskHit("rfc9110.txt", 20)]

    assert score_answer("See rfc9110.txt:30.", hits).mismatches == ("rfc9110.txt has no line 30",)
    assert score_answer("See rfc6750.txt:10.", hits).mismatches == (
        "rfc6750.txt is not an Ask hit",
    )


def test_a_real_address_is_still_an_address() -> None:
    pattern = PII_PATTERNS["ip_address"]

    assert pattern.search("the proxy at 203.0.113.9 answered") is not None
