"""Whitespace and non-ASCII survive a trip through the from-scratch tokenizer."""

from __future__ import annotations

from internal_ai.data.tokenizer import train_tokenizer

SAMPLES = (
    "  hello  ",
    "before\n\nafter",
    "café",
    "漢字",
    "1234567890",
    "ledgerline",
)


def test_diverse_text_round_trips() -> None:
    tokenizer = train_tokenizer(list(SAMPLES), vocab_size=400)
    for text in SAMPLES:
        assert tokenizer.decode(tokenizer.encode(text)) == text
