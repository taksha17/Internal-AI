"""Turning corpus lines into the question/answer pairs adaptation trains on.

The first T4 run learned raw RFC prose and answered in it, citing almost nothing.
Causal loss over a document never shows a model that the thing being asked for is
a sentence plus the line it came from, so one epoch of it taught the register and
not the job. A pair shows both.

The question is built from the line's own content words rather than a fabricated
sentence, because inventing questions offline would need a model, and a model that
writes the questions would be third-party output in the training set. Keywords are
mechanical, checkable, and cannot introduce text from anywhere but the corpus.
"""

from __future__ import annotations

import re
from typing import Any

INSTRUCTION = (
    "Corpus lookup. Answer only from the corpus text you were trained on, and cite the file "
    "and line you used in the form rfc9110.txt:5026."
)

MAX_KEYWORDS = 10
MAX_ANSWER_WORDS = 40

LINE_KEY = re.compile(r"^(?P<path>[^:\s]+):(?P<line>\d+)$")

STOPWORDS = frozenset(
    (
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "must",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    )
)


def prompt_for(question: str) -> str:
    """The exact text the model sees before either a question or a set of keywords.

    Training and generation share this function so a pair cannot drift from the
    prompt it will be asked under.
    """
    return f"{INSTRUCTION}\nQuestion: {question}\nAnswer:"


def keywords_for(text: str, limit: int = MAX_KEYWORDS) -> str:
    """The line's content words, in the order they appear, without repetition."""
    picked: list[str] = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text):
        if word.lower() in STOPWORDS or word in picked:
            continue
        picked.append(word)
        if len(picked) == limit:
            break
    return ", ".join(picked)


def sentence_for(text: str, limit: int = MAX_ANSWER_WORDS) -> str:
    """The first sentence, capped, because the answer budget is 64 tokens."""
    first = re.split(r"(?<=\.)\s+", text.strip())[0]
    return " ".join(first.split()[:limit])


def citation_pairs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One training text per corpus line: keywords in, cited sentence out.

    A row whose key is not ``path:line`` is skipped rather than guessed at: a pair
    whose citation is not a citation teaches the model to write the wrong thing
    confidently, which is worse than training on less text.
    """
    pairs: list[dict[str, str]] = []
    for row in rows:
        key = str(row.get("record_key") or "")
        text = str(row.get("payload") or "").strip()
        match = LINE_KEY.match(key)
        if match is None or not text:
            continue
        keywords = keywords_for(text)
        sentence = sentence_for(text)
        if not keywords or not sentence:
            continue
        answer = f"{sentence} ({match.group('path')}:{match.group('line')})"
        pairs.append({"record_key": key, "text": f"{prompt_for(keywords)}\n{answer}"})
    return pairs
