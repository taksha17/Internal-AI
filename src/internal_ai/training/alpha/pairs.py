"""Turning corpus lines into the question/answer pairs adaptation trains on.

The first T4 run learned raw RFC prose and answered in it, citing almost nothing.
Causal loss over a document never shows a model that the thing being asked for is
a sentence plus the line it came from, so one epoch of it taught the register and
not the job. A pair shows both.

A record is one wrapped line, not a paragraph: a line can end mid-clause and the
next one is its continuation. Training on lines alone therefore taught the model
to answer "Network Working Group", which is a header. A passage is the line plus
the consecutive lines needed to finish its sentence, and the citation stays the
line the passage starts on, because that is what a reader jumps to.

The question is built from the passage's own content words rather than a
fabricated sentence, because inventing questions offline would need a model, and a
model that wrote the questions would be third-party output in the training set.
"""

from __future__ import annotations

import re
from typing import Any

INSTRUCTION = (
    "Corpus lookup. Answer only from the corpus text you were trained on, and cite the file "
    "and line you used in the form rfc9110.txt:5026."
)

MAX_KEYWORDS = 10
MAX_ANSWER_WORDS = 60
MIN_ANSWER_WORDS = 8
MAX_PASSAGE_WORDS = 80

LINE_KEY = re.compile(r"^(?P<path>[^:\s]+):(?P<line>\d+)$")

# A sentence ends on these; a line ending in a colon or a bare word continues.
SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")

# Page furniture, not prose. These carry a line number like anything else.
HEADER_LINE = re.compile(
    r"^(?:"
    r"RFC \d+|Internet Engineering Task Force|Internet Engineering Steering Group|"
    r"Network Working Group|Request for Comments|Category:|ISSN:|Obsoletes:|Updates:|"
    r"BCP:|STD:|DOI:|\[?Page \d+\]?$"
    r")",
    re.IGNORECASE,
)

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
    """The text's content words, in the order they appear, without repetition."""
    picked: list[str] = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text):
        if word.lower() in STOPWORDS or word in picked:
            continue
        picked.append(word)
        if len(picked) == limit:
            break
    return ", ".join(picked)


def _lines(rows: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    """Readable corpus lines in file order, with page furniture dropped."""
    parsed: list[tuple[str, int, str]] = []
    for row in rows:
        match = LINE_KEY.match(str(row.get("record_key") or ""))
        text = str(row.get("payload") or "").strip()
        if match is None or not text or HEADER_LINE.match(text):
            continue
        parsed.append((match.group("path"), int(match.group("line")), text))
    parsed.sort(key=lambda item: (item[0], item[1]))
    return parsed


def _passage(lines: list[tuple[str, int, str]], start: int) -> tuple[str, int]:
    """The line and its continuations, up to the end of its first sentence."""
    path = lines[start][0]
    parts = [lines[start][2]]
    index = start + 1
    while (
        not SENTENCE_END.search(parts[-1])
        and index < len(lines)
        and lines[index][0] == path
        and lines[index][1] == lines[index - 1][1] + 1
        and len(" ".join(parts).split()) < MAX_PASSAGE_WORDS
    ):
        parts.append(lines[index][2])
        index += 1
    joined = " ".join(parts)
    sentence = re.split(r"(?<=\.)\s+", joined)[0]
    return " ".join(sentence.split()[:MAX_ANSWER_WORDS]), index


def citation_pairs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One training text per answerable passage: keywords in, cited sentence out.

    Rows that are not ``path:line``, and passages too short to be a sentence, are
    skipped rather than guessed at: a pair whose citation is not a citation, or
    whose answer is half a clause, teaches the model to write the wrong thing
    confidently, which is worse than training on less text.
    """
    lines = _lines(rows)
    pairs: list[dict[str, str]] = []
    index = 0
    while index < len(lines):
        path, line, _text = lines[index]
        sentence, next_index = _passage(lines, index)
        index = max(next_index, index + 1)
        if len(sentence.split()) < MIN_ANSWER_WORDS:
            continue
        keywords = keywords_for(sentence)
        if not keywords:
            continue
        pairs.append(
            {
                "record_key": f"{path}:{line}",
                "text": f"{prompt_for(keywords)}\n{sentence} ({path}:{line})",
            }
        )
    return pairs
