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
MAX_ANSWER_WORDS = 80
MIN_ANSWER_WORDS = 8
MAX_PASSAGE_WORDS = 80

# The second T4 run learned to write a citation and could not pick a line: it
# cited runs of consecutive line numbers out of whichever file it had seen most.
# Asking a 4B model to recall which line of which RFC answers a question it has
# never seen is the wrong task, so the candidates a search would offer are put in
# the prompt and the model's job becomes choosing one. Five because keyword
# search over this corpus ranks the right passage inside the top five for most
# questions; the number is measured, not chosen for looks.
CANDIDATE_LIMIT = 5
CANDIDATE_WORDS = 40
# A candidate is copied, so the copy is what is scored: cite the key exactly as
# the candidate list wrote it.
SELECTION = (
    "Corpus lookup. Exactly one candidate below answers the question: copy it and cite its "
    "key as rfc9110.txt:5026. Write the copied sentence and the citation, nothing else."
)
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

# A footer wraps across the page edge and lands in the middle of a sentence, so
# it is dropped wherever it appears rather than only at the start of a line.
FOOTER_LINE = re.compile(
    r"\[Page \d+\]|\bRFC \d+\s+[A-Za-z].*\b(?:19|20)\d{2}\b|Standards Track",
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
        if match is None or not text or HEADER_LINE.match(text) or FOOTER_LINE.search(text):
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


def passages(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Every answerable passage as its citation key and its sentence.

    Rows that are not ``path:line``, and passages too short to be a sentence, are
    skipped rather than guessed at: a passage whose citation is not a citation, or
    whose answer is half a clause, teaches the model to write the wrong thing
    confidently, which is worse than training on less text.
    """
    lines = _lines(rows)
    found: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        path, line, _text = lines[index]
        sentence, next_index = _passage(lines, index)
        index = max(next_index, index + 1)
        if len(sentence.split()) < MIN_ANSWER_WORDS or not SENTENCE_END.search(sentence):
            continue
        found.append((f"{path}:{line}", sentence))
    return found


def _shown(text: str) -> str:
    """The candidate as the model sees it, so the copy can be the target."""
    return " ".join(text.split()[:CANDIDATE_WORDS])


def candidates_for(
    question: str, corpus: list[tuple[str, str]], limit: int = CANDIDATE_LIMIT
) -> list[tuple[str, str]]:
    """The passages a keyword search offers, in the order it ranked them.

    Ties are broken by corpus order so two runs cannot offer different lists for
    the same question, which would make the answers incomparable.
    """
    from internal_ai.evals.metrics import bm25_scores

    keys = [key for key, _text in corpus]
    scores = bm25_scores(question, [text for _key, text in corpus])
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return [(keys[index], _shown(corpus[index][1])) for index in order[:limit]]


def prompt_for_candidates(question: str, candidates: list[tuple[str, str]]) -> str:
    """The exact text the model sees when the answer is somewhere in the list.

    Training and generation share this, for the same reason they share
    ``prompt_for``: a pair that was asked differently from the way it is tested
    measures the difference between two prompts.
    """
    offered = "\n".join(f"[{key}] {text}" for key, text in candidates)
    return f"{SELECTION}\nQuestion: {question}\nCandidates:\n{offered}\nAnswer:"


def retrieval_pairs(
    rows: list[dict[str, Any]],
    corpus: list[tuple[str, str]],
    limit: int = CANDIDATE_LIMIT,
) -> list[dict[str, str]]:
    """Selection pairs: keywords, the candidates a search returns, the one that answers.

    ``corpus`` is the whole index a deployed search would cover, while ``rows`` is
    the training split the pairs may be built from, so a pair can offer a nearby
    passage the adapter does not learn from. A passage the search does not return
    is skipped rather than appended: padding the list with the answer would teach
    the model to pick whichever candidate looks appended.
    """
    pairs: list[dict[str, str]] = []
    for key, sentence in passages(rows):
        keywords = keywords_for(sentence)
        if not keywords:
            continue
        offered = candidates_for(keywords, corpus, limit)
        shown = dict(offered)
        if key not in shown or len(shown[key].split()) < MIN_ANSWER_WORDS:
            continue
        pairs.append(
            {
                "record_key": key,
                "text": f"{prompt_for_candidates(keywords, offered)}\n{shown[key]} ({key})",
            }
        )
    return pairs


def citation_pairs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One training text per answerable passage: keywords in, cited sentence out.

    Kept as the record of the format the second run was trained on. It is the
    format that taught the shape of an answer and not the choice of one.
    """
    pairs: list[dict[str, str]] = []
    for key, sentence in passages(rows):
        keywords = keywords_for(sentence)
        if not keywords:
            continue
        pairs.append({"record_key": key, "text": f"{prompt_for(keywords)}\n{sentence} ({key})"})
    return pairs
