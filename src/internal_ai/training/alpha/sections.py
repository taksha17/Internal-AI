"""Questions built from an RFC's own sentences, so a person can check them.

A frozen set is only evidence if a model did not write its questions: that would
put third-party output in the question set and in the training text at once. So
nothing here is generated. A question is a template applied to a sentence the
corpus already carries, and the reference answer is that same sentence, which is
a span a reviewer can compare against the cited line by eye.

Three cues are recognisable without a parser and cover most of what an RFC says:

- a definition — "A URI-reference is either a URI or a relative reference."
- a description — "The Range header field specifies ..."
- a requirement — "A server MUST send ..."

Everything else is skipped rather than guessed at, because a question the text
does not answer is not a question. A heading is recorded so a reviewer knows
which part of a document an item came from; the question never asserts anything
the sentence does not say.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# RFC headings are numbered and separated from their title by two or more
# spaces. A table-of-contents line has the same shape, so it is excluded by its
# trailing leader dots and page number.
_HEADING = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?\s{2,}(?P<title>[A-Z][^\n]{0,80})$")
_LEADER = re.compile(r"\.{2,}\s*\d+$")

# Page furniture travels through the corpus and answers nothing. A question
# written from "RFC 9110 HTTP Semantics May 2025" would be answered by the header
# of every page, so those lines never become a sentence.
_FURNITURE = (
    re.compile(r"\[Page\s+\d+\]"),
    re.compile(r"^(?:RFC|BCP|STD)\s+\d+\b"),
    re.compile(r"^(?:Internet-Draft|Intended status|Obsoletes|Updates|Category)\s*:"),
    re.compile(r"^Expires\b"),
    re.compile(r"^(?:Author's|Authors')\s+Addresses?$"),
    re.compile(r"^\[[A-Z]{2,}[A-Z0-9-]*\d\]"),
    re.compile(r"^(?:<blank>|\f)$"),
)

# A definition names a thing and then says what it is.
_DEFINITION = re.compile(
    r"^(?P<subject>(?:The|A|An)\s+[A-Za-z0-9][^.;:()]{2,60}?)\s+(?:is|are)\s+"
    r"(?:a|an|the|either|defined|sent|used|required|intended)\b"
)

# A description names a thing and then says what it does, which is a question
# with the same subject and the verb as its own base form.
_DESCRIPTION = re.compile(
    r"^(?P<subject>(?:The\s+)?[A-Z][A-Za-z0-9/._-]*(?:\s+[A-Za-z0-9/._-]+){0,5}?)\s+"
    r"(?P<verb>means|indicates|specifies|defines|describes|provides|denotes|contains|"
    r"refers to|consists of)\b"
)
_BASE_FORM = {
    "means": "mean",
    "indicates": "indicate",
    "specifies": "specify",
    "defines": "define",
    "describes": "describe",
    "provides": "provide",
    "denotes": "denote",
    "contains": "contain",
    "refers to": "refer to",
    "consists of": "consist of",
}

# RFC 2119 key words. `SHALL` means the same thing as `MUST` there, and
# `REQUIRED` and `RECOMMENDED` sit where a modal sits.
_NORMATIVE = re.compile(
    r"\b(?P<modal>MUST NOT|SHALL NOT|MUST|SHALL|SHOULD NOT|SHOULD|REQUIRED|MAY)\b"
)
_AUXILIARY = {
    "MUST": ("must", False),
    "REQUIRED": ("must", False),
    "SHALL": ("must", False),
    "MUST NOT": ("must", True),
    "SHALL NOT": ("must", True),
    "SHOULD": ("should", False),
    "SHOULD NOT": ("should", True),
    "MAY": ("may", False),
}

# A subject that contains or trails into one of these is a clause, not a thing,
# and the question would read as nonsense.
_DANGLING = frozenset(
    {"that", "which", "who", "whom", "whose", "when", "where", "if", "and", "or", "but"}
)
_PRONOUN = frozenset({"it", "this", "that", "they", "these", "those", "such", "one"})
_GENERIC = frozenset({"rfc", "document", "section", "specification", "memo", "text", "appendix"})
_CONDITIONAL = frozenset({"if", "when", "while", "unless", "because", "although", "once", "after"})
_LEADING = re.compile(
    r"^(?:However|Therefore|Thus|Also|Note that|In addition|In this case|"
    r"Consequently|Furthermore|Finally|For example)\s*,?\s*",
    re.IGNORECASE,
)

# A key word quoted inside a parenthetical ("MUST", "MAY") is being discussed,
# not used as a requirement, and an ABNF production is a grammar rule. Neither
# becomes a question: the sentence around them is prose about notation.
_GRAMMAR = re.compile(r"::=|\s=\s")
_QUOTED_WORD = re.compile(r'"(?:MUST NOT|MUST|SHOULD NOT|SHOULD|REQUIRED|RECOMMENDED|MAY)"')
_PARENTHETICAL = re.compile(r"\([^()]*\)")


@dataclass(frozen=True, slots=True)
class Heading:
    number: str
    title: str
    line: int

    @property
    def label(self) -> str:
        return f"{self.number} {self.title}"


def headings(path: Path) -> list[Heading]:
    """Every section heading in one RFC, in file order."""
    found: list[Heading] = []
    for index, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        match = _HEADING.match(raw.rstrip())
        if match is None or _LEADER.search(raw):
            continue
        found.append(Heading(match.group("number"), match.group("title").strip(), index + 1))
    return found


def section_for(found: list[Heading], line: int) -> Heading | None:
    """The last heading at or before a line, which is the section it sits in."""
    enclosing: Heading | None = None
    for heading in found:
        if heading.line > line:
            break
        enclosing = heading
    return enclosing


def _furniture(line: str) -> bool:
    """Page furniture, and a heading, which is prose about the document."""
    text = line.strip()
    if not text or _HEADING.match(text) is not None:
        return True
    return any(pattern.search(text) for pattern in _FURNITURE)


def prose(payload: str) -> str:
    """One paragraph's lines joined back into prose.

    The corpus is hard-wrapped at the source's 72 columns, so a sentence spans
    lines and a hyphenated word is split across them. Joining with a space would
    turn "WWW-Authenticate" into "WWW- Authenticate", which no model can cite
    and no reviewer can search for.
    """
    joined = ""
    for raw in payload.splitlines():
        line = raw.strip()
        if _furniture(line):
            # A dropped line is also a sentence boundary: what follows a page
            # break is a different sentence, not the tail of the one before it.
            if joined and not joined.endswith((".", " ")):
                joined += ". "
            continue
        if joined.endswith("-") and line[:1].isalpha():
            # RFC text wraps at a hyphen the word already has, so keeping it is
            # what makes "WWW-" and "Authenticate" one searchable term again.
            joined += line
        else:
            joined += " " + line
    return re.sub(r"\s+", " ", joined).strip()


def sentences(payload: str) -> list[str]:
    """The sentences of one corpus record that could carry a question."""
    text = prose(payload)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text)
    return [part.strip() for part in parts if part.strip()]


def _usable(sentence: str) -> bool:
    words = sentence.split()
    if not 8 <= len(words) <= 70 or len(sentence) > 420:
        return False
    if not sentence.endswith(".") or sentence.count('"') % 2:
        return False
    if _furniture(sentence) or _GRAMMAR.search(sentence) or _QUOTED_WORD.search(sentence):
        return False
    return any(pattern.search(sentence) for pattern in (_DEFINITION, _DESCRIPTION, _NORMATIVE))


def _phrase(text: str, limit: int = 10) -> str | None:
    """A subject phrase from the text before a verb, or ``None``.

    A parenthetical is dropped rather than fatal — an RFC writes "the server (as
    described in Section 7) MUST" and the requirement is still about the server —
    and a comma starts a new clause, so only the last one can be the subject. A
    quote or a colon left over means the key word was being discussed, not used.
    """
    cleaned = _PARENTHETICAL.sub(" ", text)
    cleaned = cleaned.rsplit(",", 1)[-1]
    cleaned = _LEADING.sub("", cleaned.strip())
    if any(char in cleaned for char in '";:') or not cleaned:
        return None
    words = cleaned.split()
    if words and words[0].lower() in _CONDITIONAL:
        return None
    if len(words) > limit:
        words = words[len(words) - limit :]
    while words and words[0].lower() in {"that", "which", "and", "or", "but", "so", "then"}:
        words = words[1:]
    while words and words[-1].lower() in {*_DANGLING, "of", "to", "for", "with", "at", "in"}:
        words = words[:-1]
    if len(words) < 2 or len(words[0]) < 1 or set(words) & _DANGLING:
        return None
    phrase = " ".join(words).strip(" ,;:")
    return phrase if phrase[:1].isalpha() else None


def question_for(sentence: str) -> str | None:
    """The question this sentence answers, or ``None`` when it answers none.

    The template is chosen by the cue, never by content: a definition becomes
    "What is X?", a description "What does X specify?", a requirement "What must
    X do?". The sentence itself is the reference answer, so a wrong question
    shows up as a reviewer reading one line and disagreeing, not as a hidden
    judgement.
    """
    definition = _DEFINITION.match(sentence)
    if definition is not None:
        subject = definition.group("subject").strip(" ,;:")
        words = subject.split()
        if len(words) >= 2 and not set(words) & (_DANGLING | _PRONOUN):
            article, rest = words[0], " ".join(words[1:])
            verb = "are" if sentence[len(subject) :].lstrip().startswith("are") else "is"
            return f"What {verb} {article.lower()} {rest}?"
    description = _DESCRIPTION.match(sentence)
    if description is not None:
        subject = description.group("subject").strip(" ,;:")
        words = subject.split()
        # "this document specifies" is a sentence about the document, and the
        # question it makes — what does this document specify? — is answered by
        # every line in it.
        if (
            2 <= len(words) <= 6
            and words[0].lower() not in _CONDITIONAL
            and not set(words) & (_PRONOUN | _DANGLING)
            and not {word.lower() for word in words} & _GENERIC
        ):
            base = _BASE_FORM[description.group("verb")]
            return f"What does {subject[0].lower()}{subject[1:]} {base}?"
    normative = _NORMATIVE.search(sentence)
    if normative is None:
        return None
    subject = _phrase(sentence[: normative.start()])
    if subject is None:
        return None
    auxiliary, negated = _AUXILIARY[normative.group("modal")]
    if subject.split()[0].lower() in {"the", "a", "an"}:
        head, _, tail = subject.partition(" ")
        subject = f"{head.lower()} {tail}"
    return f"What {auxiliary} {subject} {'not ' if negated else ''}do?"


def questions(payload: str, per_record: int = 2) -> list[tuple[str, str]]:
    """Up to ``per_record`` (question, reference answer) pairs from one record."""
    found: list[tuple[str, str]] = []
    for sentence in sentences(payload):
        if not _usable(sentence):
            continue
        question = question_for(sentence)
        if question is None:
            continue
        found.append((question, sentence))
        if len(found) == per_record:
            break
    return found
