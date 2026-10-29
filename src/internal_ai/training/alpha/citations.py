"""An answer may name an Ask hit, and a line number stays on that hit's file."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Both the prose form ("line 12") and the hit form ("src/app.ts:12") count. A
# digit before the colon is excluded so a clock time is not read as a citation.
LINE_NUMBER = re.compile(r"(?:\bline\s+|(?<!\d):)(\d+)\b", re.IGNORECASE)

# A run of characters that could be a path. Whether it names a file is decided
# by its ending, not by where it sits in the sentence.
PATH_RUN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*")

# The characters that may sit inside a path. A hit path only counts when it
# stands on its own, so `app.ts` is not cited by `src/app.ts`.
_PATH_BODY = r"A-Za-z0-9_/-"


@dataclass(frozen=True, slots=True)
class AskHit:
    path: str
    line: int


@dataclass(frozen=True, slots=True)
class CitationReport:
    passed: bool
    mismatches: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "mismatches": list(self.mismatches)}


def _stands_alone(sentence: str, path: str) -> bool:
    """True when the answer wrote this path, and not a longer path around it."""
    pattern = re.compile(
        rf"(?<![{_PATH_BODY}]){re.escape(path)}"
        rf"(?![{_PATH_BODY}])(?!\.[{_PATH_BODY}])"
    )
    return pattern.search(sentence) is not None


def _names_a_file(token: str) -> bool:
    if "." not in token:
        return False
    extension = token.rsplit(".", 1)[1]
    return extension[:1].isalpha() and 2 <= len(extension) <= 8


def _named_paths(sentence: str, known: dict[str, int]) -> list[str]:
    """The files a sentence names.

    A hit path counts when it stands on its own. Every other file-shaped token
    counts too, because a line number on a file that was never a hit is the
    move this check exists to catch.
    """
    named: list[str] = []
    for path in known:
        if _stands_alone(sentence, path) and path not in named:
            named.append(path)
    for match in PATH_RUN.finditer(sentence):
        token = match.group(0).rstrip(".")
        if not _names_a_file(token) or token in named:
            continue
        named.append(token)
    return named


def score_answer(answer: str, hits: list[AskHit]) -> CitationReport:
    """Fail when a sentence puts one hit's line number on a different hit's path."""
    known = {hit.path: hit.line for hit in hits}
    mismatches: list[str] = []
    for sentence in re.split(r"(?<=\.)\s+|\n", answer):
        lines = [int(number) for number in LINE_NUMBER.findall(sentence)]
        if not lines:
            continue
        for path in _named_paths(sentence, known):
            owner_line = known.get(path)
            if owner_line is None:
                mismatches.append(f"{path} is not an Ask hit")
                continue
            for line in lines:
                if owner_line == line:
                    continue
                owner = next((item for item in hits if item.line == line), None)
                if owner is None:
                    mismatches.append(f"{path} has no line {line}")
                else:
                    mismatches.append(f"line {line} belongs to {owner.path}, not {path}")
    return CitationReport(passed=not mismatches, mismatches=tuple(mismatches))
