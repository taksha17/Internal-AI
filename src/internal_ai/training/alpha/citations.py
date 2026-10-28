"""An answer may name an Ask hit, and a line number stays on that hit's file."""

from __future__ import annotations

import re
from dataclasses import dataclass

LINE_NUMBER = re.compile(r"\bline\s+(\d+)\b", re.IGNORECASE)


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


def score_answer(answer: str, hits: list[AskHit]) -> CitationReport:
    """Fail when a sentence puts one hit's line number on a different hit's path."""
    known = {hit.path: hit.line for hit in hits}
    mismatches: list[str] = []
    for sentence in re.split(r"(?<=\.)\s+|\n", answer):
        paths = [path for path in known if path in sentence]
        lines = [int(number) for number in LINE_NUMBER.findall(sentence)]
        if not paths or not lines:
            continue
        for path in paths:
            for line in lines:
                if known[path] == line:
                    continue
                owner = next((item for item in hits if item.line == line), None)
                if owner is None:
                    mismatches.append(f"{path} has no line {line}")
                else:
                    mismatches.append(f"line {line} belongs to {owner.path}, not {path}")
    return CitationReport(passed=not mismatches, mismatches=tuple(mismatches))
