"""Instruction-shaped text inside records, so a document cannot give orders.

A ticket that says "ignore previous instructions" is data. Both places that turn
corpus text into model behaviour - the release audit and the evaluation - check
for it, because a corpus is the one input nobody reads line by line.
"""

from __future__ import annotations

import re
from typing import Any

INSTRUCTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override": re.compile(
        r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|context)\b"
    ),
    # "act as a client" is how a protocol describes a program, so only a role
    # aimed at a model counts here; otherwise every specification is an injection.
    "persona": re.compile(
        r"(?i)\byou\s+are\s+now\s+(?:a|an|the)\b"
        r"|\bact\s+as\s+(?:a|an|the)\s+(?:ai|assistant|chatbot|model|language\s+model|dan)\b"
        r"|\bsystem\s+prompt\b"
    ),
    "exfiltration": re.compile(
        r"(?i)\b(?:send|post|upload|email|exfiltrate)\b[^.\n]{0,40}\b(?:to|at)\b"
        r"\s*(?:https?://|[\w.-]+@[\w.-]+)"
    ),
    "tooling": re.compile(
        r"(?i)\b(?:run|execute|call|invoke)\s+(?:the\s+)?(?:command|shell|tool|script|function)\b"
    ),
}


def instruction_kinds(text: str) -> list[str]:
    """Which instruction shapes a piece of text carries, sorted for stable output."""
    return sorted(kind for kind, pattern in INSTRUCTION_PATTERNS.items() if pattern.search(text))


def instruction_hits(records: list[Any]) -> list[dict[str, str]]:
    """Records that carry an instruction, named by their record key."""
    hits: list[dict[str, str]] = []
    for record in records:
        kinds = instruction_kinds(record.text)
        if kinds:
            hits.append({"document": record.record_key, "kinds": ", ".join(kinds)})
    return hits
