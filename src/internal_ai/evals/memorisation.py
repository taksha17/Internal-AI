"""Documents an embedding model maps onto their own opening line.

A healthy encoder puts a document's title near its body but not on top of it:
the title is a summary and the body is the text. When the two vectors are
almost identical, the model is reproducing the string rather than encoding what
it means, which is what memorisation looks like from the retrieval side.

Two guards keep the measure honest, because a raw cosine says little on its own.
The opening must be a small share of the document — a twenty-word prefix of a
seventy-word ticket lands near the ticket under any encoder, so a document is
probed only when its body outweighs its opening nine times over. And the opening
must be closer to its own document than documents are to each other, which rules
out an encoder whose vectors all point the same way: there the pair is high but
nothing is being reproduced.

A document with nothing beyond its opening is not probed: there the comparison
would be against itself and every such document would be flagged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from internal_ai.data.records import Record

OPENING_WORDS = 24
MIN_BODY_RATIO = 9
DEFAULT_THRESHOLD = 0.995
DEFAULT_MARGIN = 0.05
DEFAULT_LIMIT = 50


@dataclass(frozen=True, slots=True)
class RetrievalMemorisationReport:
    passed: bool
    flagged: tuple[dict[str, Any], ...]
    probed: int
    skipped: int
    threshold: float
    margin: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "flagged": [dict(item) for item in self.flagged],
            "probed": self.probed,
            "skipped": self.skipped,
            "threshold": self.threshold,
            "margin": self.margin,
        }


def opening_line(text: str, limit: int = OPENING_WORDS) -> str:
    """The first non-empty line, trimmed to a summary's worth of words."""
    for line in text.splitlines():
        words = line.split()
        if words:
            return " ".join(words[:limit])
    return ""


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return matrix / lengths


def _probe_pair(record: Record, limit: int) -> tuple[str, bool]:
    """The opening line to compare against, and whether the document qualifies."""
    opening = opening_line(record.text, limit)
    if not opening:
        return "", False
    body_words = len(record.text.split()) - len(opening.split())
    if body_words < MIN_BODY_RATIO * len(opening.split()):
        return opening, False
    return opening, True


def probe(
    model: Any,
    records: Sequence[Record],
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
    margin: float = DEFAULT_MARGIN,
) -> RetrievalMemorisationReport:
    """Documents whose opening line the model cannot tell from the whole record."""
    pairs: list[tuple[Record, str]] = []
    skipped = 0
    for record in list(records)[:limit]:
        opening, qualifies = _probe_pair(record, OPENING_WORDS)
        if not qualifies:
            skipped += 1
            continue
        pairs.append((record, opening))
    if not pairs:
        return RetrievalMemorisationReport(True, (), 0, skipped, threshold, margin)
    openings = _normalise(np.asarray(model.embed([opening for _record, opening in pairs])))
    bodies = _normalise(np.asarray(model.embed([record.text for record, _opening in pairs])))
    scores = openings @ bodies.T
    flagged: list[dict[str, Any]] = []
    for index, (record, opening) in enumerate(pairs):
        similarity = float(scores[index, index])
        others = np.delete(scores[index], index)
        background = float(np.median(others)) if others.size else None
        if similarity < threshold:
            continue
        if background is not None and similarity - background < margin:
            continue
        flagged.append(
            {
                "document": record.record_key,
                "similarity": round(similarity, 6),
                "background": round(background, 6) if background is not None else None,
                "opening": " ".join(opening.split()[:8]),
            }
        )
    return RetrievalMemorisationReport(
        passed=not flagged,
        flagged=tuple(flagged),
        probed=len(pairs),
        skipped=skipped,
        threshold=threshold,
        margin=margin,
    )
