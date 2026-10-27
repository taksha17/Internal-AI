"""Which model is worth building from a source, read off what the scan measured.

Someone who has never built a model can hand over a folder and still not know
where to start. This ranks the work the data can carry, best first, and says
what each option would need. It reads measurements only, so the same source
always gives the same answer, and it decides nothing: the person still chooses,
and every build behind a choice keeps its usual gate, approval and audit.
"""

from __future__ import annotations

import json
from typing import Any

SEARCH = "search"
DECIDE = "decide"
GENERATE = "generate"

KINDS: tuple[str, ...] = (SEARCH, DECIDE, GENERATE)

# Below this many usable records a keyword index beats a model trained from
# scratch, which is the result STATUS.md records at real size (98 records).
GENERATE_RECORDS = 500

# Line-oriented tables only: a column name has to sit on the first line, and a
# whole-file JSON object does not promise that.
TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".ndjson"})

# A column a classifier could learn. The name is the only evidence the scan can
# afford; counting the values in every row would mean reading the whole corpus.
LABEL_HINTS = frozenset(
    {
        "category",
        "class",
        "classification",
        "disposition",
        "intent",
        "label",
        "labels",
        "outcome",
        "priority",
        "queue",
        "resolution",
        "sentiment",
        "severity",
        "status",
        "tag",
        "tags",
        "team",
        "topic",
        "type",
    }
)

MAX_HEADER_CHARS = 2_000

_SCORES = {SEARCH: 30, DECIDE: 40, GENERATE: 50}

# Ranking among the options the data cannot carry. Nothing usable at all still
# has to name something, and search is the honest fallback: it needs no
# training and no labels.
_UNREADY = {SEARCH: 15, DECIDE: 10, GENERATE: 5}

# A table with no label column scores under search: it can be searched today,
# and it could be a classifier only after someone adds the answer column.
_DECIDE_UNLABELLED = 20


def _tabular(formats: dict[str, Any]) -> int:
    return sum(int(count) for suffix, count in formats.items() if str(suffix) in TABULAR_SUFFIXES)


def label_columns(suffix: str, text: str) -> list[str]:
    """Column names in the first row that could hold a classifier's answer.

    The scan already reads each text file, so this costs nothing extra: it looks
    at the first line only, and gives up on a line too long to be a header, so
    one wide row cannot make the walk expensive.
    """
    lowered = suffix.lower()
    if lowered not in TABULAR_SUFFIXES:
        return []
    first = text.split("\n", 1)[0]
    if not first.strip() or len(first) > MAX_HEADER_CHARS:
        return []
    if lowered in {".jsonl", ".ndjson"}:
        try:
            payload = json.loads(first)
        except ValueError:
            return []
        names = list(payload) if isinstance(payload, dict) else []
    else:
        delimiter = "\t" if lowered == ".tsv" else ","
        names = [name.strip().strip('"').strip() for name in first.split(delimiter)]
    return sorted({name.lower() for name in names if name and name.lower() in LABEL_HINTS})


def _search(usable: int) -> dict[str, Any]:
    return {
        "kind": SEARCH,
        "score": _SCORES[SEARCH] if usable else _UNREADY[SEARCH],
        "ready": usable > 0,
        "why": (
            "Search over these files answers questions with citations, and there is "
            "nothing to train."
        )
        if usable
        else "There are no usable text files to search.",
        "needs": "" if usable else "text files the scan can read",
    }


def _decide(tabular: int, labels: list[str], rows: bool) -> dict[str, Any]:
    """Whether a classifier is really within reach.

    ``rows`` is true only when the source is read a row at a time. A folder is
    read as one record per file, so a label column inside it has nowhere to
    live: promising a classifier there would promise what the build cannot do.
    """
    if labels and rows:
        return {
            "kind": DECIDE,
            "score": _SCORES[DECIDE],
            "ready": True,
            "why": (
                f"Rows carry a label column ({', '.join(sorted(labels))}), so a small "
                "classifier can learn to sort them."
            ),
            "needs": "",
        }
    if labels:
        return {
            "kind": DECIDE,
            "score": _DECIDE_UNLABELLED,
            "ready": False,
            "why": (
                f"The answer column is here ({', '.join(sorted(labels))}), but a folder "
                "is read one record per file, so a row cannot keep its own label."
            ),
            "needs": "a single table file, or a folder read row by row",
        }
    return {
        "kind": DECIDE,
        "score": _DECIDE_UNLABELLED if tabular else _UNREADY[DECIDE],
        "ready": False,
        "why": (
            f"Rows are here ({tabular} in a table format), but nothing in them says "
            "what the answer is."
        )
        if tabular
        else "This is prose, with no rows to learn a decision from.",
        "needs": "a column that holds the answer, such as label or status",
    }


def _generate(counted: int, unit: str) -> dict[str, Any]:
    return {
        "kind": GENERATE,
        "score": _SCORES[GENERATE] if counted >= GENERATE_RECORDS else _UNREADY[GENERATE],
        "ready": counted >= GENERATE_RECORDS,
        "why": (
            f"At about {counted:,} usable {unit} there is enough text for a model "
            "trained from scratch to earn its place."
        )
        if counted >= GENERATE_RECORDS
        else (
            f"At about {counted:,} usable {unit} keyword search wins, and a model "
            "trained from scratch would have to beat it to be promoted."
        ),
        "needs": "" if counted >= GENERATE_RECORDS else f"about {GENERATE_RECORDS:,} usable {unit}",
    }


def suggest_model_types(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank search, decide and generate for one source, best first.

    ``summary`` is an inspection ``to_dict()``, and may carry a built dataset's
    ``records`` once one exists. Once records are known they are the count that
    matters; before that, usable text files stand in. Options the data cannot
    carry stay in the list, unready, so the path stays visible.
    """
    text_files = int(summary.get("text_files") or 0)
    records = int(summary.get("records") or 0)
    formats = summary.get("formats") or {}
    labels = [str(name) for name in summary.get("label_columns") or []]
    usable = records or text_files
    unit = "records" if records else "text files"
    rows = str(summary.get("kind") or "") == "file"

    ranked = [
        _search(usable),
        _decide(_tabular(formats), labels, rows),
        _generate(usable, unit),
    ]
    ranked.sort(key=lambda option: (-option["score"], KINDS.index(option["kind"])))
    return [
        {
            "kind": option["kind"],
            "rank": rank,
            "ready": option["ready"],
            "why": option["why"],
            "needs": option["needs"],
        }
        for rank, option in enumerate(ranked, start=1)
    ]
