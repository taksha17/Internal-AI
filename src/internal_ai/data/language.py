"""Language tags and the borderline-text rule checked at release.

The tag is a heuristic, so a domain expert marks a seeded sample before release.
A toxic match is kept and flagged; release only proceeds when the dataset stores
the sentence that says so.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from internal_ai.services.actor import Actor
from internal_ai.store import Store

SAMPLE_SIZE = 30
SAMPLE_SEED = "pii-audit"
WRONG_LIMIT = 0.10
BORDERLINE_POLICY = "Borderline text is kept and flagged, not deleted."
_TERMS = Path(__file__).with_name("toxicity_terms.txt")


def language_tag(guess: str) -> str:
    if guess in {"en", "unknown"}:
        return guess
    return "other"


def load_toxicity_terms() -> tuple[str, ...]:
    terms = []
    for line in _TERMS.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            terms.append(word)
    return tuple(terms)


def is_borderline(text: str, terms: tuple[str, ...] | None = None) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return any(term in words for term in (terms or load_toxicity_terms()))


def _rows(store: Store, dataset_id: str) -> list[dict[str, Any]]:
    return store.dataset_records(dataset_id)


def language_sample(store: Store, dataset_id: str) -> list[dict[str, Any]]:
    rows = _rows(store, dataset_id)
    if len(rows) > SAMPLE_SIZE:
        return random.Random(SAMPLE_SEED).sample(rows, SAMPLE_SIZE)
    return rows


def store_language_review(
    store: Store, dataset_id: str, marks: dict[str, str], reviewed_by: str
) -> dict[str, Any]:
    sample = language_sample(store, dataset_id)
    expected = {row["id"] for row in sample}
    if set(marks) != expected:
        raise ValueError("the language review must mark every sampled record")
    if any(mark not in {"correct", "wrong"} for mark in marks.values()):
        raise ValueError("each language mark is correct or wrong")
    wrong = sum(1 for mark in marks.values() if mark == "wrong")
    review = {
        "reviewed_by": reviewed_by,
        "sampled": len(sample),
        "wrong": wrong,
        "marks": dict(sorted(marks.items())),
    }
    _write_stats(store, dataset_id, {"language_review": review})
    return review


def accept_language_sample(store: Store, dataset_id: str, reviewed_by: str = "fixture") -> None:
    """Mark the seeded sample correct. A domain expert uses ``submit_language_review``."""
    rows = language_sample(store, dataset_id)
    if not any(row["meta"].get("language") for row in rows):
        return
    store_language_review(store, dataset_id, {row["id"]: "correct" for row in rows}, reviewed_by)


def submit_language_review(
    store: Store, actor: Actor, dataset_id: str, marks: dict[str, str]
) -> dict[str, Any]:
    if not actor.holds("domain_expert"):
        raise PermissionError("a domain expert reviews the language sample")
    return store_language_review(store, dataset_id, marks, actor.id)


def store_borderline_policy(store: Store, dataset_id: str, sentence: str) -> None:
    _write_stats(store, dataset_id, {"borderline_policy": sentence.strip()})


def check_release_quality(store: Store, dataset_id: str) -> dict[str, Any]:
    """Flag borderline rows and refuse a release the language review does not support."""
    rows = _rows(store, dataset_id)
    flagged = 0
    for row in rows:
        if not is_borderline(str(row["payload"])):
            continue
        flagged += 1
        meta = dict(row["meta"])
        meta["borderline"] = True
        store.conn.execute(
            "UPDATE dataset_records SET meta = ? WHERE id = ?",
            (json.dumps(meta, sort_keys=True), row["id"]),
        )
    dataset = store.get_dataset(dataset_id)
    stats = dict(dataset.stats) if dataset is not None else {}
    if flagged and stats.get("borderline_policy") != BORDERLINE_POLICY:
        raise PermissionError(
            f"{flagged} record(s) match the toxicity list; release needs the borderline policy"
        )
    tagged = [row for row in rows if row["meta"].get("language")]
    review = stats.get("language_review") or {}
    if tagged:
        sample = language_sample(store, dataset_id)
        marks = review.get("marks") or {}
        if {row["id"] for row in sample} != set(marks):
            raise PermissionError("a domain expert has not reviewed the language sample")
        wrong = sum(1 for row in sample if marks.get(row["id"]) == "wrong")
        if sample and wrong / len(sample) > WRONG_LIMIT:
            raise PermissionError(
                f"language review marked {wrong} of {len(sample)} wrong, over 10 percent"
            )
    stats["toxicity_flagged"] = flagged
    _write_stats(store, dataset_id, {"toxicity_flagged": flagged})
    return {"toxicity_flagged": flagged, "language_review": review}


def _write_stats(store: Store, dataset_id: str, patch: dict[str, Any]) -> None:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    stats = dict(dataset.stats)
    stats.update(patch)
    store.conn.execute(
        "UPDATE datasets SET stats = ? WHERE id = ?",
        (json.dumps(stats, sort_keys=True, default=str), dataset_id),
    )
