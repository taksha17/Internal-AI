"""Train, validation and test splits with a leakage guard (FR-8)."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from internal_ai.data.records import Record
from internal_ai.data.transforms import hamming, simhash


@dataclass(slots=True)
class SplitResult:
    splits: dict[str, list[Record]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _bucket(key: str, seed: str) -> float:
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _record_bucket(record: Record, seed: str) -> float:
    return _bucket(record.record_key, seed)


def _near_lookup(fingerprints: list[int], threshold: int) -> Callable[[int], bool]:
    """Whether a SimHash is within ``threshold`` of any indexed one, exactly.

    Comparing a held-out record against every training record is quadratic, and at
    the size of the PMC corpus it is 1.7e9 comparisons -- hours of work in a step
    that runs immediately before a release. Splitting the 64-bit hash into
    ``threshold + 1`` blocks makes the lookup exact rather than sampled: two hashes
    within ``threshold`` bits differ in at most ``threshold`` blocks, so they agree
    on at least one block and the pair is still found.
    """
    blocks = min(8, max(1, threshold + 1))
    width = 64 // blocks
    mask = (1 << width) - 1
    index: dict[tuple[int, int], list[int]] = {}
    for position, value in enumerate(fingerprints):
        for block in range(blocks):
            key = (block, (value >> (block * width)) & mask)
            index.setdefault(key, []).append(position)

    def within(fingerprint: int) -> bool:
        for block in range(blocks):
            key = (block, (fingerprint >> (block * width)) & mask)
            for position in index.get(key, ()):
                if hamming(fingerprint, fingerprints[position]) <= threshold:
                    return True
        return False

    return within


def make_splits(
    records: list[Record],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: str = "forge-v1",
    leakage_threshold: int = 3,
    group_of: Callable[[Record], str] | None = None,
    test_groups: set[str] | None = None,
) -> SplitResult:
    """Split records deterministically and drop cross-split near duplicates.

    The default seed is kept as it was issued before the rename: it is the
    identifier the released dataset versions were split with, and changing it
    would silently produce a different assignment for every corpus.

    ``group_of`` holds whole groups out together, hashing the group's name
    instead of each record's. Splitting paragraphs lets a held-out line sit next
    to its neighbour in the training text, so a model can answer a question
    about it by having read the paragraph beside it; a document-level holdout is
    the only version of that test a reader can believe. ``test_groups`` names
    groups a person chose to hold out, because with a handful of documents the
    bucket alone can put every small one in test and leave nothing to ask about.
    """
    train_ratio, val_ratio, _ = ratios
    forced = set(test_groups or ())
    splits: dict[str, list[Record]] = {"train": [], "validation": [], "test": []}
    groups: Counter[str] = Counter()
    for record in records:
        if group_of is None:
            value = _record_bucket(record, seed)
            group = ""
        else:
            group = group_of(record)
            groups[group] += 1
            if group in forced:
                splits["test"].append(record)
                continue
            value = _bucket(group, seed)
        if value < train_ratio:
            splits["train"].append(record)
        elif value < train_ratio + val_ratio:
            splits["validation"].append(record)
        else:
            splits["test"].append(record)

    train_fingerprints = [simhash(record.text) for record in splits["train"]]
    train_exact = {record.fingerprint() for record in splits["train"]}
    overlapping = _near_lookup(train_fingerprints, leakage_threshold)
    removed: Counter[str] = Counter()
    for name in ("validation", "test"):
        kept: list[Record] = []
        for record in splits[name]:
            if record.fingerprint() in train_exact:
                removed[f"{name}_exact_overlap"] += 1
                continue
            if overlapping(simhash(record.text)):
                removed[f"{name}_near_overlap"] += 1
                continue
            kept.append(record)
        splits[name] = kept

    stats: dict[str, Any] = {
        "seed": seed,
        "ratios": list(ratios),
        "counts": {name: len(items) for name, items in splits.items()},
        "contamination_removed": dict(sorted(removed.items())),
        "contamination_total": sum(removed.values()),
        "leakage_guard": "simhash_hamming",
    }
    if group_of is not None:
        stats["grouped_by"] = "document"
        stats["groups"] = dict(sorted(groups.items()))
        stats["test_groups"] = sorted(forced)
    return SplitResult(splits=splits, stats=stats)


def contamination_report(
    eval_records: list[Record],
    training_records: list[Record],
    threshold: int = 3,
) -> dict[str, Any]:
    """Check an eval set against a training corpus (FR-8, FR-14)."""
    train_fingerprints = [simhash(record.text) for record in training_records]
    train_exact = {record.fingerprint() for record in training_records}
    overlapping = _near_lookup(train_fingerprints, threshold)
    exact_hits = 0
    near_hits = 0
    for record in eval_records:
        if record.fingerprint() in train_exact:
            exact_hits += 1
            continue
        if overlapping(simhash(record.text)):
            near_hits += 1
    total = len(eval_records)
    contaminated = exact_hits + near_hits
    return {
        "eval_records": total,
        "exact_overlap": exact_hits,
        "near_overlap": near_hits,
        "contaminated_records": contaminated,
        "contamination_rate": round(contaminated / total, 6) if total else 0.0,
        "passed": contaminated == 0,
        "threshold": threshold,
    }
