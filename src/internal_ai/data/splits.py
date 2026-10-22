"""Train, validation and test splits with a leakage guard (FR-8)."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from internal_ai.data.records import Record
from internal_ai.data.transforms import hamming, simhash


@dataclass(slots=True)
class SplitResult:
    splits: dict[str, list[Record]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _bucket(record: Record, seed: str) -> float:
    digest = hashlib.blake2b(f"{seed}:{record.record_key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def make_splits(
    records: list[Record],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: str = "forge-v1",
    leakage_threshold: int = 3,
) -> SplitResult:
    """Split records deterministically and drop cross-split near duplicates."""
    train_ratio, val_ratio, _ = ratios
    splits: dict[str, list[Record]] = {"train": [], "validation": [], "test": []}
    for record in records:
        value = _bucket(record, seed)
        if value < train_ratio:
            splits["train"].append(record)
        elif value < train_ratio + val_ratio:
            splits["validation"].append(record)
        else:
            splits["test"].append(record)

    train_fingerprints = [simhash(record.text) for record in splits["train"]]
    train_exact = {record.fingerprint() for record in splits["train"]}
    removed: Counter[str] = Counter()
    for name in ("validation", "test"):
        kept: list[Record] = []
        for record in splits[name]:
            if record.fingerprint() in train_exact:
                removed[f"{name}_exact_overlap"] += 1
                continue
            fingerprint = simhash(record.text)
            if any(
                hamming(fingerprint, candidate) <= leakage_threshold
                for candidate in train_fingerprints
            ):
                removed[f"{name}_near_overlap"] += 1
                continue
            kept.append(record)
        splits[name] = kept

    stats = {
        "seed": seed,
        "ratios": list(ratios),
        "counts": {name: len(items) for name, items in splits.items()},
        "contamination_removed": dict(sorted(removed.items())),
        "contamination_total": sum(removed.values()),
        "leakage_guard": "simhash_hamming",
    }
    return SplitResult(splits=splits, stats=stats)


def contamination_report(
    eval_records: list[Record],
    training_records: list[Record],
    threshold: int = 3,
) -> dict[str, Any]:
    """Check an eval set against a training corpus (FR-8, FR-14)."""
    train_fingerprints = [simhash(record.text) for record in training_records]
    train_exact = {record.fingerprint() for record in training_records}
    exact_hits = 0
    near_hits = 0
    for record in eval_records:
        if record.fingerprint() in train_exact:
            exact_hits += 1
            continue
        fingerprint = simhash(record.text)
        if any(hamming(fingerprint, candidate) <= threshold for candidate in train_fingerprints):
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
