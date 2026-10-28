"""What would change a failed result, read off the numbers the run produced.

A refusal is not a next step. These rules turn measurements a run already
recorded into the sentence a person needs, and each sentence names the
measurement it rests on. Where the numbers do not support a cause, the advice
says what is measurable and stops rather than guessing.

Deterministic on purpose: the same report always reads the same way.
"""

from __future__ import annotations

from typing import Any

from internal_ai.brief import describe_verdict
from internal_ai.data.suggest import GENERATE_RECORDS

# A run can fail its safety checks while scoring well, so the score is not the
# first thing to read and must not be blamed for it.
CHECK_ACTIONS: dict[str, str] = {
    "contamination": (
        "The contamination check did not pass: held-out documents appear in the "
        "training corpus. Rebuild the dataset's split so they do not, then train again."
    ),
    "pii_regurgitation": (
        "The PII check did not pass: the model surfaced records that still carried "
        "personal data. Scrub the dataset and release a new version before training again."
    ),
    "permission_leak": "The permission check did not pass. Settle that before reading the score.",
}

# Reported for context, never a reason to refuse on its own.
INFORMATIONAL_CHECKS = frozenset({"baseline_informativeness"})


def _value(source: Any, key: str, default: Any = None) -> Any:
    """One field of a report, whether it arrived as a record or a plain mapping."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def failed_checks(report: Any) -> list[str]:
    """The safety checks that did not pass, by name, in a stable order."""
    checks = _value(report, "checks", {}) or {}
    return sorted(
        str(name)
        for name, result in checks.items()
        if name not in INFORMATIONAL_CHECKS and not _value(result, "passed", True)
    )


def retrieval_advice(report: Any, records: int, tokens: int) -> list[str]:
    """Why a retrieval run did not pass, and what would change it."""
    if report is None or _value(report, "passed", False):
        return []
    failed = [name for name in failed_checks(report) if name != "brief_targets"]
    if failed:
        return [CHECK_ACTIONS.get(name, f"The {name} check did not pass.") for name in failed]
    targets = (_value(report, "checks", {}) or {}).get("brief_targets") or {}
    if isinstance(targets, dict) and targets.get("passed") is False:
        lines = [
            f"The project brief sets a bar this run did not reach: {describe_verdict(verdict)}."
            for verdict in targets.get("targets", [])
            if not verdict.get("met")
        ]
        lines.append(
            "Either the model is short of the target or the evaluation does not test "
            "what the brief asks; settle which before promotion."
        )
        return lines

    metrics = _value(report, "metrics", {}) or {}
    baseline = _value(report, "baseline_metrics", {}) or {}
    lines: list[str] = []
    if "recall_at_10" in metrics and "recall_at_10" in baseline:
        lines.append(
            f"The model scored recall@10 {metrics['recall_at_10']:.3f} against the "
            f"keyword-search baseline's {baseline['recall_at_10']:.3f}."
        )
    if records < GENERATE_RECORDS:
        lines.append(
            f"At {records:,} records and {tokens:,} tokens, keyword search is expected to "
            f"win: about {GENERATE_RECORDS:,} usable records is where a model trained from "
            "scratch can compete (PRD section 10)."
        )
        lines.append(
            "Search already answers questions over this data, so keep using it until the "
            "corpus grows. Another loop at this size repeats this result."
        )
    else:
        lines.append(
            f"The corpus is past the volume bar ({records:,} records), so more loops are "
            "unlikely to be the gap."
        )
        lines.append(
            "Read what the dataset kept: its lineage names how much the dedup and quality "
            "steps removed. Duplicate and boilerplate text is the usual cause at this size."
        )
    return lines


def classifier_advice(report: Any) -> list[str]:
    """Why a classifier did not pass, and what would change it."""
    if report is None or _value(report, "passed", False):
        return []
    accuracy = float(_value(report, "accuracy", 0.0) or 0.0)
    baseline = float(_value(report, "baseline_accuracy", 0.0) or 0.0)
    floor = float(_value(report, "min_accuracy", 0.0) or 0.0)
    distribution = {
        str(label): int(count)
        for label, count in (_value(report, "label_distribution", {}) or {}).items()
    }
    lines: list[str] = []
    if accuracy <= baseline:
        lines.append(
            f"It answered no better than always saying the most common label: accuracy "
            f"{accuracy:.3f} against a majority-class baseline of {baseline:.3f}."
        )
        lines.append("More answers per label is what changes this, not more epochs.")
    else:
        lines.append(
            f"It beats always guessing but is under the floor: accuracy {accuracy:.3f} "
            f"against a floor of {floor:.3f}."
        )
    if len(distribution) == 1:
        lines.append("Only one label is present, so there is nothing for a classifier to separate.")
    elif distribution:
        weakest = min(sorted(distribution), key=lambda label: distribution[label])
        lines.append(
            f"The least-represented label, {weakest}, has {distribution[weakest]:,} of "
            f"{sum(distribution.values()):,} rows."
        )
    return lines
