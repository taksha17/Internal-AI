"""Normalization, deduplication, PII scrubbing and quality filtering.

Each step reports per-step statistics so the data quality report can show where
records were lost (FR-6).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from internal_ai.data.inventory import SECRET_PATTERNS
from internal_ai.data.records import Record

# --------------------------------------------------------------------- normalize

WHITESPACE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
    normalized = WHITESPACE.sub(" ", normalized)
    normalized = BLANK_LINES.sub("\n\n", normalized)
    lines = [line.strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def normalize(records: list[Record]) -> tuple[list[Record], dict[str, Any]]:
    kept: list[Record] = []
    changed = 0
    for record in records:
        cleaned = normalize_text(record.text)
        if cleaned != record.text:
            changed += 1
        record.text = cleaned
        if cleaned:
            kept.append(record)
    stats = {
        "input": len(records),
        "output": len(kept),
        "normalized": changed,
        "empty_dropped": len(records) - len(kept),
    }
    return kept, stats


# ------------------------------------------------------------------ deduplicate


def shingles(text: str, size: int = 5) -> set[str]:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def simhash(text: str, size: int = 5) -> int:
    vector = [0] * 64
    for shingle in shingles(text, size):
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += 1 if (digest >> bit) & 1 else -1
    fingerprint = 0
    for bit in range(64):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


@dataclass(slots=True)
class DedupResult:
    records: list[Record]
    stats: dict[str, Any] = field(default_factory=dict)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def deduplicate(
    records: list[Record],
    threshold: int = 12,
    jaccard_threshold: float = 0.8,
) -> tuple[list[Record], dict[str, Any]]:
    """Exact and near-duplicate removal.

    Candidates come from banded SimHash lookup, then a Jaccard check over
    character shingles confirms the match so the looser hash radius does not
    produce false positives.
    """
    seen_exact: set[str] = set()
    fingerprints: list[tuple[int, int]] = []
    shingles_by_index: list[set[str]] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    kept: list[Record] = []
    exact_removed = 0
    near_removed = 0
    bands = 8
    band_bits = 64 // bands
    for index, record in enumerate(records):
        digest = record.fingerprint()
        if digest in seen_exact:
            exact_removed += 1
            continue
        seen_exact.add(digest)
        fingerprint = simhash(record.text)
        document_shingles = shingles(record.text)
        duplicate = False
        candidates: set[int] = set()
        for band in range(bands):
            key = (band, (fingerprint >> (band * band_bits)) & ((1 << band_bits) - 1))
            for candidate in buckets.get(key, []):
                if hamming(fingerprint, fingerprints[candidate][0]) <= threshold:
                    candidates.add(candidate)
        for candidate in candidates:
            if jaccard(document_shingles, shingles_by_index[candidate]) >= jaccard_threshold:
                duplicate = True
                break
        if duplicate:
            near_removed += 1
            continue
        fingerprints.append((fingerprint, index))
        shingles_by_index.append(document_shingles)
        for band in range(bands):
            key = (band, (fingerprint >> (band * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(len(fingerprints) - 1)
        kept.append(record)
    stats = {
        "input": len(records),
        "output": len(kept),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "duplicate_rate": round((exact_removed + near_removed) / len(records), 6)
        if records
        else 0.0,
        "hamming_radius": threshold,
        "jaccard_threshold": jaccard_threshold,
    }
    return kept, stats


def duplicate_report(
    records: list[Record],
    threshold: int = 12,
    jaccard_threshold: float = 0.8,
    residual_max: float = 0.01,
) -> dict[str, Any]:
    """Count exact copies and the near-duplicate residual the release gate uses.

    Exact groups always fail a release. Near copies are the same SimHash pass
    ``deduplicate`` already runs, and they fail only above ``residual_max``.
    """
    seen: dict[str, str] = {}
    grouped: set[str] = set()
    groups = 0
    pairs: list[list[str]] = []
    for record in records:
        digest = record.fingerprint()
        prior = seen.get(digest)
        if prior is None:
            seen[digest] = record.record_key
            continue
        if digest not in grouped:
            groups += 1
            grouped.add(digest)
        if len(pairs) < 5:
            pairs.append([prior, record.record_key])
    _, stats = deduplicate(records, threshold=threshold, jaccard_threshold=jaccard_threshold)
    return {
        "exact_duplicate_groups": groups,
        "near_duplicate_pairs": stats["near_duplicates_removed"],
        "residual_rate": stats["duplicate_rate"],
        "pairs": pairs,
        "residual_max": residual_max,
    }


# ----------------------------------------------------------------------- scrub

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private_key": re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    ),
    "api_token": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "url_credentials": re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"),
    "mac_address": re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    "password_assignment": SECRET_PATTERNS["password_assignment"],
    "connection_string": SECRET_PATTERNS["connection_string"],
}

REDACTION_TEMPLATE = "[REDACTED:{kind}]"

Luhn_REFERENCE = {0, 2, 4, 6, 8}


def _luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or len(digits) < 13:
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


ISBN_PREFIXES = ("978", "979")


def is_credit_card(text: str) -> bool:
    """A number a card network would issue, not a digit run that happens to fit.

    Luhn alone is not enough on technical prose: a dotted-decimal example and a
    book's ISBN both look like candidates, and the audit used to flag them while
    the scrubber quietly ignored them. ISBN-13 has the same length but starts
    with 978 or 979, which no issuer uses, so those are read as books.
    """
    digits = re.sub(r"\D", "", text)
    if not 13 <= len(digits) <= 19:
        return False
    if len(digits) == 13 and digits.startswith(ISBN_PREFIXES):
        return False
    return _luhn_valid(digits)


def scrub_pii(
    records: list[Record], detectors: list[str] | None = None
) -> tuple[list[Record], dict[str, Any]]:
    selected = detectors or list(PII_PATTERNS)
    counters: Counter[str] = Counter()
    touched = 0
    for record in records:
        original = record.text
        for kind in selected:
            pattern = PII_PATTERNS[kind]
            if kind == "credit_card":

                def replace_card(match: re.Match[str]) -> str:
                    if is_credit_card(match.group(0)):
                        counters["credit_card"] += 1
                        return REDACTION_TEMPLATE.format(kind="credit_card")
                    return match.group(0)

                record.text = pattern.sub(replace_card, record.text)
                continue
            hits = len(pattern.findall(record.text))
            if hits:
                counters[kind] += hits
                record.text = pattern.sub(REDACTION_TEMPLATE.format(kind=kind), record.text)
        if record.text != original:
            touched += 1
            record.meta.setdefault("scrubbed", []).extend(
                [kind for kind, count in counters.items() if count]
            )
    stats = {
        "records_scanned": len(records),
        "records_modified": touched,
        "redactions_by_type": dict(sorted(counters.items())),
        "total_redactions": sum(counters.values()),
    }
    return records, stats


def secret_scan(text: str) -> list[str]:
    """Detect secrets in text without modifying it, for pre-ingest checks."""
    return [kind for kind, pattern in PII_PATTERNS.items() if pattern.search(text)]


def detector_recall() -> dict[str, Any]:
    """How often the planted email, API token and private key are found and removed.

    This is a fixture, not a claim about a customer's corpus. A detector that
    misses its own planted sample scores below 1.0.
    """
    from internal_ai.data.records import Record

    token = "sk-" + "abcdefghijklmnopqrstuvwxyz"
    header = "-----BEGIN " + "PRIVATE KEY-----"
    footer = "-----END " + "PRIVATE KEY-----"
    planted = {
        "email": "reach the owner at ada@example.com about the invoice today",
        "api_token": f"the deploy token is {token} today",
        "private_key": f"{header}\nabc\n{footer}",
    }
    per_detector: dict[str, float] = {}
    for kind, text in planted.items():
        found = kind in secret_scan(text)
        scrubbed, _stats = scrub_pii(
            [Record(id="planted", record_key=kind, text=text, source_id="fixture")]
        )
        removed = kind not in secret_scan(scrubbed[0].text)
        per_detector[kind] = 1.0 if found and removed else 0.0
    return {"planted": len(planted), "per_detector": per_detector}


# --------------------------------------------------------------------- quality

ENGLISH_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "has",
    "was",
    "were",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "her",
    "our",
    "out",
    "day",
    "get",
    "his",
    "how",
    "man",
    "new",
    "now",
    "see",
    "two",
    "way",
    "who",
    "boy",
    "did",
    "its",
    "let",
    "put",
    "say",
    "she",
    "too",
    "use",
}

BOILERPLATE_MARKERS = (
    "copyright",
    "all rights reserved",
    "terms of service",
    "privacy policy",
    "cookie policy",
    "unsubscribe",
    "click here",
    "javascript is disabled",
    "enable javascript",
)


def _language_guess(text: str) -> str:
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)
    if not words:
        return "unknown"
    hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    ratio = hits / len(words)
    if ratio > 0.08:
        return "en"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    if any("\u0400" <= char <= "\u04ff" for char in text):
        return "ru"
    if any("\u0600" <= char <= "\u06ff" for char in text):
        return "ar"
    return "unknown"


def _boilerplate_ratio(text: str) -> float:
    lowered = text.lower()
    if not lowered:
        return 1.0
    markers = sum(1 for marker in BOILERPLATE_MARKERS if marker in lowered)
    return markers / 8


def filter_quality(
    records: list[Record],
    min_chars: int = 40,
    max_chars: int = 200_000,
    allowed_languages: tuple[str, ...] = ("en",),
    min_alpha_ratio: float = 0.55,
    max_boilerplate_ratio: float = 0.25,
) -> tuple[list[Record], dict[str, Any]]:
    kept: list[Record] = []
    reasons: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for record in records:
        text = record.text
        language = _language_guess(text)
        languages[language] += 1
        if len(text) < min_chars:
            reasons["too_short"] += 1
            continue
        if len(text) > max_chars:
            reasons["too_long"] += 1
            continue
        letters = sum(1 for char in text if char.isalpha())
        alpha_ratio = letters / max(len(text), 1)
        if alpha_ratio < min_alpha_ratio:
            reasons["low_alpha_ratio"] += 1
            continue
        if _boilerplate_ratio(text) > max_boilerplate_ratio:
            reasons["boilerplate"] += 1
            continue
        if allowed_languages and language not in allowed_languages:
            reasons["language_filtered"] += 1
            continue
        record.meta["language"] = language if language in {"en", "unknown"} else "other"
        kept.append(record)
    stats = {
        "input": len(records),
        "output": len(kept),
        "dropped_by_reason": dict(sorted(reasons.items())),
        "language_distribution": dict(sorted(languages.items())),
    }
    return kept, stats
