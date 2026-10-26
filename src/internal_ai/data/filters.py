"""What an inspection skips: built-in noise, then the platform's own rules.

These rules change what the console reads when someone inspects a path. They do
not change what a dataset build reads, and the scan that goes to a gate records
them, so an approver can see what was left out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MAX_RULES = 40
MAX_RULE_LENGTH = 40


@dataclass(frozen=True, slots=True)
class ScanFilters:
    folders: frozenset[str] = frozenset()
    suffixes: frozenset[str] = frozenset()

    def skips_folder(self, name: str) -> bool:
        return name.lower() in self.folders

    def skips_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    def to_dict(self) -> dict[str, list[str]]:
        return {"folders": sorted(self.folders), "suffixes": sorted(self.suffixes)}


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _folder(item: str) -> str:
    if len(item) > MAX_RULE_LENGTH:
        raise ValueError(f"folder names are limited to {MAX_RULE_LENGTH} characters: {item}")
    if "/" in item or "\\" in item:
        raise ValueError(f"give a folder name, not a path: {item}")
    return item.lower()


def _suffix(item: str) -> str:
    cleaned = item.lower().lstrip("*").strip()
    if not cleaned or cleaned == ".":
        raise ValueError("a file type cannot be empty")
    if not cleaned.startswith("."):
        cleaned = "." + cleaned
    if not cleaned[1:].isalnum() or len(cleaned) > 12:
        raise ValueError(f"give a file extension such as .bin: {item}")
    return cleaned


def parse_filters(folders: str, suffixes: str) -> ScanFilters:
    folder_rules = {_folder(item) for item in _split(folders)}
    suffix_rules = {_suffix(item) for item in _split(suffixes)}
    if len(folder_rules) > MAX_RULES or len(suffix_rules) > MAX_RULES:
        raise ValueError(f"at most {MAX_RULES} rules of each kind; keep the list short")
    return ScanFilters(frozenset(folder_rules), frozenset(suffix_rules))


def load_filters(raw: str | None) -> ScanFilters:
    if not raw:
        return ScanFilters()
    try:
        stored = json.loads(raw)
    except ValueError:
        return ScanFilters()
    if not isinstance(stored, dict):
        return ScanFilters()
    folders = _rules(stored, "folders")
    suffixes = _rules(stored, "suffixes")
    return ScanFilters(folders, suffixes)


def _rules(stored: dict, key: str) -> frozenset[str]:
    return frozenset(str(item).strip().lower() for item in stored.get(key, []) if str(item).strip())


def dump_filters(filters: ScanFilters) -> str:
    return json.dumps(filters.to_dict(), sort_keys=True)
