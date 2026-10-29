"""Where a dataset's tokens came from, and the shard files a run can stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from internal_ai.data.records import Record
from internal_ai.data.tokenizer import Tokenizer

SHARD_ROWS = 256


def folder_of(record_key: str) -> str:
    return record_key.split("/", 1)[0] or "root"


def mixture_rows(records: list[Record]) -> list[dict[str, Any]]:
    """Share of tokens in each top folder. Ratios sum to 1."""
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        folder = folder_of(record.record_key)
        bucket = grouped.setdefault(folder, {"folder": folder, "records": 0, "tokens": 0})
        bucket["records"] += 1
        bucket["tokens"] += int(record.tokens or 0) or max(1, len(record.text) // 4)
    total = sum(int(item["tokens"]) for item in grouped.values()) or 1
    rows = []
    for folder in sorted(grouped):
        item = grouped[folder]
        item["ratio"] = round(int(item["tokens"]) / total, 6)
        rows.append(item)
    if rows:
        rows[-1]["ratio"] = round(rows[-1]["ratio"] + (1 - sum(item["ratio"] for item in rows)), 6)
    return rows


def write_shards(rows: list[dict[str, Any]], tokenizer: Tokenizer, directory: Path) -> list[Path]:
    """One jsonl file per 256 records, in record-key order."""
    directory.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: str(row["record_key"]))
    paths: list[Path] = []
    for index, start in enumerate(range(0, len(ordered), SHARD_ROWS)):
        path = directory / f"shard-{index:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in ordered[start : start + SHARD_ROWS]:
                ids = tokenizer.content_ids(tokenizer.encode(str(row["payload"])))
                handle.write(
                    json.dumps({"ids": ids, "record_key": row["record_key"]}, sort_keys=True) + "\n"
                )
        paths.append(path)
    return paths
