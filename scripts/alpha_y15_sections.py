#!/usr/bin/env python3
"""Draft a question set from the held-out RFC text, for a person to review.

The Y1.5 set holds 32 questions a person wrote. That is a set the promotion
floor refuses, and rightly: 200 correlated questions are the smallest set on
which a lead can be told from luck. Writing two hundred by hand is how a
question set stops being written at all, and asking a model to write them would
put third-party output in the set the model is judged on.

So the questions are mechanical: a sentence the corpus already carries, a
template chosen by the sentence's own cue, and the sentence as the reference
answer. A reviewer opens the cited line and agrees or drops the question. No
model writes a question, and no model decides whether the set is frozen.

This script therefore stops at a request. It creates the draft set, checks the
answers stay inside the held-out split, and opens the ``eval_set_freeze`` gate.
Deciding that gate is a person's job, and the set is not frozen until they do.

    .venv/bin/python scripts/alpha_y15_sections.py --root var_alpha/install \\
        --dataset ds_9350fb760231 --rfc-dir var_alpha/rfc --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.evals.harness import EvalHarness
from internal_ai.people import PeopleRegistry
from internal_ai.services import build
from internal_ai.services.actor import Actor
from internal_ai.training.alpha.citations import AskHit, score_answer
from internal_ai.training.alpha.sections import Heading, headings, questions, section_for
from internal_ai.workflows import PROMOTION_MIN_CASES


def _held_out(platform: Platform, dataset_id: str) -> list[dict[str, Any]]:
    """The test-split records, in key order so a rebuild is byte-identical."""
    return sorted(
        platform.store.dataset_records_by_split(dataset_id, "test"),
        key=lambda row: row["record_key"],
    )


def build_items(
    platform: Platform,
    dataset_id: str,
    rfc_dir: Path,
    per_record: int = 2,
    limit: int = 240,
) -> list[dict[str, Any]]:
    """One question per usable sentence, never the same question twice.

    A duplicate question with two different answers is an item nobody can
    answer, so the first one wins and the rest are dropped. That is also why the
    count is not the number of sentences: the set has to be checked, not padded.
    """
    cache: dict[str, list[Heading]] = {}
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _held_out(platform, dataset_id):
        path, _, line = row["record_key"].rpartition(":")
        if not line.isdigit() or not path:
            continue
        source = rfc_dir / path
        if not source.is_file():
            continue
        if path not in cache:
            cache[path] = headings(source)
        section = section_for(cache[path], int(line))
        for query, answer in questions(row["payload"], per_record=per_record):
            key = " ".join(query.lower().split())
            if key in seen:
                continue
            hits = [AskHit(path, int(line))]
            if not score_answer(answer, hits).passed:
                continue
            seen.add(key)
            items.append(
                {
                    "query": query,
                    "answer": answer,
                    "relevant_ids": [row["record_key"]],
                    "hits": [{"path": path, "line": int(line), "text": answer}],
                    "dataset_id": dataset_id,
                    "source_id": row.get("source_id") or "rfc",
                    "kind": "citation",
                    "section": section.label if section is not None else "",
                }
            )
        if len(items) >= limit:
            break
    return items[:limit]


def _actor(platform: Platform, role: str) -> Actor | None:
    people = PeopleRegistry(platform.store).list_people()
    real = [p for p in people if role in p["roles"] and not p["placeholder"]]
    return Actor.from_person(real[0], authenticated=True) if real else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("var_alpha/install"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rfc-dir", type=Path, default=Path("var_alpha/rfc"))
    parser.add_argument("--name", default="rfc-sections")
    parser.add_argument("--per-record", type=int, default=2)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--out", type=Path, default=Path("var_alpha/eval_sections.json"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the draft and report the count, touching no gate and no set",
    )
    parser.add_argument("--show", type=int, default=0, help="print this many draft items")
    args = parser.parse_args(argv)

    platform = Platform.create(root=args.root, llm_kind="scripted")
    try:
        items = build_items(
            platform, args.dataset, args.rfc_dir, per_record=args.per_record, limit=args.limit
        )
        draft = {
            "name": args.name,
            "dataset_id": args.dataset,
            "items": len(items),
            "floor": PROMOTION_MIN_CASES,
            "meets_floor": len(items) >= PROMOTION_MIN_CASES,
            "questions": items,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for item in items[: args.show]:
            print(json.dumps(item, indent=2, sort_keys=True))
        print(f"drafted {len(items)} questions into {args.out}", file=sys.stderr)
        if len(items) < PROMOTION_MIN_CASES:
            print(
                f"refused: {len(items)} questions is below the {PROMOTION_MIN_CASES}-case "
                "floor, so a report on this set could not promote anything",
                file=sys.stderr,
            )
            return 1
        if args.dry_run:
            return 0

        # An expert writes the questions and the ml lead decides the freeze, so
        # the requester is never the one who approves their own set.
        actor = _actor(platform, "domain_expert")
        if actor is None:
            print("refused: no account holds the domain_expert role", file=sys.stderr)
            return 1
        harness = EvalHarness(platform.store, platform.audit)
        eval_set = harness.create_eval_set(args.name, items, actor=actor.id)
        gate = build.request_freeze(platform, actor, eval_set.id)
        result = {
            "eval_set_id": eval_set.id,
            "items": len(eval_set.items),
            "content_hash": eval_set.content_hash,
            "gate_id": gate.id,
            "frozen": False,
            "note": (
                "the set is not frozen: deciding eval_set_freeze is a person's job, and "
                "freezing without a decided gate is refused"
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        platform.close()


if __name__ == "__main__":
    raise SystemExit(main())
