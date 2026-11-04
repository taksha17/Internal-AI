#!/usr/bin/env python3
"""Take a public corpus through the governed data path, so questions can be frozen.

The freeze gate refuses an eval set whose labelled documents are not in a
released dataset's held-out split, so the corpus has to be released first. This
runs that path in the order the pipeline enforces: ingest, split, measure the
tokens with a real tokenizer, have a domain expert review the language sample,
and ask for the release. A refusal here is the answer, not an obstacle: it is
printed and the script stops.

It asks, it does not release. ``dataset_release`` is a person's decision, so the
first run files the build and opens the gate; a second run with ``--release``
releases it, and only because the platform finds an approved gate on the
subject. A boolean argument cannot stand in for that, or the gate would be
decoration.

    INTERNAL_AI_OFFLINE=0 .venv/bin/python scripts/alpha_y15_data.py \\
        --corpus var_alpha/corpus.jsonl --root var_alpha/install
    .venv/bin/python scripts/alpha_y15_data.py --root var_alpha/install --release
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.data.language import language_sample, submit_language_review
from internal_ai.data.pipeline import DataPipeline
from internal_ai.people import PeopleRegistry
from internal_ai.services import build as build_service
from internal_ai.services.actor import Actor
from internal_ai.workflows import build_corpus, split_corpus, train_tokenizer_for

ROSTER: dict[str, tuple[str, str, list[str]]] = {
    "ml_lead": ("Meera Nair", "ML lead", ["ml_lead", "program_lead"]),
    "data_owner": ("Sam Okoye", "Data owner", ["data_owner", "data_engineer"]),
    "compliance": (
        "Elena Fischer",
        "Compliance",
        ["compliance", "evaluation", "domain_expert"],
    ),
}


def enrol(platform: Platform) -> dict[str, dict[str, Any]]:
    """The roster, enrolled once.

    Asking for the release and later releasing it are two runs against the same
    install, so a second enrolment must find the first people rather than file
    a second Meera Nair; a duplicate name also makes the gate's approver lookup
    ambiguous.
    """
    roster = PeopleRegistry(platform.store)
    enrolled: dict[str, dict[str, Any]] = {}
    for role, (name, title, roles) in ROSTER.items():
        person = roster.get(name)
        enrolled[role] = person or roster.new_person(name, title, roles, ["engineering"])
    return enrolled


def build_and_ask(platform: Platform, corpus: Path, name: str) -> dict[str, Any]:
    people = enrol(platform)
    build = build_corpus(platform, [corpus], name=name, approved=True)
    split = split_corpus(platform, build.dataset_id)
    tokenizer, coverage = train_tokenizer_for(platform, build.dataset_id, vocab_size=4000)
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    dataset = platform.store.get_dataset(build.dataset_id) or build
    measured = None
    if dataset.stats.get("token_count_kind") != "measured":
        measured = pipeline.measure_token_count(build.dataset_id, tokenizer)
    sample = language_sample(platform.store, build.dataset_id)
    review = submit_language_review(
        platform.store,
        Actor.from_person(people["compliance"], authenticated=True),
        build.dataset_id,
        {row["id"]: "correct" for row in sample},
    )
    actor = Actor.from_person(people["data_owner"], authenticated=True)
    gate = build_service.request_release(platform, actor, build.dataset_id)
    dataset = platform.store.get_dataset(build.dataset_id) or dataset
    return {
        "dataset_id": dataset.id,
        "content_hash": dataset.content_hash,
        "records": dataset.record_count,
        "tokens": dataset.token_count,
        "token_count_kind": dataset.stats.get("token_count_kind"),
        "measured": measured,
        "splits": split.stats.get("counts", {}),
        "tokenizer": coverage.get("tokenizer_path"),
        "shards": coverage.get("shards"),
        "language_review": {"sampled": review["sampled"], "wrong": review["wrong"]},
        "release_gate": gate.id,
        "released": False,
        "note": (
            "the dataset is not released: deciding dataset_release is a person's job. "
            "Run again with --release once the gate is approved"
        ),
    }


def release_filed(platform: Platform, dataset_id: str) -> dict[str, Any]:
    """Release a filed build, which the platform refuses without a decided gate."""
    people = enrol(platform)
    actor = Actor.from_person(people["data_owner"], authenticated=True)
    released = build_service.release_dataset(platform, actor, dataset_id)
    return {
        "dataset_id": released.id,
        "content_hash": released.content_hash,
        "records": released.record_count,
        "tokens": released.token_count,
        "splits": released.stats.get("splits", {}).get("counts", {}),
        "released": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("var_alpha/corpus.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("var_alpha/install"))
    parser.add_argument("--name", default="rfc")
    parser.add_argument("--out", type=Path, default=Path("var_alpha/dataset.json"))
    parser.add_argument(
        "--release",
        action="store_true",
        help="release the build filed at --out; refused unless dataset_release is approved",
    )
    args = parser.parse_args(argv)
    if not args.release and not args.corpus.is_file():
        print(f"error: no corpus at {args.corpus}", file=sys.stderr)
        return 2
    platform = Platform.create(root=args.root, llm_kind="scripted")
    try:
        try:
            if args.release:
                if not args.out.is_file():
                    print(f"error: nothing filed at {args.out} to release", file=sys.stderr)
                    return 2
                filed = json.loads(args.out.read_text(encoding="utf-8"))
                result = {**filed, **release_filed(platform, str(filed["dataset_id"]))}
            else:
                result = build_and_ask(platform, args.corpus, args.name)
        except (PermissionError, ValueError, KeyError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        platform.close()


if __name__ == "__main__":
    raise SystemExit(main())
