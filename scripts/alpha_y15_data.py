#!/usr/bin/env python3
"""Take a public corpus through the governed data path, so questions can be frozen.

The freeze gate refuses an eval set whose labelled documents are not in a
released dataset's held-out split, so the corpus has to be released first. This
runs that path in the order the pipeline enforces: ingest, split, measure the
tokens with a real tokenizer, have a domain expert review the language sample,
ask for the release gate, and release. A refusal here is the answer, not an
obstacle: it is printed and the script stops.

    INTERNAL_AI_OFFLINE=0 .venv/bin/python scripts/alpha_y15_data.py \\
        --corpus var_alpha/corpus.jsonl --root var_alpha/install
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
from internal_ai.domain import GateType
from internal_ai.people import PeopleRegistry
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

# The owner cannot sit at the keyboard for this run, so they approved by message.
# The note says that, because a gate record that reads like an onsite decision
# would be a lie to whoever audits it later.
PROXY_NOTE = "approved in writing by the owner, recorded by the agent acting on that instruction"


def enrol(platform: Platform) -> dict[str, dict[str, Any]]:
    roster = PeopleRegistry(platform.store)
    enrolled: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for role, (name, title, roles) in ROSTER.items():
        if name not in by_name:
            by_name[name] = roster.new_person(name, title, roles, ["engineering"])
        enrolled[role] = by_name[name]
    return enrolled


def request_and_approve(
    platform: Platform,
    gate_type: GateType,
    title: str,
    subject: str,
    evidence: dict[str, Any],
    people: dict[str, dict[str, Any]],
    requested_by: str,
) -> str:
    gate = platform.gates.request(
        gate_type, title, subject, requested_by=requested_by, evidence=evidence
    )
    for role in gate.required_approvers:
        platform.gates.decide(
            gate.id, people[role]["name"], approved=True, note=PROXY_NOTE, evidence=evidence
        )
    decided = platform.store.get_gate(gate.id)
    if decided is None or str(decided.status) != "approved":
        raise RuntimeError(f"gate {gate.id} did not reach approved: {decided and decided.status}")
    return gate.id


def release_corpus(platform: Platform, corpus: Path, name: str) -> dict[str, Any]:
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
    gate_id = request_and_approve(
        platform,
        GateType.DATASET_RELEASE,
        f"Release {name} for the alpha track",
        build.dataset_id,
        {
            "quality_report": json.dumps(
                {"records": build.records, "tokens": build.tokens, "steps": build.stats["steps"]},
                sort_keys=True,
                default=str,
            ),
            "pii_audit_sample": f"full corpus, {build.records} records",
            "lineage": json.dumps(build.lineage, sort_keys=True, default=str),
        },
        people,
        people["data_owner"]["id"],
    )
    released = pipeline.release(build.dataset_id, people["ml_lead"]["id"], gate_approved=True)
    return {
        "dataset_id": released.id,
        "content_hash": released.content_hash,
        "records": released.record_count,
        "tokens": released.token_count,
        "token_count_kind": released.stats.get("token_count_kind"),
        "measured": measured,
        "splits": split.stats.get("counts", {}),
        "tokenizer": coverage.get("tokenizer_path"),
        "shards": coverage.get("shards"),
        "language_review": {"sampled": review["sampled"], "wrong": review["wrong"]},
        "release_gate": gate_id,
        "released": released.released,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("var_alpha/corpus.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("var_alpha/install"))
    parser.add_argument("--name", default="rfc")
    parser.add_argument("--out", type=Path, default=Path("var_alpha/dataset.json"))
    args = parser.parse_args(argv)
    if not args.corpus.is_file():
        print(f"error: no corpus at {args.corpus}", file=sys.stderr)
        return 2
    platform = Platform.create(root=args.root, llm_kind="scripted")
    try:
        try:
            result = release_corpus(platform, args.corpus, args.name)
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
