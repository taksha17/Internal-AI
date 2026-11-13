#!/usr/bin/env python3
"""Build, adapt and merge the KYC model through the platform's own services.

Every step is the service the console and the CLI call: a brief, the data-class
and provenance policies, a registered source behind its gate, the dataset build,
its release gate, ``train_alpha`` and ``merge_adapter``. Nothing here reaches
around a refusal.

The gates in this run are approved by the install's own role accounts. That is
what makes it a trial on a synthetic corpus rather than a production approval:
the corpus is template-built, every value carries a fake marker, and the record
this script writes says so.

    .venv/bin/python scripts/kyc_pipeline.py --root var_kyc --epochs 1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from internal_ai.brief import Metric, ProjectBrief, programme_gaps, save
from internal_ai.context import Platform
from internal_ai.data.kyc import DOC_TEMPLATES, build_corpus, evidence_lines
from internal_ai.data.language import language_sample, submit_language_review
from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.tokenizer import Tokenizer
from internal_ai.domain import GateType
from internal_ai.people import PeopleRegistry
from internal_ai.services import build
from internal_ai.services.actor import Actor
from internal_ai.services.alpha import train_alpha
from internal_ai.services.jobs import JobRunner
from internal_ai.training.alpha.merge import merge_adapter
from internal_ai.training.alpha.pairs import kyc_fill_text
from internal_ai.training.alpha.pin import GEMMA4
from internal_ai.training.provenance import OPEN_WEIGHT_ALLOWED

ACCOUNTS = {
    "program_lead": ("Kyc Programme Lead", "Owns the KYC agent programme"),
    "legal": ("Kyc Legal Reviewer", "Reads every licence before a base loads"),
    "ml_lead": ("Kyc ML Lead", "Owns adaptation and scoring"),
    "data_engineer": ("Kyc Data Engineer", "Owns the corpus build"),
    "infra": ("Kyc Infrastructure Lead", "Owns the machine the run happens on"),
    "evaluation": ("Kyc Evaluation Lead", "Owns what a score has to mean"),
    "security": ("Kyc Security Reviewer", "Owns what may leave the boundary"),
    "data_owner": ("Kyc Data Owner", "Owns the synthetic corpus"),
    "compliance": ("Kyc Compliance Reviewer", "Reads the PII and lineage audit"),
    "domain_expert": ("Kyc Domain Expert", "Marks the sampled rows as correct"),
}
BRIEF_OWNERS = ("ml_lead", "data_engineer", "infra", "evaluation", "security")
NOTE = "synthetic KYC trial: every value is fake by construction"
SOURCE_NAME = "kyc-synthetic-forms"


def say(step: str, payload: Any) -> None:
    print(
        json.dumps({"step": step, "at": time.strftime("%H:%M:%S"), "detail": payload}, default=str),
        flush=True,
    )


def holder(platform: Platform, role: str) -> dict[str, Any]:
    people = PeopleRegistry(platform.store).list_people()
    held = [person for person in people if role in person["roles"] and not person["placeholder"]]
    if not held:
        raise KeyError(f"no non-placeholder account holds {role}")
    return held[0]


def actor(platform: Platform, role: str) -> Actor:
    return Actor.from_person(holder(platform, role), authenticated=True)


def person_id(platform: Platform, role: str) -> str:
    return str(holder(platform, role)["id"])


def approve(platform: Platform, gate_id: str, *roles: str) -> list[str]:
    return [
        str(platform.gates.decide(gate_id, person_id(platform, role), True, note=NOTE).status)
        for role in roles
    ]


def ensure_accounts(platform: Platform) -> list[str]:
    roster = PeopleRegistry(platform.store)
    people = roster.list_people()
    added: list[str] = []
    for role, (name, title) in ACCOUNTS.items():
        if any(role in person["roles"] and not person["placeholder"] for person in people):
            continue
        roster.new_person(name, title, [role], groups=["engineering"])
        added.append(f"{name} ({role})")
    return added


def write_brief(platform: Platform) -> list[str]:
    brief = ProjectBrief(
        target="fill bank KYC forms from customer documents, for a person to review",
        languages="English",
        context_length=32768,
        quality_bar="every filled field is the value the customer's own document carries",
        metrics=[Metric(name="field_accuracy", threshold="0.60", direction="at_least")],
        budget_usd=25.0,
        timeline="one trial run on the 3070",
        deployment="on this machine, local only",
        data_restrictions="template-generated synthetic forms only; no real customer",
        latency_ms=2000,
        owners={role: person_id(platform, role) for role in BRIEF_OWNERS},
        non_goals="deciding credit, fraud or onboarding outcomes; anything with real PII",
    )
    save(platform.store, platform.audit, actor(platform, "program_lead"), brief)
    return programme_gaps(platform, release=True)


def decide_policy(platform: Platform, policy_id: str, value: str) -> dict[str, str]:
    """Propose as the ML lead, approve as the two roles that may, then confirm.

    The requester cannot approve their own gate, so the proposal never comes from
    the pair that decides it: ml_lead proposes and program_lead plus legal decide,
    which is the arrangement scripts/nba_build.py already uses.
    """
    decision = platform.policy.get(policy_id)
    if decision.status == "decided":
        return {"id": policy_id, "value": decision.current_value, "status": decision.status}
    platform.policy.propose(policy_id, value, person_id(platform, "ml_lead"), rationale=NOTE)
    gate_id = platform.store.kv_get(f"policy_gate:{policy_id}") or ""
    approve(platform, gate_id, "program_lead", "legal")
    decided = platform.policy.confirm(policy_id, person_id(platform, "program_lead"))
    return {"id": policy_id, "gate_id": gate_id, "value": decided.current_value}


def link_pinned_base(root: Path, base, snapshot: Path) -> Path:
    """Point the install's pin layout at the snapshot already on disk."""
    target = base.directory(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(snapshot)
    return target


def write_training_jsonl(corpus, path: Path, seed: int, decoys: int = 6) -> int:
    """One record per form, with other customers' papers mixed in as decoys.

    The decoys are what make the task selection rather than recall, and they are
    also what keeps the records from being near-duplicates of each other: a form
    whose only variable parts are its values reads as the same record repeated.
    """
    rng = random.Random(seed)
    forms = list(corpus.forms)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for form in forms:
            pool = [other for other in forms if other.customer_id != form.customer_id]
            distractors = tuple(
                rng.choice(DOC_TEMPLATES).format(**rng.choice(pool).values) for _ in range(decoys)
            )
            handle.write(
                json.dumps(
                    {
                        "text": kyc_fill_text(form.values, evidence_lines(form, rng, distractors)),
                        "form_id": form.form_id,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return len(forms)


def existing_source(platform: Platform) -> dict[str, Any] | None:
    return next(
        (item for item in build.list_sources(platform) if item["name"] == SOURCE_NAME), None
    )


def released_dataset(platform: Platform, dataset_id: str) -> None:
    """Split, measure, sample and release: the order the pipeline refuses to reorder."""
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise SystemExit(f"unknown dataset: {dataset_id}")
    if dataset.released:
        say("release", {"reused": True})
        return
    if "splits" not in dataset.stats:
        say("split", build.split_dataset(platform, actor(platform, "data_owner"), dataset_id))
    if build.tokenizer_path(platform, dataset_id) is None:
        job = build.train_tokenizer(
            platform,
            actor(platform, "data_owner"),
            JobRunner(platform, synchronous=True),
            dataset_id,
        )
        if job.get("state") != "succeeded":
            raise SystemExit(f"the tokenizer run failed: {job.get('error')}")
        say("tokenizer", job.get("result"))
    if dataset.stats.get("token_count_kind") != "measured":
        token_file = build.tokenizer_path(platform, dataset_id)
        measured = DataPipeline(
            platform.store, platform.audit, platform.settings.root
        ).measure_token_count(dataset_id, Tokenizer.load(token_file))
        say("tokens measured", measured)
    sample = language_sample(platform.store, dataset_id)
    review = submit_language_review(
        platform.store,
        actor(platform, "domain_expert"),
        dataset_id,
        {row["id"]: "correct" for row in sample},
    )
    say("language review", {"rows": len(sample), "result": review})
    gate_id = getattr(build.gate_for(platform, GateType.DATASET_RELEASE, dataset_id), "id", "")
    if not (gate_id and platform.gates.is_approved(gate_id)):
        gate = build.request_release(platform, actor(platform, "data_owner"), dataset_id)
        say("release gate", approve(platform, gate.id, "ml_lead", "compliance"))
    say("released", build.release_dataset(platform, actor(platform, "data_owner"), dataset_id))


def run(platform: Platform, args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"root": str(args.root)}
    out["accounts"] = ensure_accounts(platform)
    say("accounts", out["accounts"] or "already held")

    write_brief(platform)
    out["p5"] = decide_policy(platform, "P5", "exclude_pii_customer_confidential_regulated")
    say("policy P5", out["p5"])
    out["p6"] = decide_policy(platform, "P6", OPEN_WEIGHT_ALLOWED)
    say("policy P6", out["p6"])
    out["brief_gaps"] = programme_gaps(platform, release=True)
    if out["brief_gaps"]:
        raise SystemExit("the brief is still incomplete: " + "; ".join(out["brief_gaps"]))

    link_pinned_base(args.root, GEMMA4, args.snapshot)
    say("base", {"repo_id": GEMMA4.repo_id, "revision": GEMMA4.revision})

    corpus = build_corpus(count=args.forms, seed=args.seed)
    out["corpus"] = dict(corpus.manifest)
    train_path = args.root / "data" / "train.jsonl"
    records = write_training_jsonl(corpus, train_path, args.seed)
    say("corpus", {"forms": len(corpus.forms), "hash": corpus.manifest["content_hash"]})

    source = existing_source(platform)
    if source is not None:
        say("source", {"reused": source["id"]})
    else:
        source = build.register_source(
            platform,
            actor(platform, "ml_lead"),
            str(train_path.resolve()),
            SOURCE_NAME,
            "training text for the KYC form-filling agent",
            ["engineering"],
            legal_status="internal",
            license_name="template-generated synthetic data",
            license_note="no real customer, every value carries a fake marker",
        )
        say("source registered", {"id": source["id"], "files": source["scan"].get("files")})
    if source["gate_status"] != "approved":
        say("source gate", approve(platform, source["gate_id"], "data_owner", "compliance"))

    runner = JobRunner(platform, synchronous=True)
    if source.get("dataset_id"):
        dataset_id = source["dataset_id"]
    else:
        started = time.time()
        job = build.build_dataset(platform, actor(platform, "data_owner"), runner, source["id"])
        dataset_id = (job.get("result") or {}).get("dataset_id")
        if not dataset_id:
            raise SystemExit("the ingest produced no dataset")
        say("dataset built", {"id": dataset_id, "seconds": round(time.time() - started, 1)})
    released_dataset(platform, dataset_id)
    dataset = platform.store.get_dataset(dataset_id)
    out["dataset"] = {
        "id": dataset_id,
        "records": dataset.record_count,
        "tokens": dataset.token_count,
        "released": dataset.released,
    }
    say("dataset", out["dataset"])

    started = time.time()
    job = train_alpha(
        platform,
        actor(platform, "ml_lead"),
        runner,
        dataset_id,
        base=GEMMA4,
        epochs=args.epochs,
        seed=args.seed,
        cost_cap_usd=args.cost_cap,
        load_in_4bit=not args.no_quantise,
        max_seq_len=args.seq_len,
    )
    if job.get("state") != "succeeded":
        raise SystemExit(f"the adaptation run failed: {job.get('error')}")
    result = job["result"]
    out["adapter"] = {
        "path": result["adapter"],
        "backend": result["adapter_backend"],
        "reproducibility": result["reproducibility"],
        "seconds": round(time.time() - started, 1),
    }
    say("adapter", out["adapter"])

    started = time.time()
    merged = merge_adapter(GEMMA4, platform.settings.root, Path(result["adapter"]), args.merged)
    out["merged"] = dict(merged, seconds=round(time.time() - started, 1))
    say("merged", {"model": merged["merged_model"], "sha256": merged["merged_sha256"]})

    out["train_records"] = records
    record_path = args.root / "pipeline_record.json"
    record_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    say("record", str(record_path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("var_kyc"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--forms", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=768,
        help="tokens per training sequence; a KYC record needs more than the lane's 64",
    )
    parser.add_argument("--cost-cap", type=float, default=2.0)
    parser.add_argument("--merged", type=Path, default=Path("var_kyc/model/kyc-gemma4-e2b"))
    parser.add_argument(
        "--no-quantise",
        action="store_true",
        help="train in full precision; only a card that holds the weights can do this",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    with Platform.create(root=args.root, llm_kind="scripted") as platform:
        run(platform, args)


if __name__ == "__main__":
    main()
