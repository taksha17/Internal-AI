#!/usr/bin/env python3
"""Run the KYC form-filling agent over a held-out dev split and score it.

The split is a seeded holdout for development, not a frozen eval set: nothing
here grants `eval_set_freeze`, and the numbers are a progress signal, not a
promotion claim. The report carries the agent's accuracy, the keyword
first-hit baseline (what a grep already does), and the cross-form consistency
the memory exists for, plus the throughput the optimisation phase will have
to beat.

    .venv/bin/python scripts/kyc_fill.py --corpus var_kyc/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from internal_ai.data.kyc import DOC_TEMPLATES, KYC_FIELDS, KycForm, evidence_lines
from internal_ai.evals.metrics import exact_match
from internal_ai.services.kyc_agent import FIELD_PATTERNS, KycAgent, agent_role, open_registry


def load_forms(path: Path) -> list[KycForm]:
    forms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            forms.append(
                KycForm(
                    form_id=row["form_id"],
                    customer_id=row["customer_id"],
                    values=row["values"],
                )
            )
    return forms


def keyword_baseline(forms: list[KycForm], evidence: dict[str, list[str]]) -> float:
    """First line that shares a token with the field name, then the pattern."""
    predictions: list[str] = []
    references: list[str] = []
    for form in forms:
        for name in KYC_FIELDS:
            cue = set(name.split("_"))
            pattern = re.compile(FIELD_PATTERNS[name])
            value = ""
            for line in evidence[form.form_id]:
                if cue & set(line.lower().replace("-", " ").replace("/", " ").split()):
                    match = pattern.search(line)
                    if match:
                        value = match.group(1)
                        break
            predictions.append(value)
            references.append(form.values[name])
    return exact_match(predictions, references)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("var_kyc/corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--distractors", type=int, default=10)
    parser.add_argument("--workdir", type=Path, default=Path("var_kyc/run"))
    parser.add_argument("--report", type=Path, default=Path("var_kyc/report.json"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    forms = load_forms(args.corpus)
    customers = sorted({form.customer_id for form in forms})
    cutoff = int(len(customers) * (1 - args.dev_fraction))
    dev_customers = set(customers[cutoff:])
    dev = [form for form in forms if form.customer_id in dev_customers]

    evidence: dict[str, list[str]] = {}
    for form in dev:
        pool = [other for other in forms if other.customer_id != form.customer_id]
        decoys = [
            rng.choice(DOC_TEMPLATES).format(**rng.choice(pool).values)
            for _ in range(args.distractors)
        ]
        evidence[form.form_id] = evidence_lines(form, rng, tuple(decoys))

    args.workdir.mkdir(parents=True, exist_ok=True)
    agent = KycAgent(open_registry(args.workdir, args.workdir), agent_role())
    started = time.perf_counter()
    report = agent.run(dev, evidence)
    elapsed = time.perf_counter() - started
    baseline = keyword_baseline(dev, evidence)

    result = {
        "forms": report.forms,
        "fields_scored": report.fields_scored,
        "agent_accuracy": report.field_accuracy,
        "keyword_baseline": baseline,
        "consistency": report.consistency,
        "memory_hits": report.memory_hits,
        "tool_calls": report.tool_calls,
        "elapsed_s": round(elapsed, 2),
        "forms_per_s": round(report.forms / elapsed, 2) if elapsed else 0.0,
        "dev_split_note": "seeded holdout, not a frozen eval set",
    }
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
