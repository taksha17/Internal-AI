#!/usr/bin/env python3
"""Fill KYC forms with the merged model through the repo's own agent, and score it.

The sample form is printed first, blank, with the evidence a reviewer would
hold: that is the input a person sees. The long-horizon run then takes every
held-out form in order, with the model answering each field through the agent's
``read_evidence`` and ``write_field`` tools, and memory carrying a customer's
first fill to their later forms.

Three numbers are reported on the same held-out forms, because one is not
evidence: the merged model, the pinned base without the adapter, and the
keyword-and-pattern agent that needs no model at all. The dev split is a seeded
holdout, not a frozen eval set.

    .venv/bin/python scripts/kyc_trial.py --model var_kyc/model/kyc-gemma4-e2b
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from internal_ai.data.kyc import DOC_TEMPLATES, KYC_FIELDS, KycForm, evidence_lines
from internal_ai.services.kyc_agent import KycAgent, agent_role, open_registry
from internal_ai.training.alpha.serve import complete, load_for_serving


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


def held_out(forms: list[KycForm], fraction: float) -> list[KycForm]:
    customers = sorted({form.customer_id for form in forms})
    cutoff = int(len(customers) * (1 - fraction))
    keep = set(customers[cutoff:])
    return [form for form in forms if form.customer_id in keep]


def evidence_for(
    forms: list[KycForm], dev: list[KycForm], seed: int, decoys: int
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for form in dev:
        pool = [other for other in forms if other.customer_id != form.customer_id]
        distractors = tuple(
            rng.choice(DOC_TEMPLATES).format(**rng.choice(pool).values) for _ in range(decoys)
        )
        out[form.form_id] = evidence_lines(form, rng, distractors)
    return out


def sample_card(form: KycForm, lines: list[str]) -> str:
    blank = "\n".join(f"  {name: <20} ____________________" for name in KYC_FIELDS)
    documents = "\n".join(f"  - {line}" for line in lines)
    return f"KYC FORM {form.form_id}\nDocuments held:\n{documents}\n\nFields to complete:\n{blank}"


def run_agent(forms, evidence, reader=None, workdir: Path | None = None):
    workdir = workdir or Path("var_kyc/run")
    workdir.mkdir(parents=True, exist_ok=True)
    agent = KycAgent(open_registry(workdir, workdir), agent_role(), reader=reader)
    started = time.perf_counter()
    report = agent.run(forms, evidence)
    return report, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("var_kyc/corpus.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("var_kyc/model/kyc-gemma4-e2b"))
    parser.add_argument(
        "--base", type=Path, default=None, help="the pinned base, to score unadapted"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--max-forms", type=int, default=0, help="0 means every held-out form")
    parser.add_argument("--decoys", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("var_kyc/trial_report.json"))
    parser.add_argument("--quantised", action="store_true", help="hold the merged model in nf4")
    parser.add_argument(
        "--detail",
        type=Path,
        default=None,
        help="write per-field accuracy plus every wrong answer here",
    )
    args = parser.parse_args()

    forms = load_forms(args.corpus)
    dev = held_out(forms, args.dev_fraction)
    if args.max_forms:
        dev = dev[: args.max_forms]
    evidence = evidence_for(forms, dev, args.seed, args.decoys)

    print(sample_card(dev[0], evidence[dev[0].form_id]))
    print()
    results: dict[str, object] = {
        "forms": len(dev),
        "fields": len(dev) * len(KYC_FIELDS),
        "dev_split": "seeded holdout, not a frozen eval set",
    }

    model, tokenizer = load_for_serving(args.model, load_in_4bit=args.quantised)

    def reader(prompt: str) -> str:
        return complete(model, tokenizer, prompt, max_new_tokens=24)

    report, elapsed = run_agent(dev, evidence, reader=reader)
    results["merged_model"] = {
        "field_accuracy": report.field_accuracy,
        "consistency": report.consistency,
        "model_calls": report.model_calls,
        "tool_calls": report.tool_calls,
        "memory_hits": report.memory_hits,
        "seconds": round(elapsed, 1),
        "forms_per_s": round(len(dev) / elapsed, 3) if elapsed else 0.0,
    }
    print(json.dumps({"merged_model": results["merged_model"]}, indent=2))

    if args.base is not None:
        base_model, base_tokenizer = load_for_serving(args.base, load_in_4bit=args.quantised)

        def base_reader(prompt: str) -> str:
            return complete(base_model, base_tokenizer, prompt, max_new_tokens=24)

        unadapted, unadapted_elapsed = run_agent(dev, evidence, reader=base_reader)
        results["base_unadapted"] = {
            "field_accuracy": unadapted.field_accuracy,
            "consistency": unadapted.consistency,
            "model_calls": unadapted.model_calls,
            "seconds": round(unadapted_elapsed, 1),
        }
        print(json.dumps({"base_unadapted": results["base_unadapted"]}, indent=2))

    baseline, base_elapsed = run_agent(dev, evidence)
    results["keyword_baseline"] = {
        "field_accuracy": baseline.field_accuracy,
        "consistency": baseline.consistency,
        "seconds": round(base_elapsed, 1),
    }
    print(json.dumps({"keyword_baseline": results["keyword_baseline"]}, indent=2))

    if args.detail is not None:
        triples = zip(report.fields, report.predictions, report.references, strict=True)
        wrong = [
            {"field": name, "predicted": predicted, "reference": reference}
            for name, predicted, reference in triples
            if predicted != reference
        ]
        detail = {"by_field": report.by_field, "wrong": wrong}
        args.detail.parent.mkdir(parents=True, exist_ok=True)
        args.detail.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"detail": str(args.detail)}, indent=2))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
