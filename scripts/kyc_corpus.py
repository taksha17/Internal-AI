#!/usr/bin/env python3
"""Generate the synthetic KYC corpus the form-filling agent runs against.

Every form is template-built from a seed: no model output anywhere, no real
customer, every value marked fake. The manifest records the seed, the count
and the content hash, so a rerun can prove it produced the same corpus.

    .venv/bin/python scripts/kyc_corpus.py --count 1200 --out var_kyc/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internal_ai.data.kyc import DEFAULT_FORMS, build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_FORMS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("var_kyc/corpus.jsonl"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("var_kyc/train.jsonl"))
    args = parser.parse_args()

    corpus = build_corpus(count=args.count, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for form in corpus.forms:
            handle.write(json.dumps(form.as_dict(), sort_keys=True) + "\n")
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n")
    import random

    from internal_ai.data.kyc import evidence_lines
    from internal_ai.training.alpha.pairs import kyc_fill_text

    rng = random.Random(args.seed)
    with args.train_jsonl.open("w", encoding="utf-8") as handle:
        for form in corpus.forms:
            lines = evidence_lines(form, rng)
            handle.write(
                json.dumps(
                    {
                        "text": kyc_fill_text(form.values, lines),
                        "form_id": form.form_id,
                        "customer_id": form.customer_id,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"forms={len(corpus.forms)} hash={corpus.manifest['content_hash']}")
    print(f"train_records={args.train_jsonl}")


if __name__ == "__main__":
    main()
