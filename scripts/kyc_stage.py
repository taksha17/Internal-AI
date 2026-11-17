#!/usr/bin/env python3
"""Stage the released KYC corpus for the rented GPU. See services/kyc_stage.

.venv/bin/python scripts/kyc_stage.py --root var_kyc --dataset ds_xxx --out var_kyc/kaggle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internal_ai.context import Platform
from internal_ai.services.kyc_stage import KERNEL_SLUG, TITLE, stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("var_kyc"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, default=Path("var_kyc/kaggle"))
    parser.add_argument("--title", default=TITLE, help="Kaggle dataset title to stage into")
    parser.add_argument("--slug", default=KERNEL_SLUG, help="Kaggle kernel slug to push")
    parser.add_argument("--owner", default=None, help="Kaggle account; defaults to the token file")
    parser.add_argument("--rows", type=int, default=None, help="cap the training-split rows")
    parser.add_argument("--batch-size", type=int, default=1, help="sequences per step on the lane")
    parser.add_argument(
        "--gradient-accumulation", type=int, default=1, help="micro-batches per optimizer step"
    )
    args = parser.parse_args()
    platform = Platform.create(root=args.root, llm_kind="scripted")
    try:
        print(
            json.dumps(
                stage(
                    platform,
                    args.dataset,
                    args.out,
                    args.title,
                    args.slug,
                    args.owner,
                    args.rows,
                    args.batch_size,
                    args.gradient_accumulation,
                ),
                indent=2,
            )
        )
        return 0
    finally:
        platform.close()


if __name__ == "__main__":
    raise SystemExit(main())
