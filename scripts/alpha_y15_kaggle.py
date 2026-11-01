#!/usr/bin/env python3
"""Stage the Y1.5 adaptation run as a private Kaggle dataset.

The GPU box gets the released training text, the frozen questions, the base pin
and a copy of the repository's own training step, and nothing else. It does not
get the reference answers: a run that could read what it will be scored against
would produce a number, not evidence.

The staged copy of the trainer matters. An adapter only counts if it came from
the step this repository audits, so the six modules that step needs travel with
the data, hashed, instead of trusting whatever is installed on a rented box.

    .venv/bin/python scripts/alpha_y15_kaggle.py --root var_alpha/install \\
        --out var_alpha/kaggle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.training.alpha.pin import PRIMARY

DATASET = "ds_9350fb760231"
EVAL_SET = "evs_99206c7f7cb8"
TITLE = "internal-ai-alpha-y15"
# The dataset already owns this name, and Kaggle refuses a kernel on the slug a
# dataset holds, so the kernel is pushed under its own.
KERNEL_SLUG = "alpha-y15-t4-run"
PACKAGE_FILES = (
    "__init__.py",
    "ids.py",
    "evals/__init__.py",
    "evals/metrics.py",
    "training/__init__.py",
    "training/alpha/__init__.py",
    "training/alpha/pin.py",
    "training/alpha/adapt.py",
    "training/alpha/pairs.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def commit(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def username() -> str:
    """The Kaggle account, from the token file. The key is never read out."""
    payload = json.loads(Path(".secrets/kaggle.json").read_text(encoding="utf-8"))
    return str(payload["username"])


def copy_package(destination: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "internal_ai"
    for name in PACKAGE_FILES:
        target = destination / "internal_ai" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)


def stage(platform: Platform, dataset_id: str, eval_set_id: str, out: Path) -> dict[str, Any]:
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None or not dataset.released:
        raise SystemExit(f"refused: dataset {dataset_id} is not released")
    rows = platform.store.dataset_records_by_split(dataset_id, "train")
    if not rows:
        raise SystemExit(f"refused: dataset {dataset_id} has no training split")
    from internal_ai.training.alpha.pairs import CANDIDATE_LIMIT, passages, retrieval_pairs

    # Raw prose taught the register and not the job, and keywords-to-sentence
    # taught the shape of an answer and not the choice of one: the second run
    # cited runs of consecutive lines out of whichever file it had seen most. The
    # task is selection, so a pair offers the candidates a search returns. The
    # index it searches is every released passage, which is what a deployed
    # search covers; the pairs themselves still come from the train split only.
    corpus = passages(platform.store.dataset_records(dataset_id))
    if not corpus:
        raise SystemExit(f"refused: dataset {dataset_id} produced no searchable passages")
    texts = retrieval_pairs(rows, corpus)
    if not texts:
        raise SystemExit(f"refused: dataset {dataset_id} produced no selection pairs")
    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise SystemExit(f"refused: unknown eval set {eval_set_id}")
    if not getattr(eval_set, "frozen", False):
        raise SystemExit(f"refused: eval set {eval_set_id} is not frozen")
    queries = [
        {"id": str(item.get("id") or f"q{index:04d}"), "query": str(item["query"])}
        for index, item in enumerate(eval_set.items)
    ]
    carried = sorted({key for item in queries for key in item})
    if carried != ["id", "query"]:
        raise SystemExit(f"refused: the question file would carry {carried} to the GPU box")

    dataset_dir = out / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_file = dataset_dir / "train.jsonl"
    train_file.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in texts), encoding="utf-8"
    )
    queries_file = dataset_dir / "queries.json"
    queries_file.write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
    # The index ships, not the answers: the box needs the passages to search, and
    # the question file still carries nothing but an id and a question.
    corpus_file = dataset_dir / "corpus.jsonl"
    corpus_file.write_text(
        "".join(
            json.dumps({"key": key, "text": text}, sort_keys=True) + "\n" for key, text in corpus
        ),
        encoding="utf-8",
    )
    copy_package(dataset_dir)
    # A dataset version uploads files, not directories: the first attempt carried
    # train.jsonl and lost internal_ai entirely, so the kernel had no trainer to
    # import. One archive is a file, and a file arrives.
    bundle = dataset_dir / "internal_ai.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((dataset_dir / "internal_ai").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(dataset_dir).as_posix())
    shutil.copyfile(Path(__file__).parent / "alpha_y15_kernel.py", dataset_dir / "kernel.py")

    recipe = {
        "commit": commit(Path(__file__).resolve().parents[1]),
        "dataset_id": dataset.id,
        "dataset_content_hash": dataset.content_hash,
        "train_records": len(texts),
        "pair_format": "candidates_to_copied_passage",
        "candidates": CANDIDATE_LIMIT,
        "corpus_records": len(corpus),
        "corpus_sha256": sha256(corpus_file),
        "train_sha256": sha256(train_file),
        "queries_sha256": sha256(queries_file),
        "questions": len(queries),
        "eval_set_id": eval_set.id,
        "eval_set_hash": getattr(eval_set, "content_hash", ""),
        # A pair is the instruction, the question, five candidates and the answer.
        # 256 cut the candidates off the end of every example and left the model
        # no list to choose from.
        "max_seq_len": 512,
        "base": {
            "repo_id": PRIMARY.repo_id,
            "revision": PRIMARY.revision,
            "license": PRIMARY.license,
            "shards": [
                {"name": shard.name, "sha256": shard.sha256, "size": shard.size}
                for shard in PRIMARY.shards
            ],
        },
    }
    recipe_file = dataset_dir / "recipe.json"
    recipe_file.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    owner = username()
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": TITLE,
                "id": f"{owner}/{TITLE}",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    kernel_dir = out / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).parent / "alpha_y15_kernel.py", kernel_dir / "kernel.py")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{owner}/{KERNEL_SLUG}",
                "title": KERNEL_SLUG,
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{owner}/{TITLE}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "train_records": len(texts),
        "train_sha256": recipe["train_sha256"],
        "corpus_records": len(corpus),
        "corpus_sha256": recipe["corpus_sha256"],
        "questions": len(queries),
        "queries_sha256": recipe["queries_sha256"],
        "commit": recipe["commit"],
        "base": recipe["base"]["repo_id"],
        "eval_set_hash": recipe["eval_set_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("var_alpha/install"))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--eval-set", default=EVAL_SET)
    parser.add_argument("--out", type=Path, default=Path("var_alpha/kaggle"))
    args = parser.parse_args()
    platform = Platform.create(root=args.root, llm_kind="scripted")
    try:
        print(json.dumps(stage(platform, args.dataset, args.eval_set, args.out), indent=2))
        return 0
    finally:
        platform.close()


if __name__ == "__main__":
    sys.exit(main())
