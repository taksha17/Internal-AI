"""Stage the KYC adaptation run as a private Kaggle dataset.

The GPU box gets the released training text, the dev questions by id with
their documents, the base pin and a copy of the repository's own training
step, and nothing else. The answer block never travels: the box returns what
the adapted model says, and the scoring happens at home against the released
dataset it never saw.

The export lane is guarded like every data path: the dataset has to be
released, every staged text has to read as template-generated synthetic (the
fake markers the generator puts on every value), the source behind it has to
be an internal synthetic one, and the install has to have switched the
offline default off explicitly (``INTERNAL_AI_OFFLINE=0``). The stage leaves
an audit event naming the dataset, the counts and the commit it staged.

The command line lives in ``scripts/kyc_stage.py``; this module holds the
stage so tests can drive it without a shell.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from internal_ai.config import offline_mode
from internal_ai.context import Platform
from internal_ai.training.alpha.pin import GEMMA4

TITLE = "internal-ai-kyc-fill"
KERNEL_SLUG = "kyc-fill-t4-run"
# The generator marks every value, so a staged text without one of these is
# not ours: it is a leak, a corruption, or a hand edit, and none of those
# travel to a rented box.
FAKE_MARKERS = ("SAMPLE-", "TEST", "example.test", "+1-555-01", "Test Street")
PACKAGE_FILES = (
    "__init__.py",
    "ids.py",
    "data/__init__.py",
    "data/kyc.py",
    "evals/__init__.py",
    "evals/metrics.py",
    "training/__init__.py",
    "training/alpha/__init__.py",
    "training/alpha/pin.py",
    "training/alpha/adapt.py",
    "training/alpha/pairs.py",
    "training/alpha/serve.py",
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
    source = Path(__file__).resolve().parents[2] / "internal_ai"
    for name in PACKAGE_FILES:
        target = destination / "internal_ai" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload", {})
    if isinstance(payload, str):
        return json.loads(payload)
    return dict(payload)


def _sources(dataset) -> list[dict[str, Any]]:
    return [dict(item) for item in (dataset.lineage.get("sources") or [])]


def stage(
    platform: Platform,
    dataset_id: str,
    out: Path,
    title: str = TITLE,
    slug: str = KERNEL_SLUG,
    owner: str | None = None,
    row_limit: int | None = None,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
) -> dict[str, Any]:
    from internal_ai.training.alpha.pairs import kyc_documents_from_text

    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None or not dataset.released:
        raise SystemExit(f"refused: dataset {dataset_id} is not released")
    sources = _sources(dataset)
    if not sources or any(item.get("legal_status") != "internal" for item in sources):
        raise SystemExit(
            f"refused: dataset {dataset_id} is not from an internal source; "
            "the export lane runs on synthetic corpora only"
        )
    if not all("synthetic" in str(item.get("license_name") or "") for item in sources):
        raise SystemExit(f"refused: dataset {dataset_id} is not template-generated synthetic data")
    train_rows = platform.store.dataset_records_by_split(dataset_id, "train")
    if not train_rows:
        raise SystemExit(f"refused: dataset {dataset_id} has no training split")
    if row_limit is not None:
        train_rows = sorted(train_rows, key=lambda row: row["record_key"])[:row_limit]
    texts: list[str] = []
    for row in train_rows:
        text = str(_payload(row).get("text") or "")
        if not text:
            raise SystemExit(f"refused: row {row['record_key']} carries no training text")
        texts.append(text)
    test_rows = platform.store.dataset_records_by_split(dataset_id, "test")
    if not test_rows:
        raise SystemExit(f"refused: dataset {dataset_id} has no dev split to ask from")
    questions: list[dict[str, Any]] = []
    for row in test_rows:
        text = str(_payload(row).get("text") or "")
        questions.append({"id": str(row["record_key"]), "documents": kyc_documents_from_text(text)})
    staged = texts + ["\n".join(item["documents"]) for item in questions]
    bare = [text for text in staged if not any(marker in text for marker in FAKE_MARKERS)]
    if bare:
        raise SystemExit(f"refused: {len(bare)} of {len(staged)} staged texts carry no fake marker")
    if offline_mode():
        raise SystemExit(
            "refused: staging writes a package to a third-party box and the install is "
            "offline (the default, INTERNAL_AI_OFFLINE=1); a lab may set "
            "INTERNAL_AI_OFFLINE=0 for this run"
        )

    dataset_dir = out / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_file = dataset_dir / "train.jsonl"
    train_file.write_text(
        "".join(json.dumps({"text": text}, sort_keys=True) + "\n" for text in texts),
        encoding="utf-8",
    )
    queries_file = dataset_dir / "queries.json"
    queries_file.write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")
    copy_package(dataset_dir)
    bundle = dataset_dir / "internal_ai.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((dataset_dir / "internal_ai").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(dataset_dir).as_posix())
    shutil.copyfile(
        Path(__file__).resolve().parents[3] / "scripts" / "kyc_kernel.py",
        dataset_dir / "kernel.py",
    )

    recipe = {
        "commit": commit(Path(__file__).resolve().parents[2]),
        "dataset_id": dataset.id,
        "dataset_content_hash": dataset.content_hash,
        "train_records": len(texts),
        "pair_format": "kyc_documents_answers_first",
        "questions": len(questions),
        "train_sha256": sha256(train_file),
        "queries_sha256": sha256(queries_file),
        # Measured on the 3070 lane rather than assumed: a rendered record is
        # 853 tokens at the median and 887 at the maximum, so the run sequences
        # whole records instead of cutting them. The weights ride in nf4 for
        # the same reason they do at home: the embedding tables alone are bf16
        # beside a quantised body, and no 8 GB card holds both.
        "max_seq_len": 1024,
        "quantisation": "nf4-4bit",
        "batch_size": max(1, int(batch_size)),
        "gradient_accumulation": max(1, int(gradient_accumulation)),
        "base": {
            "repo_id": GEMMA4.repo_id,
            "revision": GEMMA4.revision,
            "license": GEMMA4.license,
            "shards": [
                {"name": shard.name, "sha256": shard.sha256, "size": shard.size}
                for shard in GEMMA4.shards
            ],
        },
    }
    recipe_file = dataset_dir / "recipe.json"
    recipe_file.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    account = owner or username()
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": title,
                "id": f"{account}/{title}",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    kernel_dir = out / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[3] / "scripts" / "kyc_kernel.py",
        kernel_dir / "kernel.py",
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{account}/{slug}",
                "title": slug,
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{account}/{title}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    platform.audit.record(
        actor="data_engineer",
        actor_kind="human",
        action="remote_training_export",
        subject=dataset_id,
        details={
            "dataset_id": dataset.id,
            "train_records": len(texts),
            "questions": len(questions),
            "commit": recipe["commit"],
            "offline_override": offline_mode() is False,
        },
    )
    return {
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "train_records": len(texts),
        "train_sha256": recipe["train_sha256"],
        "questions": len(questions),
        "queries_sha256": recipe["queries_sha256"],
        "batch_size": recipe["batch_size"],
        "gradient_accumulation": recipe["gradient_accumulation"],
        "commit": recipe["commit"],
        "base": recipe["base"]["repo_id"],
    }
