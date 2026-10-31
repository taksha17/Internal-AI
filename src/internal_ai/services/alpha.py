"""Adapting the pinned base is a governed run, not a bare function call.

The trainer does the arithmetic and knows nothing about who asked, what it
cost or what it produced. This module is the path that does: the person's
capability is checked, the dataset must be released, the base must be one this
install is licensed to load, and the run happens as a job so a killed process
still leaves a row saying so. The adapter lands next to a
`reproducibility.json` that names the base revision, the dataset hash and the
seed, because an adapter nobody can reproduce is a claim, not an artifact.
"""

from __future__ import annotations

import json
import platform as host
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import AuditOutcome, GateStatus, GateType, RunSummary
from internal_ai.ids import file_hash, new_id, now_iso, tree_hash
from internal_ai.services.actor import Actor
from internal_ai.services.build import TRAINING_AGENT, TRAINING_COST_GATE_USD, gate_for
from internal_ai.services.jobs import JobRunner
from internal_ai.services.permissions import Action, deny, require
from internal_ai.training.alpha.adapt import BACKEND as FULL_DEPTH_BACKEND
from internal_ai.training.alpha.adapt import adapt as adapt_full_depth
from internal_ai.training.alpha.pin import RECIPE, PinnedBase, available_base
from internal_ai.training.alpha.train import BACKEND as STREAMED_BACKEND
from internal_ai.training.alpha.train import train_prefix
from internal_ai.training.ownership import require_released_dataset
from internal_ai.training.provenance import require_open_weights

REPRODUCIBILITY_FILE = "reproducibility.json"
ADAPTER_BACKENDS = frozenset({FULL_DEPTH_BACKEND, STREAMED_BACKEND})


def _commit(root: Path) -> str:
    """The code that produced the adapter, or an honest 'unknown'."""
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
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:  # pragma: no cover - the extra is optional
        return "absent"


def reproducibility_beside(adapter_path: Path) -> dict[str, Any]:
    """The record written when this adapter was trained, if there is one."""
    path = adapter_path.parent / REPRODUCIBILITY_FILE
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _training_text(platform: Platform, dataset_id: str) -> list[str]:
    """The text the adapter learns from: the training split, else every record.

    A released dataset that was never split is still a released dataset, and
    the adapter should learn from all of it rather than silently from nothing.
    """
    rows = platform.store.dataset_records_by_split(dataset_id, "train")
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    texts = [str(row["payload"]) for row in rows if str(row["payload"]).strip()]
    if not texts:
        raise ValueError("the released dataset holds no text to adapt on")
    return texts


def _require_cost_gate(platform: Platform, actor: Actor, subject: str, cost_cap_usd: float) -> None:
    if cost_cap_usd <= TRAINING_COST_GATE_USD:
        return
    gate = gate_for(platform, GateType.TRAINING_RUN_COST, subject)
    if gate is None or gate.status != GateStatus.APPROVED:
        raise deny(
            platform.audit,
            actor,
            "train_alpha",
            subject,
            f"an adaptation run capped at {cost_cap_usd:.2f} USD needs an approved "
            f"{GateType.TRAINING_RUN_COST} gate for {subject}",
        )


def train_alpha(
    platform: Platform,
    actor: Actor,
    runner: JobRunner,
    dataset_id: str,
    layers: int = 2,
    sequences: int = 4,
    seed: int = 0,
    cost_cap_usd: float = 2.0,
    epochs: int = 1,
    backend: str = FULL_DEPTH_BACKEND,
    base: PinnedBase | None = None,
) -> dict[str, Any]:
    """Adapt the pinned base on a released dataset, as a job with a record."""
    require(platform.audit, actor, Action.TRAIN_MODEL, dataset_id)
    mode = require_open_weights(platform, "adapting a base model")
    dataset = require_released_dataset(platform.store.get_dataset(dataset_id))
    if backend not in ADAPTER_BACKENDS:
        raise ValueError(
            f"unknown adapter backend {backend}; this package runs {sorted(ADAPTER_BACKENDS)}"
        )
    if backend == STREAMED_BACKEND and not 1 <= layers <= 4:
        raise ValueError("adapt between one and four layers; the recipe is streamed by layer")
    if epochs < 1:
        raise ValueError("an adaptation run needs at least one pass over the text")
    if sequences < 1:
        raise ValueError("a run needs at least one sequence")
    if cost_cap_usd <= 0:
        raise ValueError("an adaptation run needs a positive cost cap")
    if seed < 0:
        raise ValueError("the seed is a non-negative integer")
    chosen = base or available_base(platform.settings.root)
    subject = f"{dataset_id}:{chosen.repo_id}"
    _require_cost_gate(platform, actor, subject, cost_cap_usd)
    texts = _training_text(platform, dataset_id)
    stem = chosen.repo_id.split("/")[-1].lower()
    suffix = ".pt" if backend == STREAMED_BACKEND else "-adapter"
    adapter = platform.settings.artifacts_dir / "alpha" / f"{stem}-{new_id('ad')}{suffix}"

    def work() -> dict[str, Any]:
        run_id = new_id("run")
        started = now_iso()
        if backend == FULL_DEPTH_BACKEND:
            result = adapt_full_depth(
                chosen,
                platform.settings.root,
                texts,
                adapter,
                rank=RECIPE.rank,
                alpha=RECIPE.alpha,
                max_seq_len=RECIPE.max_seq_len,
                epochs=epochs,
                seed=seed,
            )
        else:
            result = train_prefix(
                chosen,
                platform.settings.root,
                texts,
                adapter,
                layers=layers,
                sequences=sequences,
                seed=seed,
            )
        finished = now_iso()
        record = {
            "run_id": run_id,
            "track": "alpha",
            "adapter_backend": backend,
            "provenance_mode": mode,
            "commit": _commit(platform.settings.root),
            "base": {
                "repo_id": chosen.repo_id,
                "revision": chosen.revision,
                "license": chosen.license,
                "license_class": str(chosen.license_class),
            },
            "dataset": {"id": dataset.id, "content_hash": dataset.content_hash},
            "recipe": {
                "backend": backend,
                "epochs": epochs if backend == FULL_DEPTH_BACKEND else None,
                "rank": RECIPE.rank,
                "alpha": RECIPE.alpha,
                "max_seq_len": RECIPE.max_seq_len,
                "quantization": RECIPE.quantization,
                "layers": result.get("layers", layers),
                "sequences": result.get("sequences", len(texts)),
            },
            "seed": seed,
            "adapter_sha256": result.get("adapter_sha256")
            or (tree_hash(adapter) if adapter.is_dir() else file_hash(adapter)),
            "environment": {
                "python": host.python_version(),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "peft": _package_version("peft"),
            },
        }
        record_path = adapter.parent / REPRODUCIBILITY_FILE
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        platform.store.record_run(
            RunSummary(
                task_id=f"alpha:{dataset.id}",
                agent=TRAINING_AGENT,
                status="completed",
                output=(
                    f"adapted {result.get('layers')} layers of {chosen.repo_id} "
                    f"on {dataset.name} v{dataset.version}"
                ),
                artifacts=[str(adapter), str(record_path)],
                started_at=started,
                finished_at=finished,
                metadata={
                    "run_id": run_id,
                    "track": "alpha",
                    "base_revision": chosen.revision,
                    "dataset_hash": dataset.content_hash,
                    "seed": seed,
                    "adapter_backend": backend,
                },
            )
        )
        platform.audit.record(
            actor=TRAINING_AGENT,
            action="alpha.train_finished",
            subject=run_id,
            outcome=AuditOutcome.SUCCESS,
            details={
                "adapter_sha256": record["adapter_sha256"],
                "base_repo_id": chosen.repo_id,
                "base_revision": chosen.revision,
                "license_class": record["base"]["license_class"],
                "provenance_mode": mode,
                "dataset_id": dataset.id,
                "seed": seed,
            },
        )
        return {
            **result,
            "run_id": run_id,
            "adapter_backend": backend,
            "adapter": str(adapter),
            "reproducibility": str(record_path),
            "license_class": record["base"]["license_class"],
        }

    return runner.submit(
        actor,
        "train_alpha",
        f"Adapt {chosen.repo_id} on {dataset.name} v{dataset.version}",
        {
            "dataset_id": dataset_id,
            "base_repo_id": chosen.repo_id,
            "base_revision": chosen.revision,
            "layers": layers,
            "sequences": sequences,
            "seed": seed,
            "epochs": epochs,
            "backend": backend,
        },
        work,
        agent=TRAINING_AGENT,
        cost_cap_usd=cost_cap_usd,
    )
