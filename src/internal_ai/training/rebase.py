"""What a rebase repeats, and the four numbers that decide whether it counts.

The last promise in the RFS is that a company keeps what it gained when the base
under it changes. That is a procedure, not a claim: adapt again on the new base,
evaluate every candidate on the same frozen set, and let the brief's own bars
decide whether a promotion may even be proposed. A recipe is the record of what
was repeated, so the next rebase is the same act and not a fresh invention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from internal_ai.brief import Metric, ProjectBrief, describe_verdict, load, metric_key
from internal_ai.context import Platform
from internal_ai.domain import AuditOutcome
from internal_ai.ids import new_id, now_iso
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, deny, require
from internal_ai.training.alpha.pin import DEFAULT_ALLOWED, PinnedBase
from internal_ai.training.ownership import require_released_dataset

KEY = "rebase:{}"
COLUMN_ORDER = ("old", "new", "base_only", "rented")
COLUMN_LABELS = {
    "old": "old base + adapter",
    "new": "new base + adapter",
    "base_only": "new base, un-adapted",
    "rented": "rented API",
}


@dataclass(frozen=True, slots=True)
class Recipe:
    """Everything a rebase has to reproduce, frozen at the moment it was agreed."""

    id: str
    name: str
    dataset_id: str
    dataset_hash: str
    base_repo_id: str
    base_revision: str
    license_class: str
    eval_set_ids: tuple[str, ...]
    metrics: tuple[Metric, ...]
    sequences: int
    seed: int
    created_at: str
    created_by: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "base_repo_id": self.base_repo_id,
            "base_revision": self.base_revision,
            "license_class": self.license_class,
            "eval_set_ids": list(self.eval_set_ids),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "sequences": self.sequences,
            "seed": self.seed,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Recipe:
        return cls(
            id=str(payload["id"]),
            name=str(payload.get("name", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            dataset_hash=str(payload.get("dataset_hash", "")),
            base_repo_id=str(payload.get("base_repo_id", "")),
            base_revision=str(payload.get("base_revision", "")),
            license_class=str(payload.get("license_class", "")),
            eval_set_ids=tuple(str(item) for item in payload.get("eval_set_ids") or ()),
            metrics=tuple(Metric.from_dict(item) for item in payload.get("metrics") or ()),
            sequences=int(payload.get("sequences") or 0),
            seed=int(payload.get("seed") or 0),
            created_at=str(payload.get("created_at", "")),
            created_by=str(payload.get("created_by", "")),
        )


def build_recipe(
    platform: Platform,
    actor: Actor,
    *,
    name: str,
    dataset_id: str,
    base: PinnedBase,
    eval_set_ids: list[str],
    sequences: int = 4,
    seed: int = 0,
) -> Recipe:
    """Write down what a rebase will repeat, refusals first.

    The bars are copied from the brief as it stands, not read later: a report
    has to say how it was judged, or editing the brief rewrites history. The
    dataset and the eval sets are checked now for the same reason - a recipe
    that names a set which can still move is not a comparison.
    """
    require(platform.audit, actor, Action.WRITE_RECIPE, name)
    dataset = require_released_dataset(platform.store.get_dataset(dataset_id))
    if base.license_class not in DEFAULT_ALLOWED:
        raise deny(
            platform.audit,
            actor,
            "build_recipe",
            name,
            f"{base.repo_id} is {base.license} ({base.license_class}); a recipe adapts a "
            "base this install may load",
        )
    if not eval_set_ids:
        raise ValueError("a recipe names at least one eval set to compare on")
    for eval_set_id in eval_set_ids:
        eval_set = platform.store.get_eval_set(eval_set_id)
        if eval_set is None:
            raise KeyError(f"unknown eval set: {eval_set_id}")
        if not eval_set.frozen:
            raise PermissionError(
                f"eval set {eval_set_id} is not frozen; a rebase compares on a set that cannot move"
            )
    metrics = tuple(load(platform.store).metrics)
    if not any(metric.name and metric.threshold for metric in metrics):
        raise ValueError(
            "the project brief records no metric, so a rebase would have nothing to be judged by"
        )
    recipe = Recipe(
        id=new_id("rcp"),
        name=name,
        dataset_id=dataset.id,
        dataset_hash=dataset.content_hash,
        base_repo_id=base.repo_id,
        base_revision=base.revision,
        license_class=str(base.license_class),
        eval_set_ids=tuple(eval_set_ids),
        metrics=metrics,
        sequences=sequences,
        seed=seed,
        created_at=now_iso(),
        created_by=actor.id,
    )
    platform.store.kv_set(KEY.format(recipe.id), json.dumps(recipe.to_dict(), sort_keys=True))
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="rebase.recipe_saved",
        subject=recipe.id,
        details={
            "name": name,
            "dataset_hash": recipe.dataset_hash,
            "base_revision": recipe.base_revision,
            "eval_set_ids": list(recipe.eval_set_ids),
        },
    )
    return recipe


def load_recipe(platform: Platform, recipe_id: str) -> Recipe:
    stored = platform.store.kv_get(KEY.format(recipe_id), "")
    if not stored:
        raise KeyError(f"unknown recipe: {recipe_id}")
    return Recipe.from_dict(json.loads(stored))


@dataclass(slots=True)
class RebaseReport:
    recipe_id: str
    columns: dict[str, dict[str, float]]
    rows: list[dict[str, Any]]
    breaches: list[str]

    @property
    def proposes_promotion(self) -> bool:
        return not self.breaches

    def text(self) -> str:
        """One line per metric, four columns wide, in the order a person reads them."""
        lines = [f"{'metric':<24}{'old':>8}{'new':>8}{'base':>8}{'rented':>8}"]
        for row in self.rows:
            cells = "".join(
                f"{row[key]:>8.3f}" if row[key] is not None else f"{'—':>8}" for key in COLUMN_ORDER
            )
            lines.append(f"{row['name']:<24}{cells}")
        return "\n".join(lines)


def _lookup(values: dict[str, float], name: str) -> float | None:
    wanted = metric_key(name)
    for key, value in values.items():
        if metric_key(key) == wanted:
            return float(value)
    return None


def _worse(direction: str, candidate: float, reference: float) -> bool:
    """A tie is not a gain, the same rule the base comparison already applies."""
    if direction == "at_most":
        return candidate >= reference
    return candidate <= reference


def compare(
    recipe: Recipe,
    *,
    old: dict[str, float],
    new: dict[str, float],
    base_only: dict[str, float],
    rented: dict[str, float],
) -> RebaseReport:
    """Four columns of the same metrics, judged by the recipe's own bars.

    Every column has to measure every bar. A gap is a breach rather than a blank
    cell, because "the rented API does not report recall@10" is a fact the
    decision needs, not a formatting problem.
    """
    columns = {
        "old": dict(old),
        "new": dict(new),
        "base_only": dict(base_only),
        "rented": dict(rented),
    }
    for key in COLUMN_ORDER:
        if not columns[key]:
            raise ValueError(f"the {COLUMN_LABELS[key]} column has no measurements")
    verdicts = ProjectBrief(metrics=list(recipe.metrics)).metric_verdicts(columns["new"])
    by_name = {metric_key(verdict["name"]): verdict for verdict in verdicts}
    rows: list[dict[str, Any]] = []
    breaches: list[str] = []
    for metric in recipe.metrics:
        verdict = by_name.get(metric_key(metric.name), {})
        row: dict[str, Any] = {
            "name": metric.name,
            "threshold": metric.threshold,
            "direction": metric.direction,
            "met": verdict.get("met"),
        }
        for key in COLUMN_ORDER:
            row[key] = _lookup(columns[key], metric.name)
        rows.append(row)
        missing = [COLUMN_LABELS[key] for key in COLUMN_ORDER if row[key] is None]
        if missing:
            breaches.append(f"{metric.name}: no measurement from the " + ", ".join(missing))
            continue
        if verdict.get("met") is None:
            breaches.append(
                f"{metric.name}: {verdict.get('note') or 'the new adapter could not be judged'}"
            )
            continue
        if not verdict["met"]:
            breaches.append(describe_verdict(verdict) + f" ({COLUMN_LABELS['new']})")
            continue
        if _worse(metric.direction, float(row["new"]), float(row["base_only"])):
            breaches.append(
                f"{metric.name}: {COLUMN_LABELS['new']} measures {row['new']:.3f} against "
                f"{row['base_only']:.3f} un-adapted, so the adapter is not a gain"
            )
        if _worse(metric.direction, float(row["new"]), float(row["old"])):
            breaches.append(
                f"{metric.name}: {COLUMN_LABELS['new']} measures {row['new']:.3f} against "
                f"{row['old']:.3f} from the {COLUMN_LABELS['old']}, so the rebase lost ground"
            )
    return RebaseReport(recipe_id=recipe.id, columns=columns, rows=rows, breaches=breaches)


def propose(platform: Platform, actor: Actor, report: RebaseReport) -> RebaseReport:
    """Allow a promotion to be requested, or refuse it and say why.

    This is not the gate and does not open one: it is the check that stops a
    request being made at all when the brief is short. The gate, the approvers
    and the serving pointer stay exactly where they were.
    """
    require(platform.audit, actor, Action.WRITE_RECIPE, report.recipe_id)
    if report.breaches:
        platform.audit.record(
            actor=actor.id,
            actor_kind=actor.kind,
            action="rebase.promotion_refused",
            subject=report.recipe_id,
            outcome=AuditOutcome.DENIED,
            details={"breaches": report.breaches[:6]},
        )
        raise PermissionError(
            "this rebase does not earn a promotion: " + "; ".join(report.breaches[:3])
        )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="rebase.promotion_proposed",
        subject=report.recipe_id,
        outcome=AuditOutcome.SUCCESS,
        details={"metrics": [row["name"] for row in report.rows]},
    )
    return report
