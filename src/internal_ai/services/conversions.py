"""Re-running the privacy check and exporting conversations, from the console.

Neither of these is new logic: the audit is ``DataPipeline.run_pii_audit`` and the
exports are ``export_feedback`` / ``export_preferences``, the same functions the CLI
calls. What the page adds is a way to reach them and the refusal text they already
raise -- an export is still a role check, and a failed audit is still a report.
"""

from __future__ import annotations

from typing import Any

from internal_ai.context import Platform
from internal_ai.data.pipeline import DataPipeline
from internal_ai.domain import DatasetVersion
from internal_ai.feedback import export_feedback
from internal_ai.preferences import export_preferences
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require

EXPORTS = ("feedback", "preferences")


def rerun_pii(
    platform: Platform, actor: Actor, dataset_id: str, sample_size: int = 50
) -> dict[str, Any]:
    """A sampled re-check of one dataset. Reports; it does not change the dataset."""
    require(platform.audit, actor, Action.AUDIT_DATASET, dataset_id)
    if platform.store.get_dataset(dataset_id) is None:
        raise ValueError(f"no dataset {dataset_id}")
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root, platform.blobs)
    return pipeline.run_pii_audit(dataset_id, sample_size=sample_size)


def audit_message(result: dict[str, Any]) -> str:
    if result.get("passed"):
        return (
            f"{result.get('sampled', 0)} records sampled and no critical finding; "
            "the dataset is unchanged by this check"
        )
    findings = result.get("findings") or {}
    kinds = ", ".join(f"{kind} x{count}" for kind, count in sorted(findings.items()))
    return (
        f"{result.get('critical_findings', 0)} critical findings in "
        f"{result.get('sampled', 0)} sampled records ({kinds or 'none named'}); "
        "do not release this dataset until it is rebuilt"
    )


def export(platform: Platform, actor: Actor, kind: str) -> DatasetVersion:
    """One of the two conversation exports, as a new unreleased dataset."""
    if kind == "feedback":
        return export_feedback(platform, actor)
    if kind == "preferences":
        return export_preferences(platform, actor)
    raise ValueError(f"unknown export {kind!r}; the exports are {', '.join(EXPORTS)}")


def export_message(dataset: DatasetVersion) -> str:
    return (
        f"{dataset.name} v{dataset.version} created with {dataset.record_count} records; "
        "it is unreleased, so it still needs its gates before anything trains on it"
    )
