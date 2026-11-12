"""Experiment tracking behind the store interface.

SQLite is the tracker: every run is recorded there, and nothing here changes
that. When ``INTERNAL_AI_MLFLOW_URI`` points at a self-hosted MLflow server,
the run's aggregates are mirrored there too: numeric metrics (cost,
duration) and plain tags (task, agent, status). Output text, artifacts and
metadata never leave the boundary, and an offline install never calls out.

The mirror never breaks the run. A missing package, an unreachable server
or an offline install only changes the marker on the returned run, so the
log says what happened instead of hiding it.
"""

from __future__ import annotations

import os
from typing import Any

from internal_ai.config import offline_mode
from internal_ai.domain import RunSummary

TRACKING_URI_ENV = "INTERNAL_AI_MLFLOW_URI"
EXPERIMENT_ENV = "INTERNAL_AI_MLFLOW_EXPERIMENT"
DEFAULT_EXPERIMENT = "internal-ai"


class TrackerUnavailable(Exception):
    """The MLflow package is not installed where the platform runs."""


def _import_mlflow() -> Any:
    import mlflow

    return _FluentClient(mlflow)


class _RunHandle:
    """One MLflow run behind the same surface the tests fake."""

    def __init__(self, module: Any, context: Any) -> None:
        self._module = module
        self._context = context

    def __enter__(self) -> _RunHandle:
        self._context.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._context.__exit__(*exc)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._module.log_metrics(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._module.set_tags(tags)


class _FluentClient:
    """The real mlflow module behind a narrow, fakeable surface."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def set_tracking_uri(self, uri: str) -> None:
        self._module.set_tracking_uri(uri)

    def set_experiment(self, name: str) -> None:
        self._module.set_experiment(name)

    def start_run(self, run_name: str) -> _RunHandle:
        return _RunHandle(self._module, self._module.start_run(run_name=run_name))


def _marker(reason: str, detail: str = "") -> str:
    text = f"mlflow:{reason}"
    if detail:
        text += f":{detail[:120]}"
    return text


def mirror_run(run: RunSummary) -> str:
    """Mirror a run's aggregates to MLflow, returning what happened.

    ``sqlite`` means nothing was configured. Anything starting with
    ``mlflow:`` names the outcome; only ``mlflow:ok`` sent anything, and
    what it sent was metrics and tags, never record content.
    """
    uri = os.environ.get(TRACKING_URI_ENV, "").strip()
    if not uri:
        return "sqlite"
    if offline_mode():
        return _marker("skipped", "install is offline")
    try:
        client = _import_mlflow()
    except TrackerUnavailable as exc:
        return _marker("skipped", str(exc))
    except ImportError as exc:
        return _marker("skipped", f"mlflow package is not installed ({exc})")
    try:
        experiment = os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT)
        client.set_tracking_uri(uri)
        client.set_experiment(experiment)
        with client.start_run(run_name=run.task_id) as handle:
            handle.log_metrics({"cost_usd": run.cost_usd, "duration_s": run.duration_s})
            handle.set_tags({"task_id": run.task_id, "agent": run.agent, "status": run.status})
    except Exception as exc:  # the mirror must never break a run
        return _marker("failed", str(exc))
    return "mlflow:ok"
