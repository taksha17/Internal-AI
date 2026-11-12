"""Experiment tracking stays in SQLite unless an MLflow server is configured.

The store is the tracker: runs are always recorded there. When
``INTERNAL_AI_MLFLOW_URI`` points at a self-hosted server, the run's
aggregates (cost, duration, status) are mirrored there too. Record content —
output text, artifacts, metadata — never leaves the boundary, and a missing
package or an unreachable server never breaks the run: the marker on the
returned run says what happened.
"""

from __future__ import annotations

import dataclasses
import sys
from contextlib import contextmanager

from internal_ai import tracking
from internal_ai.domain import RunSummary


def a_run() -> RunSummary:
    return RunSummary(
        task_id="task_1",
        agent="trainer",
        status="ok",
        output="secret model output that stays home",
        artifacts=["artifact_1"],
        cost_usd=1.25,
        duration_s=30.0,
        metadata={"anything": "stays home"},
    )


class FakeMlflowRun:
    def __init__(self, client: FakeMlflowClient) -> None:
        self.client = client

    def __enter__(self) -> FakeMlflowRun:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def log_metrics(self, metrics: dict) -> None:
        self.client.metrics.update(metrics)

    def set_tags(self, tags: dict) -> None:
        self.client.tags.update(tags)


class FakeMlflowClient:
    def __init__(self) -> None:
        self.experiments: list = []
        self.metrics: dict = {}
        self.tags: dict = {}
        self.uris: list = []

    def set_tracking_uri(self, uri: str) -> None:
        self.uris.append(uri)

    def set_experiment(self, name: str) -> None:
        self.experiments.append(name)

    def start_run(self, run_name: str) -> FakeMlflowRun:
        self.client_run_name = run_name
        return FakeMlflowRun(self)


@contextmanager
def fake_mlflow(monkeypatch):  # type: ignore[no-untyped-def]
    client = FakeMlflowClient()
    monkeypatch.setattr(tracking, "_import_mlflow", lambda: client)
    yield client


def test_without_a_uri_everything_stays_in_sqlite(platform, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("INTERNAL_AI_MLFLOW_URI", raising=False)

    def explode() -> object:
        raise AssertionError("mlflow must not even be imported")

    monkeypatch.setattr(tracking, "_import_mlflow", explode)
    saved = platform.store.record_run(a_run())
    assert saved.metadata["tracker"] == "sqlite"
    assert platform.store.list_runs("task_1"), "the run must still be stored"


def test_aggregates_are_mirrored_and_records_stay_home(platform, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setenv("INTERNAL_AI_MLFLOW_URI", "http://mlflow:5000")
    with fake_mlflow(monkeypatch) as client:
        saved = platform.store.record_run(a_run())
    assert saved.metadata["tracker"] == "mlflow:ok"
    assert client.uris == ["http://mlflow:5000"]
    assert client.experiments == ["internal-ai"]
    assert client.metrics == {"cost_usd": 1.25, "duration_s": 30.0}
    assert client.tags["agent"] == "trainer"
    assert client.tags["status"] == "ok"
    assert client.tags["task_id"] == "task_1"
    everything_sent = str(client.metrics) + str(client.tags)
    assert "secret model output" not in everything_sent
    assert "artifact_1" not in everything_sent
    assert "stays home" not in everything_sent


def test_a_missing_mlflow_package_skips_the_mirror(platform, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_AI_MLFLOW_URI", "http://mlflow:5000")
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    def missing() -> object:
        raise tracking.TrackerUnavailable("mlflow package is not installed")

    monkeypatch.setattr(tracking, "_import_mlflow", missing)
    saved = platform.store.record_run(a_run())
    assert saved.metadata["tracker"].startswith("mlflow:skipped:")
    assert platform.store.list_runs("task_1"), "the run must still be stored"


def test_an_unreachable_server_never_breaks_the_run(platform, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_AI_OFFLINE", "0")
    monkeypatch.setenv("INTERNAL_AI_MLFLOW_URI", "http://mlflow:5000")

    class Broken(FakeMlflowClient):
        def start_run(self, run_name: str) -> FakeMlflowRun:
            raise ConnectionError("connection refused")

    monkeypatch.setattr(tracking, "_import_mlflow", lambda: Broken())
    saved = platform.store.record_run(a_run())
    assert saved.metadata["tracker"].startswith("mlflow:failed:")
    assert platform.store.list_runs("task_1"), "the run must still be stored"


def test_an_offline_install_never_calls_out(platform, monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_AI_MLFLOW_URI", "http://mlflow:5000")
    monkeypatch.delenv("INTERNAL_AI_OFFLINE", raising=False)

    def explode() -> object:
        raise AssertionError("offline installs make no outbound calls")

    monkeypatch.setattr(tracking, "_import_mlflow", explode)
    saved = platform.store.record_run(a_run())
    assert saved.metadata["tracker"].startswith("mlflow:skipped:")
    assert platform.store.list_runs("task_1"), "the run must still be stored"


def test_the_returned_run_is_a_copy(platform) -> None:  # type: ignore[no-untyped-def]
    run = a_run()
    before = dict(run.metadata)
    platform.store.record_run(run)
    assert dict(run.metadata) == before, "record_run must not mutate its argument"
    assert dataclasses.is_dataclass(run)
