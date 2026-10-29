"""A dataset gets a compute suggestion. Attaching a machine does not start training."""

from __future__ import annotations

from internal_ai.compute.advise import advise_dataset, attach_backend, make_backend
from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.sandbox import Sandbox


def _dataset(platform, dataset_id: str, tokens: int, kind: str) -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="corpus",
            version=1,
            content_hash="abc",
            record_count=1,
            token_count=tokens,
            created_at=now_iso(),
            stats={"token_count_kind": kind},
        )
    )


def test_a_measured_small_run_is_local(platform) -> None:
    _dataset(platform, "ds_small", 200_000_000, "measured")
    advice = advise_dataset(platform, "ds_small")
    small = advice["sizes"][0]
    assert small["size"] == "small"
    assert small["machine"] == "local"
    assert small["runnable"] is True
    assert advice["backend"] == "local"


def test_an_estimate_and_a_short_corpus_wait(platform) -> None:
    _dataset(platform, "ds_big", 20_000_000_000, "estimate")
    advice = advise_dataset(platform, "ds_big")
    one_b = next(row for row in advice["sizes"] if row["size"] == "1b")
    assert one_b["machine"] == "H100-class"
    assert one_b["planning_gpu_hours"] > 1
    assert one_b["runnable"] is False
    attach_backend(platform, "slurm")
    advised = advise_dataset(platform, "ds_big")
    one_b = next(row for row in advised["sizes"] if row["size"] == "1b")
    assert one_b["meets_floor"] is True
    assert one_b["runnable"] is False
    assert advised["measured"] is False


def test_a_measured_long_run_becomes_runnable_when_slurm_is_attached(platform) -> None:
    _dataset(platform, "ds_long", 20_000_000_000, "measured")
    assert advise_dataset(platform, "ds_long")["sizes"][1]["runnable"] is False
    attach_backend(platform, "slurm")
    advice = advise_dataset(platform, "ds_long")
    assert advice["sizes"][1]["runnable"] is True
    backend = make_backend(platform, Sandbox(platform.settings.sandbox_dir))
    assert backend.name == "slurm"
    assert platform.store.list_models() == []
