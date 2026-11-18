"""A dataset gets a compute suggestion. Attaching a machine does not start training."""

from __future__ import annotations

import pytest

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


def test_a_world_size_below_one_is_refused() -> None:
    from internal_ai.training.cost import planning_vram_gib

    with pytest.raises(ValueError, match="world size"):
        planning_vram_gib(10_000_000, world_size=0)


def test_a_measured_small_run_is_local(platform) -> None:
    _dataset(platform, "ds_small", 200_000_000, "measured")
    advice = advise_dataset(platform, "ds_small")
    small = advice["sizes"][0]
    assert small["size"] == "small"
    assert small["machine"] == "local"
    assert small["runnable"] is True
    assert advice["backend"] == "local"
    assert advice["world_size"] == 1
    assert advice["queue_wait"] is None
    from internal_ai.training.trainer import TrainConfig

    step = TrainConfig()
    assert advice["tokens_per_step"] == step.batch_size * step.seq_len
    assert small["planning_vram_gib"] > 0
    assert advice["sizes"][2]["planning_vram_gib"] > small["planning_vram_gib"]


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
