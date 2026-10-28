"""The only published bases alpha is allowed to load."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from internal_ai.ids import file_hash


@dataclass(frozen=True, slots=True)
class WeightShard:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class PinnedBase:
    repo_id: str
    revision: str
    license: str
    shards: tuple[WeightShard, ...]

    def directory(self, root: Path) -> Path:
        safe = self.repo_id.replace("/", "--")
        return root / safe / self.revision


@dataclass(frozen=True, slots=True)
class AlphaRecipe:
    """The GTX 1650 recipe. Four-bit quantisation crashed on this card."""

    rank: int = 8
    alpha: int = 16
    max_seq_len: int = 64
    quantization: str | None = None
    stream_from_disk: bool = True


RECIPE = AlphaRecipe()

PRIMARY = PinnedBase(
    repo_id="Qwen/Qwen3-4B",
    revision="1cfa9a7208912126459214e8b04321603b3df60c",
    license="Apache-2.0",
    shards=(
        WeightShard(
            "model-00001-of-00003.safetensors",
            "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
            3_957_900_840,
        ),
        WeightShard(
            "model-00002-of-00003.safetensors",
            "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
            3_987_450_520,
        ),
        WeightShard(
            "model-00003-of-00003.safetensors",
            "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
            99_630_640,
        ),
    ),
)

FALLBACK = PinnedBase(
    repo_id="Qwen/Qwen2.5-3B",
    revision="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    license="qwen-research",
    shards=(
        WeightShard(
            "model-00001-of-00002.safetensors",
            "f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f",
            3_968_658_944,
        ),
        WeightShard(
            "model-00002-of-00002.safetensors",
            "51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb",
            2_203_268_048,
        ),
    ),
)


def verify_snapshot(base: PinnedBase, root: Path) -> Path:
    """Refuse a directory whose weight shards are not the pinned bytes."""
    directory = base.directory(root)
    for shard in base.shards:
        path = directory / shard.name
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned shard: {path}")
        if path.stat().st_size != shard.size:
            raise ValueError(f"pinned shard size mismatch: {path.name}")
        digest = file_hash(path)
        if digest != shard.sha256:
            raise ValueError(f"pinned shard hash mismatch: {path.name}")
    return directory


def available_base(root: Path) -> PinnedBase:
    """Qwen3-4B when its shards match, otherwise the 3B model already trained here."""
    try:
        verify_snapshot(PRIMARY, root)
    except (FileNotFoundError, ValueError):
        verify_snapshot(FALLBACK, root)
        return FALLBACK
    return PRIMARY
