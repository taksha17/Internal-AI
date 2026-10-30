"""The only published bases alpha is allowed to load."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from internal_ai.ids import file_hash


class LicenseClass(StrEnum):
    """How far a licence lets a company use the weights it covers."""

    PERMISSIVE = "permissive"
    RESTRICTED = "restricted"
    RESEARCH_ONLY = "research_only"


PERMISSIVE_LICENSES = frozenset({"Apache-2.0", "MIT"})
RESTRICTED_LICENSES = frozenset(
    {"Llama-3.1-Community", "Llama-3-Community", "Gemma", "CC-BY-NC-4.0"}
)
DEFAULT_ALLOWED: frozenset[LicenseClass] = frozenset({LicenseClass.PERMISSIVE})


class BaseRefused(PermissionError):
    """A base this install may not load. The message names the class that blocked it."""


def license_class_of(spdx: str) -> LicenseClass:
    """An unreviewed licence gets the strictest class, never the permissive one."""
    if spdx in PERMISSIVE_LICENSES:
        return LicenseClass.PERMISSIVE
    if spdx in RESTRICTED_LICENSES:
        return LicenseClass.RESTRICTED
    return LicenseClass.RESEARCH_ONLY


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

    @property
    def license_class(self) -> LicenseClass:
        return license_class_of(self.license)


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


PINNED: tuple[PinnedBase, ...] = (PRIMARY, FALLBACK)


def available_base(
    root: Path,
    allowed: frozenset[LicenseClass] = DEFAULT_ALLOWED,
    bases: tuple[PinnedBase, ...] = PINNED,
) -> PinnedBase:
    """The first pinned base on disk whose licence is allowed.

    A base that fails verification is skipped, not replaced by a worse one: the
    3B pin is research-licensed, so reaching it because the 4B pin is missing
    would hand a company a licence it cannot use while telling it nothing.
    """
    missing: list[str] = []
    refused: list[str] = []
    for base in bases:
        try:
            verify_snapshot(base, root)
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"{base.repo_id}: {exc}")
            continue
        if base.license_class not in allowed:
            refused.append(f"{base.repo_id} is {base.license} ({base.license_class})")
            continue
        return base
    if refused:
        classes = ", ".join(sorted(allowed))
        raise BaseRefused(
            "no allowed pinned base is on disk; "
            + "; ".join(refused)
            + f"; this install allows {classes}"
        )
    raise BaseRefused("no pinned base is on disk: " + "; ".join(missing))
