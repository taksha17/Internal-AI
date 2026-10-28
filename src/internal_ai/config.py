"""Runtime configuration and filesystem layout for Internal-AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_HOME = "INTERNAL_AI_HOME"
ENV_OFFLINE = "INTERNAL_AI_OFFLINE"
DEFAULT_HOME = "var"
_OFFLINE_OFF = frozenset({"0", "false", "off", "no"})


def offline_mode() -> bool:
    """True unless a lab sets INTERNAL_AI_OFFLINE=0.

    Enterprise installs make no outbound call. The default is offline so a
    missing variable cannot open the network.
    """
    flag = os.environ.get(ENV_OFFLINE, "1").strip().lower()
    return flag not in _OFFLINE_OFF


@dataclass(frozen=True, slots=True)
class Settings:
    root: Path
    home: Path
    db_path: Path
    artifacts_dir: Path
    workspace_dir: Path
    datasets_dir: Path
    models_dir: Path
    checkpoints_dir: Path
    runs_dir: Path
    sandbox_dir: Path
    logs_dir: Path
    eval_dir: Path
    lineage_dir: Path

    @classmethod
    def from_env(cls, root: Path | None = None) -> Settings:
        base = Path(root) if root is not None else Path.cwd()
        home = Path(os.environ.get(ENV_HOME, base / DEFAULT_HOME)).resolve()
        return cls(
            root=base.resolve(),
            home=home,
            db_path=home / "internal_ai.db",
            artifacts_dir=home / "artifacts",
            workspace_dir=home / "workspace",
            datasets_dir=home / "datasets",
            models_dir=home / "models",
            checkpoints_dir=home / "checkpoints",
            runs_dir=home / "runs",
            sandbox_dir=home / "sandbox",
            logs_dir=home / "logs",
            eval_dir=home / "evals",
            lineage_dir=home / "lineage",
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.home,
            self.artifacts_dir,
            self.workspace_dir,
            self.datasets_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.runs_dir,
            self.sandbox_dir,
            self.logs_dir,
            self.eval_dir,
            self.lineage_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)
