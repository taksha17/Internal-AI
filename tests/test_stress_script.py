"""The stress harness clears the release gates a person would meet.

`scripts/stress.py` is the script the README's scale numbers come from. Change
0.1 fixed the script for the language-review release gate: a domain expert now
reviews the seeded sample before the dataset releases, exactly as the console
does. This test guards that fix. The full run at scale is validated on the
machine that regenerates the README numbers, not here; at a toy size the only
gate that can still stop it is the separate tokens-per-parameter floor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "stress.py"


def test_stress_script_clears_the_language_release_gate(tmp_path: Path) -> None:
    root = tmp_path / "var_stress"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--records",
            "240",
            "--steps",
            "2",
            "--skip-serving",
            "--skip-console",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert "a domain expert has not reviewed the language sample" not in completed.stderr

    reached_training = completed.returncode == 0 or "tokens per parameter" in completed.stderr
    assert reached_training, completed.stderr
