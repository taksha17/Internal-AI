"""Scan the working tree for credentials, using the patterns the corpus audit uses.

A dataset cannot be released with a secret in it, but nothing looked at the code
itself. Run before tests in CI: a fixable leak should fail the build, not the
release audit weeks later.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from internal_ai.data.inventory import SECRET_PATTERNS

SKIP_DIRS = {
    ".git",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "var",
    "var_console",
    "var_demo",
    "var_e2e",
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".pt", ".bin", ".pyc", ".zip"}
MAX_BYTES = 2_000_000

# The documented placeholders the test fixtures and sample corpus use. A real
# credential is not on this list, so a leak still fails the build.
ALLOWED_PLACEHOLDERS = {
    "AKIAIOSFODNN7EXAMPLE",
    "sk-ABCDEFGHIJKLMNOPQRST",
    'PASSWORD = "correct-horse-battery',
    'SECRET = "api_key: ',
    "api_key: 'sk-live-not-a-real-token",
    'api_key="test-key',
    "password = 's3cret-value-here",
    'secret = "password: ',
}


@dataclass(slots=True)
class Finding:
    path: str
    line: int
    kind: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} ({self.excerpt})"


def scan_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in SECRET_PATTERNS.items():
            match = pattern.search(line)
            if match is None:
                continue
            excerpt = match.group(0)
            if excerpt in ALLOWED_PLACEHOLDERS:
                continue
            masked = excerpt[:4] + "…" if len(excerpt) > 8 else "…"
            findings.append(Finding(path=path, line=number, kind=kind, excerpt=masked))
    return findings


def candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        files.append(path)
    return files


def scan(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in candidate_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings.extend(scan_text(str(path.relative_to(root)), text))
    return findings


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else Path(__file__).resolve().parents[1]
    findings = scan(root)
    for finding in findings:
        print(finding)
    if findings:
        print(f"{len(findings)} secret(s) found in {root}", file=sys.stderr)
        return 1
    print(f"no secrets found in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
