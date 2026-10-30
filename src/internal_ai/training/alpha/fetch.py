"""Download a pinned base onto a caller-supplied disk and check every shard."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from internal_ai.config import offline_mode
from internal_ai.ids import file_hash
from internal_ai.training.alpha.pin import (
    DEFAULT_ALLOWED,
    FALLBACK,
    PRIMARY,
    BaseRefused,
    PinnedBase,
    verify_snapshot,
)


def _url(base: PinnedBase, name: str) -> str:
    return f"https://huggingface.co/{base.repo_id}/resolve/{base.revision}/{name}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "internal-ai-alpha", "Range": f"bytes={existing}-"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("ab") as handle:
        status = getattr(response, "status", 200)
        if status == 200 and existing:
            handle.seek(0)
            handle.truncate()
        written = existing if status != 200 else 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if written % (256 << 20) < (1 << 20):
                print(f"{dest.name} {written}", flush=True)
    partial.replace(dest)


def fetch_base(base: PinnedBase, root: Path) -> Path:
    """Place the pinned shards at ``root/<repo>/<revision>`` and verify them."""
    if offline_mode():
        raise RuntimeError(
            "offline mode refuses a base download; verify the pinned snapshot on disk"
        )
    directory = base.directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    for shard in base.shards:
        dest = directory / shard.name
        if dest.is_file() and dest.stat().st_size == shard.size and file_hash(dest) == shard.sha256:
            continue
        if dest.exists():
            dest.unlink()
        _download(_url(base, shard.name), dest)
    return verify_snapshot(base, root)


def fetch_primary_or_fallback(root: Path, *, allow_research_licence: bool = False) -> PinnedBase:
    """Qwen3-4B first. The research-licensed 3B pin is fetched only when named.

    A failed download used to fall through to the 3B model silently. An install
    that later trains and ships an adapter would then owe its provenance to a
    licence nobody approved, so the fallback now needs the caller to ask.
    """
    try:
        fetch_base(PRIMARY, root)
    except (OSError, ValueError) as exc:
        if FALLBACK.license_class not in DEFAULT_ALLOWED and not allow_research_licence:
            raise BaseRefused(
                f"{PRIMARY.repo_id} could not be fetched and {FALLBACK.repo_id} is "
                f"{FALLBACK.license} ({FALLBACK.license_class}); it is fetched only when "
                "allow_research_licence is set"
            ) from exc
        fetch_base(FALLBACK, root)
        return FALLBACK
    return PRIMARY
