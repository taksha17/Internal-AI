"""Build a corpus from public RFC text, with a line number on every chunk.

The alpha track is scored on whether citations survive, so the corpus has to be
documents a reader can open at the cited line. Public IETF RFCs are
redistributable and line-addressable, which is why this example can use them:
nothing here is a customer record.

    INTERNAL_AI_OFFLINE=0 .venv/bin/python scripts/build_rfc_corpus.py \
        --rfc 2119 8174 9110 \
        --root var/alpha-corpus/rfc --out var/alpha-corpus/corpus.jsonl

Exit code is 0 when the corpus was written, 1 when a document could not be
fetched, and 2 when the install is offline and the fetch was refused.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from internal_ai.config import offline_mode
from internal_ai.data.linechunks import write_corpus

EDITOR = "https://www.rfc-editor.org/rfc/rfc{number}.txt"
HEADER_BYTES = 4096


def _fetch(number: int, target: Path) -> None:
    """Download one RFC and refuse anything that does not name itself.

    A captive portal or an error page answers 200, and a corpus of error pages
    trains a model to cite a line that is not in any document.
    """
    url = EDITOR.format(number=number)
    request = urllib.request.Request(url, headers={"User-Agent": "internal-ai-corpus/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    if not body:
        raise RuntimeError(f"{url} returned nothing")
    header = body.decode("utf-8", errors="replace")[:HEADER_BYTES]
    if f"RFC {number}" not in header and f"Request for Comments: {number}" not in header:
        raise RuntimeError(f"{url} does not look like RFC {number}; refusing to keep it")
    target.write_bytes(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rfc", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lines-per-chunk", type=int, default=40)
    parser.add_argument("--refetch", action="store_true")
    args = parser.parse_args(argv)

    numbers = sorted(set(args.rfc))
    args.root.mkdir(parents=True, exist_ok=True)
    missing = [
        number
        for number in numbers
        if args.refetch or not (args.root / f"rfc{number}.txt").is_file()
    ]
    if missing and offline_mode():
        print(
            "refusing to fetch: this install is offline. Set INTERNAL_AI_OFFLINE=0 to "
            "download a document set that is already public.",
            file=sys.stderr,
        )
        return 2
    for number in missing:
        target = args.root / f"rfc{number}.txt"
        try:
            _fetch(number, target)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            print(f"could not fetch rfc{number}: {exc}", file=sys.stderr)
            return 1
        print(f"fetched rfc{number}.txt ({target.stat().st_size // 1024} KiB)")

    paths = [args.root / f"rfc{number}.txt" for number in numbers]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        print(f"missing: {', '.join(absent)}", file=sys.stderr)
        return 1
    manifest = write_corpus(paths, args.out, args.root, lines_per_chunk=args.lines_per_chunk)
    manifest_path = args.out.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {manifest['chunks']} chunks from {len(manifest['files'])} documents to {args.out}"
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
