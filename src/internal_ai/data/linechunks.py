"""Chunks that keep their line numbers, so a citation can be checked.

The alpha track is scored by whether an answer's line numbers survive, and the
text reader splits on blank lines without recording where it was reading. A
corpus cut here carries the first line of every chunk with it, so a citation
points at text a reader can open and find.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from internal_ai.data.inventory import disguised_binary

DEFAULT_LINES = 40


@dataclass(frozen=True, slots=True)
class LineChunk:
    path: str
    first_line: int
    last_line: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.first_line}"

    def to_row(self) -> dict[str, str]:
        """A row the jsonl reader keeps whole: its key is its citation."""
        return {
            "id": self.citation,
            "text": self.text,
            "path": self.path,
            "line": str(self.first_line),
        }


def _relative(path: Path, root: Path) -> str:
    """A citation has to outlive this run, so a source outside the root is refused."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is outside the corpus root {root}") from exc


def read_lines(path: Path) -> list[str]:
    """The file as lines, refusing anything that is not text.

    Weights renamed to .txt are the case this exists for: they decode to
    nonsense, and a corpus of nonsense trains a model on noise.
    """
    if disguised_binary(path):
        raise ValueError(f"{path} is a binary payload, not text")
    try:
        return path.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not utf-8 text") from exc


def chunk_lines(
    lines: list[str], path: str, lines_per_chunk: int = DEFAULT_LINES
) -> list[LineChunk]:
    """Cut lines into chunks, ending at a blank line when one is in reach.

    A chunk never runs past ``lines_per_chunk`` lines: an answer citing the
    first line of a chunk has to be able to read the whole thing.
    """
    if lines_per_chunk < 1:
        raise ValueError("a chunk needs at least one line")
    chunks: list[LineChunk] = []
    start = 0
    while start < len(lines):
        while start < len(lines) and not lines[start].strip():
            start += 1
        if start >= len(lines):
            break
        limit = min(start + lines_per_chunk, len(lines))
        end = limit
        if limit < len(lines):
            for candidate in range(limit - 1, start, -1):
                if not lines[candidate].strip():
                    end = candidate
                    break
        body = lines[start:end]
        chunks.append(
            LineChunk(path=path, first_line=start + 1, last_line=end, text="\n".join(body))
        )
        start = end
    return chunks


def verify_lines(lines: list[str], chunks: list[LineChunk]) -> list[str]:
    """What is wrong with these chunks, if anything.

    Each chunk must quote exactly the lines it claims, cover them once, and
    leave no line of writing behind.
    """
    problems: list[str] = []
    seen: dict[int, int] = {}
    for chunk in chunks:
        if chunk.first_line < 1 or chunk.last_line > len(lines):
            problems.append(f"{chunk.citation} points outside the file")
            continue
        expected = "\n".join(lines[chunk.first_line - 1 : chunk.last_line])
        if chunk.text != expected:
            problems.append(f"{chunk.citation} does not quote the lines it names")
        for number in range(chunk.first_line, chunk.last_line + 1):
            seen[number] = seen.get(number, 0) + 1
    for number, count in sorted(seen.items()):
        if count > 1:
            problems.append(f"line {number} appears in {count} chunks")
    missing = [
        number for number, line in enumerate(lines, start=1) if line.strip() and number not in seen
    ]
    if missing:
        problems.append(
            f"{len(missing)} line(s) of writing are in no chunk, first at line {missing[0]}"
        )
    return problems


def write_corpus(
    sources: Iterable[Path],
    out: Path,
    root: Path,
    lines_per_chunk: int = DEFAULT_LINES,
) -> dict[str, Any]:
    """Write one jsonl row per chunk and return the manifest of what went in.

    The manifest names every source by its hash and its chunk count, so a
    later run can say whether it was handed the same documents.
    """
    ordered = sorted({Path(source).resolve() for source in sources})
    if not ordered:
        raise ValueError("a corpus with no sources is not a corpus")
    files: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    for source in ordered:
        relative = _relative(source, root)
        lines = read_lines(source)
        chunks = chunk_lines(lines, relative, lines_per_chunk)
        problems = verify_lines(lines, chunks)
        if problems:
            raise ValueError(
                f"{relative} produced chunks that do not match its lines: {problems[0]}"
            )
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "lines": len(lines),
                "chunks": len(chunks),
            }
        )
        rows.extend(chunk.to_row() for chunk in chunks)
    if not rows:
        raise ValueError("every source was empty; there is nothing to train on")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return {
        "root": str(Path(root).resolve()),
        "lines_per_chunk": lines_per_chunk,
        "chunks": len(rows),
        "files": files,
    }
