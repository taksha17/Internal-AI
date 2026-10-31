"""A citation is only worth something if the lines it names are really there."""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.data.linechunks import (
    LineChunk,
    chunk_lines,
    read_lines,
    verify_lines,
    write_corpus,
)

DOCUMENT = "\n".join(
    [
        "Request for Comments: 9110",
        "",
        "The target resource is identified by the request target.",
        "",
        "A server MUST generate a 4xx status code for a malformed request.",
        "A client MUST NOT reuse a connection after a fatal error.",
    ]
)


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_chunk_quotes_exactly_the_lines_it_names(tmp_path: Path) -> None:
    path = _write(tmp_path, "rfc9110.txt", DOCUMENT)
    lines = read_lines(path)

    chunks = chunk_lines(lines, "rfc9110.txt", lines_per_chunk=3)

    assert verify_lines(lines, chunks) == []
    for chunk in chunks:
        assert chunk.text == "\n".join(lines[chunk.first_line - 1 : chunk.last_line])
        assert chunk.citation == f"rfc9110.txt:{chunk.first_line}"


def test_a_blank_line_ends_a_chunk_before_the_limit(tmp_path: Path) -> None:
    path = _write(tmp_path, "rfc9110.txt", DOCUMENT)
    lines = read_lines(path)

    chunks = chunk_lines(lines, "rfc9110.txt", lines_per_chunk=4)

    assert [chunk.first_line for chunk in chunks] == [1, 5]
    assert chunks[0].last_line == 3


def test_every_line_of_writing_reaches_a_chunk(tmp_path: Path) -> None:
    path = _write(tmp_path, "rfc9110.txt", DOCUMENT)
    lines = read_lines(path)

    chunks = chunk_lines(lines, "rfc9110.txt", lines_per_chunk=2)

    covered = {
        number for chunk in chunks for number in range(chunk.first_line, chunk.last_line + 1)
    }
    assert {number for number, line in enumerate(lines, start=1) if line.strip()} <= covered


def test_a_chunk_that_moved_a_line_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "rfc9110.txt", DOCUMENT)
    lines = read_lines(path)
    chunks = chunk_lines(lines, "rfc9110.txt", lines_per_chunk=4)
    tampered = [LineChunk(c.path, c.first_line + 1, c.last_line + 1, c.text) for c in chunks]

    problems = verify_lines(lines, tampered)

    assert any("does not quote the lines it names" in problem for problem in problems)


def test_a_chunk_size_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one line"):
        chunk_lines(["anything"], "x.txt", lines_per_chunk=0)


def test_weights_renamed_as_text_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "innocent.txt"
    path.write_bytes(b"\x00\x01\x02binary payload")

    with pytest.raises(ValueError, match="binary payload"):
        read_lines(path)


def test_a_file_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))

    with pytest.raises(ValueError, match="not utf-8"):
        read_lines(path)


def test_a_corpus_with_no_sources_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no sources"):
        write_corpus([], tmp_path / "corpus.jsonl", tmp_path)


def test_a_source_outside_the_root_is_refused(tmp_path: Path) -> None:
    outside = _write(tmp_path.parent, "elsewhere.txt", DOCUMENT)

    with pytest.raises(ValueError, match="outside the corpus root"):
        write_corpus([outside], tmp_path / "corpus.jsonl", tmp_path)


def test_a_corpus_of_empty_files_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "empty.txt", "\n\n\n")

    with pytest.raises(ValueError, match="nothing to train on"):
        write_corpus([tmp_path / "empty.txt"], tmp_path / "corpus.jsonl", tmp_path)


def test_two_builds_of_the_same_documents_are_identical(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", DOCUMENT)
    _write(tmp_path, "b.txt", DOCUMENT.replace("9110", "9111"))
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"

    manifest = write_corpus(sorted(tmp_path.glob("*.txt")), first, tmp_path)
    again = write_corpus(sorted(tmp_path.glob("*.txt")), second, tmp_path)

    assert first.read_bytes() == second.read_bytes()
    assert manifest == again
    assert manifest["chunks"] == 2
    assert [entry["path"] for entry in manifest["files"]] == ["a.txt", "b.txt"]
