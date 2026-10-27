"""Document text, read out of the containers a company actually keeps.

A PDF or an Office file is not text, but it holds text, and the corpus needs
that text rather than the bytes. Everything that reads a source calls this, so
the inventory, the console scan, the tree index and ingest all see the same
extraction, and a document cannot arrive by one door and be missed by another.

Each container is checked against its own suffix before it is parsed. A payload
renamed to look like a document fails that check, so the guard against weights
hiding in text survives the arrival of formats that are genuinely binary.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx", ".xlsx", ".pptx"})

# The part whose presence proves the container is what its suffix claims.
REQUIRED_PART = {
    ".docx": "word/document.xml",
    ".xlsx": "xl/workbook.xml",
    ".pptx": "ppt/presentation.xml",
}

# Parts that carry the writing for each suffix.
MEMBER_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    ".docx": (re.compile(r"^word/document\.xml$"),),
    ".xlsx": (
        re.compile(r"^xl/sharedStrings\.xml$"),
        re.compile(r"^xl/worksheets/sheet\d+\.xml$"),
    ),
    ".pptx": (re.compile(r"^ppt/slides/slide\d+\.xml$"),),
}

# Documents are far larger than text files, and their text is far smaller, so
# the caps are their own rather than the scan's per-file text cap.
MAX_DOCUMENT_BYTES = 25_000_000
MAX_EXTRACTED_CHARS = 1_000_000

_TEXT_TAG = "t"

# Every one of these ends a run of writing that belongs on its own: a Word
# paragraph, a slide paragraph, a spreadsheet row or cell, a line break. Excel
# keeps its strings in one flat table with no rows around them, so without "si"
# here the cells would run together into a single word.
_BREAK_TAGS = frozenset({"p", "tr", "row", "si", "br"})
_BREAK = "\n\n"
_SCANNED_PDF = (
    "the PDF has no text layer, so it looks like a scan; reading it needs OCR, "
    "which the platform does not do"
)
ENCRYPTED_PDF = "the PDF is encrypted"
PDF_UNSUPPORTED = "PDF reading is not installed in this environment"


@dataclass(slots=True)
class Extracted:
    """What a document yielded, and why it yielded nothing when it did not."""

    text: str = ""
    reason: str = ""
    truncated: bool = False

    @property
    def usable(self) -> bool:
        return bool(self.text.strip())


def is_document(path_or_suffix: Path | str) -> bool:
    name = str(path_or_suffix)
    suffix = name if name.startswith(".") else Path(name).suffix
    return suffix.lower() in DOCUMENT_SUFFIXES


def _capped(text: str, max_chars: int) -> Extracted:
    if len(text) <= max_chars:
        return Extracted(text=text.strip())
    return Extracted(text=text[:max_chars].strip(), truncated=True)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _office_text(blob: bytes) -> str:
    """Paragraphs and cells from one Office XML part.

    Element names are compared without their namespace: Word, Excel and
    PowerPoint each carry their own prefix over the same few local names, and
    matching on the local name covers all three.
    """
    try:
        root = ElementTree.fromstring(blob)
    except ElementTree.ParseError:
        return ""
    parts: list[str] = []
    for element in root.iter():
        name = _local(element.tag)
        if name in _BREAK_TAGS:
            parts.append(_BREAK)
        elif name == _TEXT_TAG and element.text:
            parts.append(element.text)
    joined = "".join(parts)
    return re.sub(r"\n{3,}", _BREAK, joined).strip()


def _read_office(path: Path, suffix: str, max_chars: int) -> Extracted:
    required = REQUIRED_PART[suffix]
    if not zipfile.is_zipfile(path):
        return Extracted(reason=f"the file is named {suffix} but is not an Office file")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if required not in set(names):
                return Extracted(reason=f"the file is named {suffix} but holds no {required}")
            matched = sorted(
                name for name in names for pattern in MEMBER_PATTERNS[suffix] if pattern.match(name)
            )
            chunks = [_office_text(archive.read(name)) for name in matched]
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        return Extracted(reason=f"the {suffix} file could not be read: {type(exc).__name__}")
    text = _BREAK.join(chunk for chunk in chunks if chunk.strip())
    if not text.strip():
        return Extracted(reason=f"the {suffix} file holds no readable text")
    return _capped(text, max_chars)


def _read_pdf(path: Path, max_chars: int) -> Extracted:
    try:
        with path.open("rb") as handle:
            if not handle.read(5).startswith(b"%PDF-"):
                return Extracted(reason="the file is named .pdf but does not start like one")
    except OSError as exc:
        return Extracted(reason=f"the PDF could not be opened: {type(exc).__name__}")
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - the dependency is declared
        return Extracted(reason=PDF_UNSUPPORTED)
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return Extracted(reason=ENCRYPTED_PDF)
        pages: list[str] = []
        length = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
            length += len(text)
            if length >= max_chars:
                break
    except Exception as exc:  # a malformed PDF must not stop the scan
        return Extracted(reason=f"the PDF could not be read: {type(exc).__name__}")
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text.strip():
        return Extracted(reason=_SCANNED_PDF)
    return _capped(text, max_chars)


def document_text(
    path: Path,
    max_chars: int = MAX_EXTRACTED_CHARS,
    max_bytes: int = MAX_DOCUMENT_BYTES,
) -> Extracted:
    """The text a document holds, or the reason it holds none.

    A document past the size cap is skipped by name rather than read: a scan is
    not a reason to load an arbitrarily large file into memory.
    """
    suffix = path.suffix.lower()
    if suffix not in DOCUMENT_SUFFIXES:
        return Extracted(reason=f"{suffix or 'a file without an extension'} is not a document")
    try:
        size = path.stat().st_size
    except OSError as exc:
        return Extracted(reason=f"the file could not be read: {type(exc).__name__}")
    if size > max_bytes:
        cap = max_bytes // 1_000_000
        return Extracted(reason=f"the file is {size / 1_000_000:.1f} MB, past the {cap} MB cap")
    if suffix == ".pdf":
        return _read_pdf(path, max_chars)
    return _read_office(path, suffix, max_chars)
