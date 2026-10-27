"""Documents yield their text, and a payload that lies about itself does not.

Company data arrives as PDFs and Office files, so the scan has to read them.
The guard that keeps renamed weights out of the corpus has to survive that,
which is why every container is checked against its own suffix first.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from internal_ai.data.extract import document_text
from internal_ai.data.records import Source, ingest_source
from internal_ai.data.tree import build_tree_index
from internal_ai.services import build

INVOICE = "Invoice reconciliation failed for account acct-1"
SHIPMENT = "Shipment tracking stalled for account acct-2"
SECRET = "api_key: 'sk-live-not-a-real-token'"

DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    f"<w:body><w:p><w:r><w:t>{INVOICE}</w:t></w:r></w:p>"
    f"<w:p><w:r><w:t>{SHIPMENT}</w:t></w:r></w:p></w:body></w:document>"
)

XLSX_PARTS = {
    "xl/workbook.xml": "<workbook/>",
    "xl/sharedStrings.xml": "<sst><si><t>billing</t></si><si><t>refunds</t></si></sst>",
}

PPTX_PARTS = {
    "ppt/presentation.xml": "<presentation/>",
    "ppt/slides/slide1.xml": (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
        ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Quarterly rollout plan</a:t></p:sld>"
    ),
}


def zip_file(tmp_path: Path, name: str, parts: dict[str, str]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, xml in parts.items():
            archive.writestr(member, xml)
    return path


def docx(tmp_path: Path, name: str = "report.docx", body: str = INVOICE) -> Path:
    xml = DOCX_XML.replace(INVOICE, body)
    return zip_file(tmp_path, name, {"word/document.xml": xml})


def pdf_bytes(text: str | None) -> bytes:
    """A one-page PDF with a correct cross-reference table.

    ``text=None`` builds the page a scanner would produce: a valid page with no
    text layer at all.
    """
    content = ""
    if text is not None:
        escaped = text.replace("(", "\\(").replace(")", "\\)")
        content = f"BT /F1 12 Tf 40 700 Td ({escaped}) Tj ET\n"
    bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n{content}endstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    start = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    return bytes(out)


def pdf(tmp_path: Path, name: str = "policy.pdf", text: str | None = INVOICE) -> Path:
    path = tmp_path / name
    path.write_bytes(pdf_bytes(text))
    return path


def test_a_pdf_yields_its_text(tmp_path: Path) -> None:
    result = document_text(pdf(tmp_path))

    assert result.usable
    assert INVOICE in result.text


def test_a_pdf_with_no_text_layer_is_named_rather_than_guessed(tmp_path: Path) -> None:
    result = document_text(pdf(tmp_path, text=None))

    assert not result.usable
    assert "OCR" in result.reason
    assert "scan" in result.reason


def test_a_docx_yields_its_paragraphs(tmp_path: Path) -> None:
    result = document_text(docx(tmp_path))

    assert result.usable
    assert result.text.split("\n\n") == [INVOICE, SHIPMENT]


def test_an_xlsx_yields_its_cells(tmp_path: Path) -> None:
    result = document_text(zip_file(tmp_path, "book.xlsx", XLSX_PARTS))

    assert result.usable
    # Cells keep their own boundary: run together they would read as one word.
    assert result.text.split("\n\n") == ["billing", "refunds"]


def test_a_pptx_yields_its_slides(tmp_path: Path) -> None:
    result = document_text(zip_file(tmp_path, "deck.pptx", PPTX_PARTS))

    assert result.usable
    assert "Quarterly rollout plan" in result.text


@pytest.mark.parametrize("suffix", [".pdf", ".docx", ".xlsx", ".pptx"])
def test_a_payload_renamed_as_a_document_is_refused(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"weights{suffix}"
    path.write_bytes(b"GGUF" + b"\x00" * 64)

    result = document_text(path)

    assert not result.usable
    assert "does not start like one" in result.reason or "is not an Office file" in result.reason


def test_a_zip_without_the_part_its_suffix_promises_is_refused(tmp_path: Path) -> None:
    path = zip_file(tmp_path, "fake.docx", {"other/part.xml": "<x/>"})

    result = document_text(path)

    assert not result.usable
    assert "holds no word/document.xml" in result.reason


def test_a_document_past_the_size_cap_is_skipped_by_name(tmp_path: Path) -> None:
    path = docx(tmp_path)

    result = document_text(path, max_bytes=1)

    assert not result.usable
    assert "past the" in result.reason and "cap" in result.reason


def test_a_secret_inside_a_document_is_found_by_the_scan(platform, tmp_path, monkeypatch) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    docx(folder, "leaky.docx", f"{INVOICE} {SECRET}")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(folder))

    result = build.inspect_source(platform, str(folder))

    assert result.secret_hits
    assert result.text_files == 0
    assert not result.acceptable


def test_a_folder_of_documents_is_accepted_and_indexed(platform, tmp_path, monkeypatch) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    docx(folder)
    pdf(folder)
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(folder))

    result = build.inspect_source(platform, str(folder))

    assert result.acceptable
    assert result.text_files == 2
    assert result.formats == {".docx": 1, ".pdf": 1}
    indexed = build_tree_index(folder, source_id="src_docs")
    assert len(indexed.files) == 2
    assert any(INVOICE in item.text for item in indexed.files)


def test_a_folder_of_scans_says_why_it_cannot_be_used(platform, tmp_path, monkeypatch) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    pdf(folder, "scan.pdf", text=None)
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(folder))

    result = build.inspect_source(platform, str(folder))

    assert not result.acceptable
    assert result.unreadable
    assert "OCR" in " ".join(result.problems)


def test_a_single_document_is_a_supported_source(platform, tmp_path, monkeypatch) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    document = docx(folder)
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(folder))

    result = build.inspect_source(platform, str(document))

    assert result.acceptable
    assert result.text_files == 1
    assert result.formats == {".docx": 1}


def test_ingest_turns_a_document_into_paragraph_records(tmp_path: Path) -> None:
    source = Source(
        source_id="src_doc",
        path=docx(tmp_path),
        acl=["engineering"],
        approved=True,
        owner="tester",
    )

    records = ingest_source(source)

    assert [record.text for record in records] == [INVOICE, SHIPMENT]
    assert all(record.meta["format"] == "docx" for record in records)
