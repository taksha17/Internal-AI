# Document extraction — design

Intent: a folder of the documents a company actually keeps should be usable as a
source, instead of being refused for holding no text files.

Why: the scanner's formats are code and plain text. There is no PDF, DOCX, XLSX
or PPTX anywhere in the tree, so pointing Internal-AI at a folder of policy PDFs
and Word documents produces "the folder has no text files to learn from" and a
refusal. That is the first wall anyone aiming this at their own data hits, and it
is the one the product promise runs straight into.

## Goals

- PDF, DOCX, XLSX and PPTX files count as text and yield their text to the
  corpus, at scan time and at ingest time, through one shared rule.
- Extracted text goes through the checks that already exist: the scan's secret
  patterns, then the pipeline's PII scrub. Nothing bypasses them by arriving in
  a different container.
- A payload that is not what its suffix claims stays refused. A renamed weights
  file must not slip through by being named `.docx`.
- Honest limits, reported rather than silently swallowed: a scanned PDF with no
  text layer is named as such, and size or length caps say what they dropped.
- The parsed text keeps a usable path trail, so retrieval can cite the document
  it came from.

## Non-goals

- No OCR. An image-only PDF is reported as unreadable, not guessed at.
- No legacy binary Office formats (`.doc`, `.xls`, `.ppt`). They need a different
  parser and are not the common modern case.
- No images, audio or video.
- No caching or pre-conversion step: extraction runs where the text is needed.
- No change to gates, roles, budgets, refusal rules or the sensitive-path
  classification.

## Anti-goals

- Treating a document's bytes as text so that secret scanning runs over binary
  noise would be worse than refusing it: it reports findings that are not there
  and misses the ones that are.
- Quietly skipping a document that failed to extract would make a corpus look
  complete when it is not.

## What the code does today

- `data/inventory.py:TEXT_SUFFIXES` and `data/tree.py:TEXT_SUFFIXES` are separate
  copies of the same plain-text list. Neither has a document format.
- `inventory.disguised_binary` calls a file binary when its first 512 bytes hold
  a NUL or start with `GGUF` or `PK\x03\x04`. A DOCX is a ZIP, so it matches
  `PK\x03\x04` — the guard is aimed at renamed weights and would reject every
  Office file if it were applied to them.
- `services/build.py:inspect_source` reads each candidate with
  `read_text(encoding="utf-8", errors="ignore")`, which on a PDF yields mojibake,
  and `data/tree.py:build_tree_index` does the same before checking secrets.
- `data/records.py:ingest_source` reads `.jsonl`, `.csv` and plain text.
- Dependencies are deliberately lean: `fastapi`, `uvicorn`, `pydantic`, `httpx`,
  `numpy`, `jinja2`, `python-multipart`, with `torch` behind the `train` extra.

## Approaches considered

- **In-platform extraction (picked).** The platform understands documents, so
  the scan, the secret check and the PII scrub all see the real text.
- **Out-of-band conversion.** A documented step the person runs before pointing
  the tool at a folder. Keeps the platform text-only, but hands a novice back the
  exact problem this is meant to remove, and leaves the scan unable to say
  anything useful about a `.pdf` it is handed anyway.
- **Convert once into a sidecar text mirror.** Faster on repeat scans; a second
  store of the same content to keep in step with the originals, and unrequested
  here. Left out.

## Decisions

- One module owns extraction, and the inventory scan, `inspect_source`, the
  tree index and ingest all call it, so a document cannot arrive by one door and
  be missed by another. The two plain-text suffix lists stay where they are: they
  differ already (the tree's also accepts `changelog`), and merging them would
  smuggle a behaviour change in beside this one.
- Paragraphs become chunks. A Word paragraph, a slide's paragraph and a
  spreadsheet cell each end up in their own record, and a cell keeps a boundary
  because Excel stores its strings in one flat table with nothing between them.
  Reconstructing spreadsheet rows means resolving shared-string indices, which is
  not in this cut: a spreadsheet arrives as its cells, not as its rows.
- DOCX, XLSX and PPTX are ZIP containers of XML, so they are read with the
  standard library alone (`zipfile` plus `ElementTree`): no dependency for the
  Office formats.
- PDF needs a parser. `pypdf` is small and pure Python, and it is the one new
  dependency.
- A container's magic is checked against its suffix before anything is parsed:
  PDF must start `%PDF-`, and a ZIP-based suffix must actually contain the part
  that suffix promises (`word/document.xml`, `xl/workbook.xml`,
  `ppt/presentation.xml`). A renamed payload fails that and stays refused.
- Extracted text is treated exactly like a text file from then on: the same
  secret patterns, the same PII scrub, the same quality filter.
- Extraction is bounded, and the bound is reported: a document past the size cap
  is skipped by name, and extracted text past the length cap is truncated with
  the truncation counted.
- The formats join `TEXT_SUFFIXES`, so `formats` in the scan and the suggester
  see them, and a folder of documents can recommend search the same way a folder
  of Markdown does.

## What is left to the builder

- The exact part names and the run-extraction per format, and how paragraphs and
  rows become text so a document does not read as one unbroken line.
- The size and length caps, chosen against the caps already in the scan.
- Wording for a scanned PDF, an unreadable container and a skipped oversize file.
- Whether a document that yields no text counts towards `text_files` at all. It
  must not, or a corpus of scans would look usable.
- Tests: a PDF, a DOCX, an XLSX and a PPTX each yield their text; a scanned PDF
  is reported and not counted; a weights file renamed `.docx` is refused; a
  secret inside a PDF is found by the scan; a folder of documents is accepted and
  becomes a dataset with usable records.

## Builder check (self-read)

- Answered already: the three readers that need the shared rule, the ZIP trick
  for Office formats, and the reason the disguised-binary guard cannot simply be
  applied to them.
- My calls: PPTX included because it shares the Office path exactly; legacy
  `.doc`/`.xls`/`.ppt` excluded as a different parser; no sidecar cache.
- Theirs: one, below.

## Decided on the dependency

`pypdf` is a core dependency. A refusal that reads "install an extra" is the
same wall in a different place, and the Office formats need nothing at all, so
the extra would have gated only PDF. The import stays inside the PDF reader, so
the module still loads and reports honestly if the package is ever missing.
