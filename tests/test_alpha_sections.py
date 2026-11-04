"""S3.1: questions a person can check, and a set that stays unfrozen without one.

The Y1.5 set is 32 questions a person wrote, which the 200-case promotion floor
refuses. The cheapest way to reach 200 is to have a model write the questions —
and that is the one way that must never happen, because the set would then be
judged on third-party output that the training text also cannot contain.

So questions come from the corpus itself: a sentence, a template chosen by the
sentence's own cue, and the sentence as the reference answer. Nothing here is
invented, and every item names the line it came from. Whether the set may be
frozen stays a person's decision: the script opens ``eval_set_freeze`` and stops.
"""

from __future__ import annotations

import json
from pathlib import Path

from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion, GateStatus
from internal_ai.ids import now_iso
from internal_ai.people import PeopleRegistry
from internal_ai.training.alpha.sections import (
    headings,
    prose,
    question_for,
    questions,
    section_for,
)

sys_path = Path(__file__).resolve().parents[1] / "scripts"
import sys  # noqa: E402

sys.path.insert(0, str(sys_path))
import alpha_y15_sections  # noqa: E402

MUST_SENTENCE = (
    "A server generating a 401 (Unauthorized) response MUST send a "
    "WWW-Authenticate header field containing at least one challenge."
)


def test_a_hyphenated_word_survives_the_corpus_line_breaks() -> None:
    """The corpus is wrapped at 72 columns, and the wrap splits a real term."""
    payload = "   A server MUST send a WWW-\n   Authenticate header field containing one challenge."

    assert "WWW-Authenticate" in prose(payload)
    assert "WWW- Authenticate" not in prose(payload)


def test_page_furniture_never_becomes_a_sentence() -> None:
    payload = (
        "   A name server must employ multiple concurrent activities.\n"
        "\n"
        "RFC 1035        Domain Implementation and Specification    November 1987\n"
        "\n"
        "Mockapetris [Page 33]\n"
    )

    text = prose(payload)

    assert "A name server must employ multiple concurrent activities." in text
    for furniture in ("RFC 1035", "Page 33", "November 1987"):
        assert furniture not in text


def test_a_definition_becomes_its_own_question() -> None:
    assert question_for("A URI-reference is either a URI or a relative reference.") == (
        "What is a URI-reference?"
    )


def test_a_requirement_names_the_subject_it_is_about() -> None:
    """The parenthetical is dropped and the modal is not repeated in the question."""
    question = question_for(MUST_SENTENCE)

    assert question == "What must a server generating a 401 response do?"
    assert question is not None and "MUST" not in question


def test_a_negated_requirement_keeps_its_negation() -> None:
    question = question_for(
        "A client MUST NOT send the Host header field twice in the same request."
    )

    assert question == "What must a client not do?"


def test_a_key_word_being_discussed_is_not_a_requirement() -> None:
    """A sentence about the notation says nothing a question can be asked of."""
    sentence = (
        'The key words "MUST", "MUST NOT", "REQUIRED" in this document are to be '
        "interpreted as described in RFC 2119."
    )

    assert question_for(sentence) is None


def test_a_grammar_rule_is_not_a_question() -> None:
    sentence = 'OWS = *( [ obs-fold ] WSP ) ; "optional" whitespace is not produced here.'

    assert questions(sentence) == []


def test_a_question_about_the_document_itself_is_refused() -> None:
    assert question_for("This document specifies the HTTP/1.1 message syntax already.") is None


def test_a_heading_is_the_section_a_line_sits_in(tmp_path: Path) -> None:
    source = tmp_path / "rfc9999.txt"
    source.write_text(
        "1.  Introduction\n\n   Some prose here that is long enough to matter.\n\n"
        "2.  Status Codes\n\n   More prose.\n\n2.1.  The 401 Response\n\n   Even more.\n",
        encoding="utf-8",
    )

    found = headings(source)

    assert [heading.label for heading in found] == [
        "1 Introduction",
        "2 Status Codes",
        "2.1 The 401 Response",
    ]
    assert section_for(found, 9) is not None
    assert section_for(found, 9).label == "2.1 The 401 Response"
    assert section_for(found, 0) is None


def test_a_table_of_contents_line_is_not_a_heading(tmp_path: Path) -> None:
    source = tmp_path / "rfc9998.txt"
    source.write_text(
        "Table of Contents\n\n   1.  Introduction ................................................ 3\n",
        encoding="utf-8",
    )

    assert headings(source) == []


def _install(tmp_path: Path, paragraphs: int) -> tuple[Path, str, Path, Path]:
    """A tiny released corpus whose every paragraph answers one definition."""
    rfc_dir = tmp_path / "rfc"
    rfc_dir.mkdir()
    lines = ["1.  Fields", "", "   Each field names one value a client sends.", ""]
    records = []
    for index in range(paragraphs):
        payload = f"The alpha{index} field is a header field that names value {index}."
        lines.extend(["", f"   {payload}", ""])
        records.append(
            {
                "record_key": f"rfc9999.txt:{len(lines) - 2}",
                "payload": payload,
                "source_id": "rfc",
                "acl": [],
                "sensitivity": "internal",
                "split": "test",
            }
        )
    (rfc_dir / "rfc9999.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    root = tmp_path / "install"
    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        platform.store.add_dataset(
            DatasetVersion(
                id="ds_sections",
                name="rfc",
                version=1,
                content_hash="hash-sections",
                record_count=len(records),
                token_count=len(records),
                created_at=now_iso(),
                released=True,
            )
        )
        platform.store.add_dataset_records("ds_sections", records)
        PeopleRegistry(platform.store).new_person(
            "Pat Reviewer", "Domain Expert", ["domain_expert"]
        )
    finally:
        platform.close()
    return root, "ds_sections", rfc_dir, tmp_path / "draft.json"


def test_a_set_below_the_floor_is_refused_and_nothing_is_filed(tmp_path: Path) -> None:
    root, dataset, rfc_dir, out = _install(tmp_path, paragraphs=3)

    code = alpha_y15_sections.main(
        [
            "--root",
            str(root),
            "--dataset",
            dataset,
            "--rfc-dir",
            str(rfc_dir),
            "--out",
            str(out),
            "--per-record",
            "1",
        ]
    )

    assert code == 1
    draft = json.loads(out.read_text(encoding="utf-8"))
    assert draft["meets_floor"] is False
    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        assert platform.store.list_eval_sets() == []
        assert platform.gates.pending() == []
    finally:
        platform.close()


def test_a_set_at_the_floor_requests_the_gate_and_stays_unfrozen(tmp_path: Path, capsys) -> None:
    root, dataset, rfc_dir, out = _install(tmp_path, paragraphs=210)

    code = alpha_y15_sections.main(
        [
            "--root",
            str(root),
            "--dataset",
            dataset,
            "--rfc-dir",
            str(rfc_dir),
            "--out",
            str(out),
            "--per-record",
            "1",
            "--limit",
            "240",
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["items"] >= 200
    assert result["frozen"] is False

    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        eval_set = platform.store.get_eval_set(result["eval_set_id"])
        assert eval_set is not None and eval_set.frozen is False
        gate = next(gate for gate in platform.gates.pending() if gate.id == result["gate_id"])
        assert gate.status == GateStatus.PENDING
        payload = list(platform.store.dataset_records(dataset))
        bodies = {row["record_key"]: row["payload"] for row in payload}
        for item in eval_set.items:
            key = item["relevant_ids"][0]
            assert item["answer"] in bodies[key]
    finally:
        platform.close()
