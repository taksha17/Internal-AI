"""Data plane: deduplication, PII scrubbing, exclusion, lineage and splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.data.splits import contamination_report
from internal_ai.data.tokenizer import coverage_report, train_tokenizer
from internal_ai.data.transforms import deduplicate, filter_quality, scrub_pii
from internal_ai.workflows import build_corpus, derive_eval_items, split_corpus


def test_pipeline_scrubs_pii_and_drops_duplicates(platform, corpus_path) -> None:
    result = build_corpus(
        platform, [corpus_path], name="corpus-test", approved=True, owner="data_owner"
    )
    steps = {step["step"]: step for step in result.stats["steps"]}
    assert steps["pii_scrub"]["total_redactions"] > 0
    assert steps["deduplicate"]["exact_duplicates_removed"] > 0
    assert steps["sensitivity_exclusion"]["output"] < steps["sensitivity_exclusion"]["input"]
    payloads = [row["payload"] for row in platform.store.dataset_records(result.dataset_id)]
    assert all("@example.com" not in payload for payload in payloads)
    assert all("AKIAIOSFODNN7EXAMPLE" not in payload for payload in payloads)


def test_dataset_is_versioned_with_lineage(platform, corpus_path) -> None:
    result = build_corpus(platform, [corpus_path], name="corpus-test")
    dataset = platform.store.get_dataset(result.dataset_id)
    assert dataset is not None
    assert dataset.content_hash
    assert dataset.lineage["sources"][0]["source_id"] == corpus_path.stem
    assert dataset.lineage["sources"][0]["content_hash"]
    assert dataset.lineage["code_commit"]
    assert dataset.lineage["config_hash"]
    assert len(dataset.lineage["steps"]) >= 5


def test_a_row_cannot_widen_access_or_mark_itself_readable(tmp_path, platform) -> None:
    from internal_ai.data.records import Source, ingest_source
    from internal_ai.serving.service import Principal

    path = tmp_path / "notes.jsonl"
    path.write_text(
        '{"text": "keep this private", "acl": [], "sensitivity": "confidential"}\n'
        '{"text": "reset the vpn", "acl": ["public", "engineering"], "sensitivity": "internal"}\n',
        encoding="utf-8",
    )
    records = ingest_source(
        Source(
            source_id="src_notes",
            path=path,
            sensitivity="internal",
            acl=["engineering"],
            approved=True,
        )
    )
    assert [record.acl for record in records] == [["engineering"], ["engineering"]]
    reader = Principal(user_id="ada", groups=frozenset({"engineering"}))
    assert reader.can_read(records[0].acl, records[0].sensitivity) is False
    assert reader.can_read(records[1].acl, records[1].sensitivity) is True


def test_a_row_may_not_lower_the_sensitivity_of_its_source(tmp_path, platform) -> None:
    from internal_ai.data.records import Source, ingest_source

    path = tmp_path / "notes.jsonl"
    path.write_text(
        '{"text": "keep this private", "sensitivity": "internal"}\n'
        '{"text": "reset the vpn", "sensitivity": "regulated"}\n',
        encoding="utf-8",
    )
    records = ingest_source(
        Source(
            source_id="src_notes",
            path=path,
            sensitivity="confidential",
            acl=["engineering"],
            approved=True,
        )
    )
    assert [record.sensitivity for record in records] == ["confidential", "regulated"]


def test_a_tree_dataset_keeps_the_class_of_its_source(tmp_path, platform) -> None:
    from internal_ai.data.tree import build_tree_index
    from internal_ai.workflows import build_tree_corpus

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "note.txt").write_text("reset the vpn\n")
    index = build_tree_index(folder, source_id="src_docs")
    built = build_tree_corpus(platform, index, name="docs", sensitivity="confidential")
    dataset = platform.store.get_dataset(built.dataset_id)
    assert dataset is not None
    rows = platform.store.dataset_records(built.dataset_id)
    assert [row["sensitivity"] for row in rows] == ["confidential"]
    assert dataset.lineage["sources"][0]["sensitivity"] == "confidential"


def test_the_class_of_a_registered_source_reaches_its_dataset(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enroll_person
) -> None:
    from internal_ai.services import build, governance
    from internal_ai.services.jobs import JobRunner

    root = tmp_path / "data"
    root.mkdir()
    (root / "notes.txt").write_text("reset the vpn\n")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))
    owner = enroll_person(platform, "Dee Owner", ["data_owner"])
    reviewer = enroll_person(platform, "Rae Reviewer", ["data_owner"])
    compliance = enroll_person(platform, "Cleo Compliance", ["compliance"])
    source = build.register_source(
        platform, owner, str(root / "notes.txt"), "notes", "learn", ["engineering"]
    )
    platform.store.conn.execute(
        "UPDATE data_sources SET sensitivity = ? WHERE id = ?", ("confidential", source["id"])
    )
    governance.decide_gate(
        platform,
        reviewer,
        source["gate_id"],
        approved=True,
        note="the class was read from the scan",
    )
    governance.decide_gate(
        platform, compliance, source["gate_id"], approved=True, note="classified confidential"
    )

    result = build.build_dataset(
        platform, owner, JobRunner(platform, synchronous=True), source["id"]
    )

    assert result["state"] == "succeeded", result.get("error")
    dataset_id = result["result"]["dataset_id"]
    dataset = platform.store.get_dataset(dataset_id)
    assert dataset is not None
    assert dataset.lineage["sources"][0]["sensitivity"] == "confidential"
    assert platform.store.dataset_records(dataset_id) == []


def test_a_password_assignment_is_scrubbed_and_blocks_release_if_it_remains() -> None:
    from internal_ai.data.governance import pii_spot_check
    from internal_ai.data.records import Record
    from internal_ai.data.transforms import scrub_pii

    secret = "password: 'hunter2hunter2'"
    record = Record(id="1", record_key="1", text=f"reset the vpn\n{secret}\n", source_id="s")
    scrubbed, stats = scrub_pii([record])
    assert "hunter2hunter2" not in scrubbed[0].text
    assert stats["total_redactions"] >= 1
    leaked = Record(id="2", record_key="2", text=secret, source_id="s")
    audit = pii_spot_check([leaked])
    assert audit.passed is False
    assert audit.findings.get("password_assignment", 0) >= 1


def test_unapproved_source_is_not_ingested(platform, corpus_path) -> None:
    result = build_corpus(platform, [corpus_path], name="corpus-test", approved=False)
    assert result.records == 0
    denied = [
        event for event in platform.audit.recent(limit=50) if event.action == "data.source_rejected"
    ]
    assert denied


def test_splits_are_deterministic_and_leak_free(platform, corpus_path) -> None:
    result = build_corpus(platform, [corpus_path], name="corpus-test")
    first = split_corpus(platform, result.dataset_id)
    second = split_corpus(platform, result.dataset_id)
    assert first.stats["counts"] == second.stats["counts"]
    rows = platform.store.dataset_records(result.dataset_id)
    splits = {row["record_key"]: row["split"] for row in rows}
    assert set(splits.values()) <= {"train", "validation", "test", ""}
    train_keys = {key for key, value in splits.items() if value == "train"}
    eval_keys = {key for key, value in splits.items() if value in ("test", "validation")}
    assert train_keys and eval_keys
    assert not train_keys & eval_keys


def test_contamination_report_flags_overlap() -> None:
    from internal_ai.data.records import Record

    records = [
        Record(id="1", record_key="a", text="the ledgerline feed failed", source_id="s"),
        Record(id="2", record_key="b", text="unrelated warehouse audit note", source_id="s"),
    ]
    report = contamination_report(records, records)
    assert report["exact_overlap"] == 2
    assert not report["passed"]


def test_deduplicate_removes_exact_and_near_duplicates() -> None:
    from internal_ai.data.records import Record

    base = "the ledgerline feed failed during the warehouse audit reconciliation run"
    records = [
        Record(id="1", record_key="a", text=base, source_id="s"),
        Record(id="2", record_key="b", text=base, source_id="s"),
        Record(id="3", record_key="c", text=base + " tonight", source_id="s"),
        Record(
            id="4",
            record_key="d",
            text="a completely different document about billing",
            source_id="s",
        ),
    ]
    kept, stats = deduplicate(records, threshold=12)
    assert len(kept) == 2
    assert stats["exact_duplicates_removed"] == 1
    assert stats["near_duplicates_removed"] == 1


def test_scrubber_redacts_but_keeps_record() -> None:
    from internal_ai.data.records import Record

    records = [
        Record(
            id="1",
            record_key="a",
            text="reach me at person@example.com or 415-555-0134 key AKIAIOSFODNN7EXAMPLE",
            source_id="s",
        )
    ]
    scrubbed, stats = scrub_pii(records)
    text = scrubbed[0].text
    assert "person@example.com" not in text
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "[REDACTED:email]" in text
    assert stats["total_redactions"] >= 3


def test_quality_filter_drops_short_and_foreign_records() -> None:
    from internal_ai.data.records import Record

    records = [
        Record(id="1", record_key="a", text="short", source_id="s"),
        Record(
            id="2",
            record_key="b",
            text="12345 67890 12345 67890 12345 67890 12345 67890",
            source_id="s",
        ),
        Record(
            id="3",
            record_key="c",
            text="the warehouse audit confirmed the invoice reconciliation matched the ledgerline feed",
            source_id="s",
        ),
    ]
    kept, stats = filter_quality(records, min_chars=20)
    assert [record.record_key for record in kept] == ["c"]
    assert stats["dropped_by_reason"]["too_short"] == 1


def test_tokenizer_is_trained_from_scratch_and_covers_jargon() -> None:
    texts = [
        "the ledgerline feed is reconciled with the fulfilmenthub for tenantquotas",
        "backorderqueue depth grows when the warehouse audit is delayed",
    ] * 12
    tokenizer = train_tokenizer(texts, vocab_size=400)
    assert tokenizer.vocab_size > 50
    assert tokenizer.encode("ledgerline") == [tokenizer.vocab["ledgerline"]]
    coverage = coverage_report(tokenizer, {"s": texts}, jargon=["ledgerline", "backorderqueue"])
    assert coverage["jargon_single_token_coverage"] == 1.0
    assert coverage["per_source"]["s"]["mean_tokens_per_word"] <= 1.5


def test_eval_items_come_from_the_held_out_split(platform, corpus_path) -> None:
    result = build_corpus(platform, [corpus_path], name="corpus-test")
    split_corpus(platform, result.dataset_id)
    items = derive_eval_items(platform, result.dataset_id)
    assert items
    test_keys = {
        row["record_key"]
        for row in platform.store.dataset_records_by_split(result.dataset_id, "test")
    }
    for item in items:
        assert set(item["relevant_ids"]) <= test_keys
