"""A source needs a legal status before anyone can approve it, and release re-checks."""

from __future__ import annotations

import pytest

from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.samples import generate_corpus
from internal_ai.domain import AuditOutcome, GateType
from internal_ai.services import build, governance
from internal_ai.workflows import build_corpus


def test_a_source_without_legal_status_cannot_open_a_gate(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Dana Owner", ["data_owner"])
    with pytest.raises(ValueError, match="legal_status"):
        governance.request_gate(
            platform,
            owner,
            GateType.NEW_DATA_SOURCE,
            title="Use the notes",
            subject="src_notes",
            evidence={
                "source_description": "notes",
                "sensitivity_class": "internal",
                "purpose": "learn the flow",
            },
        )
    with pytest.raises(ValueError, match="license_name"):
        governance.request_gate(
            platform,
            owner,
            GateType.NEW_DATA_SOURCE,
            title="Use the notes",
            subject="src_licensed",
            evidence={
                "source_description": "notes",
                "sensitivity_class": "internal",
                "purpose": "learn the flow",
                "legal_status": "licensed",
            },
        )
    with pytest.raises(ValueError, match="refused"):
        governance.request_gate(
            platform,
            owner,
            GateType.NEW_DATA_SOURCE,
            title="Use the notes",
            subject="src_refused",
            evidence={
                "source_description": "notes",
                "sensitivity_class": "internal",
                "purpose": "learn the flow",
                "legal_status": "refused",
            },
        )
    gate = governance.request_gate(
        platform,
        owner,
        GateType.NEW_DATA_SOURCE,
        title="Use the notes",
        subject="src_internal",
        evidence={
            "source_description": "notes",
            "sensitivity_class": "internal",
            "purpose": "learn the flow",
            "legal_status": "internal",
        },
    )
    assert gate.status == "pending"


def test_release_refuses_an_excluded_class_and_records_a_clean_sample(platform) -> None:
    path = platform.settings.root / "tiny.jsonl"
    generate_corpus(path, records=8, seed=1)
    built = build_corpus(platform, [path], name="legal-check")
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    row = platform.store.dataset_records(built.dataset_id)[0]
    platform.store.conn.execute(
        "UPDATE dataset_records SET sensitivity = ? WHERE dataset_id = ? AND record_key = ?",
        ("regulated", built.dataset_id, row["record_key"]),
    )
    with pytest.raises(PermissionError, match="regulated"):
        pipeline.release(built.dataset_id, "ml_lead", gate_approved=True)
    assert platform.store.get_dataset(built.dataset_id).released is False
    denied = platform.store.conn.execute(
        "SELECT outcome, details FROM audit_events WHERE action = 'dataset.exclusion_audit'"
        " ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert denied["outcome"] == AuditOutcome.DENIED
    assert "regulated" in denied["details"]

    platform.store.conn.execute(
        "UPDATE dataset_records SET sensitivity = ? WHERE dataset_id = ? AND record_key = ?",
        ("internal", built.dataset_id, row["record_key"]),
    )
    from internal_ai.data.language import accept_language_sample

    accept_language_sample(platform.store, built.dataset_id)
    counted = platform.store.get_dataset(built.dataset_id)
    stats = dict(counted.stats)
    stats["token_count_kind"] = "measured"
    platform.store.update_dataset_tokens(built.dataset_id, counted.token_count, stats)
    released = pipeline.release(built.dataset_id, "ml_lead", gate_approved=True)
    audit = released.stats["exclusion_audit"]
    assert audit["passed"] is True
    assert audit["seed"] == "pii-audit"
    assert audit["sampled"] <= 50


def test_registered_source_keeps_its_legal_status(platform, enroll_person) -> None:
    owner = enroll_person(platform, "Dana Owner", ["data_owner"])
    path = platform.settings.root / "data" / "tickets.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_corpus(path, records=4, seed=2)
    source = build.register_source(
        platform,
        owner,
        str(path),
        "tickets",
        "search",
        ["engineering"],
        legal_status="licensed",
        license_name="Internal use",
        license_url="https://example.internal/license",
        license_note="training on this export is allowed",
    )
    assert source["legal_status"] == "licensed"
    assert source["license_name"] == "Internal use"
