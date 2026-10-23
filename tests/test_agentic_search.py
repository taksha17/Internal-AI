"""Agentic retrieval: cited evidence and document-level permissions."""

from __future__ import annotations

from pathlib import Path

import pytest

from internal_ai.agentic import AgenticSearcher
from internal_ai.cli import main
from internal_ai.context import Platform
from internal_ai.data.records import Record
from internal_ai.domain import DatasetVersion
from internal_ai.serving.service import Principal

ENGINEERING = Record(
    id="rec_eng",
    record_key="ashby_autofill.md",
    text=(
        "The ashby autofill handler fills the application form for a candidate.\n"
        "It submits the candidate profile over the messaging channel.\n"
    ),
    source_id="repo",
    sensitivity="internal",
    acl=["engineering"],
    meta={"path": "ashby_autofill.md"},
)

LEGAL = Record(
    id="rec_legal",
    record_key="retention_policy.md",
    text=(
        "The retention policy keeps application data for thirty days.\n"
        "Deletion requests are handled by the privacy team.\n"
    ),
    source_id="repo",
    sensitivity="internal",
    acl=["legal"],
    meta={"path": "retention_policy.md"},
)

PUBLIC = Record(
    id="rec_public",
    record_key="readme.md",
    text="This workspace hosts the internal engineering tooling.\n",
    source_id="repo",
    sensitivity="internal",
    acl=[],
    meta={"path": "readme.md"},
)


def _add_dataset(platform: Platform, dataset_id: str, records: list[Record]) -> str:
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name=dataset_id,
            version=1,
            content_hash=f"hash-{dataset_id}",
            record_count=len(records),
            token_count=sum(len(record.text.split()) for record in records),
            created_at="now",
        )
    )
    platform.store.add_dataset_records(dataset_id, [record.to_row() for record in records])
    return dataset_id


def _searcher(platform: Platform) -> AgenticSearcher:
    return AgenticSearcher(platform.store, platform.audit, platform.budget, embedder=None, llm=None)


def _principal(user: str, *groups: str) -> Principal:
    return Principal(user_id=user, groups=frozenset(groups))


def test_agentic_search_cites_the_records_the_principal_may_read(platform) -> None:
    dataset_id = _add_dataset(platform, "ds_mixed", [ENGINEERING, LEGAL, PUBLIC])
    result = _searcher(platform).search(
        "ashby autofill", dataset_id, _principal("alice", "engineering")
    )
    assert result.sufficient_evidence is True
    assert "ashby_autofill.md" in result.files
    assert "retention_policy.md" not in result.files
    assert result.citations
    assert result.citations[0]["line"] >= 1


def test_agentic_search_hides_records_acl_ed_to_another_group(platform) -> None:
    dataset_id = _add_dataset(platform, "ds_mixed", [ENGINEERING, LEGAL, PUBLIC])
    result = _searcher(platform).search(
        "ashby autofill", dataset_id, _principal("mallory", "legal")
    )
    assert result.sufficient_evidence is False
    assert "ashby_autofill.md" not in result.files
    assert "autofill handler" not in result.answer


def test_agentic_search_lets_each_group_read_its_own_records(platform) -> None:
    dataset_id = _add_dataset(platform, "ds_mixed", [ENGINEERING, LEGAL, PUBLIC])
    result = _searcher(platform).search(
        "retention policy", dataset_id, _principal("mallory", "legal")
    )
    assert result.sufficient_evidence is True
    assert "retention_policy.md" in result.files
    assert "ashby_autofill.md" not in result.files


def test_agentic_search_denies_a_principal_with_nothing_readable(platform) -> None:
    dataset_id = _add_dataset(platform, "ds_restricted", [ENGINEERING])
    with pytest.raises(PermissionError):
        _searcher(platform).search("ashby autofill", dataset_id, _principal("mallory", "legal"))
    denials = [
        event
        for event in platform.audit.recent(limit=20)
        if event.action == "agentic_search.denied"
    ]
    assert denials
    assert denials[0].outcome == "denied"


def test_cli_ask_reports_a_permission_denial(tmp_path: Path, capsys) -> None:
    root = tmp_path / "cli"
    root.mkdir()
    with Platform.create(root=root) as platform:
        _add_dataset(platform, "ds_restricted", [ENGINEERING])
    exit_code = main(
        [
            "--root",
            str(root),
            "ask",
            "ashby autofill",
            "--dataset-id",
            "ds_restricted",
            "--user",
            "mallory",
            "--groups",
            "legal",
        ]
    )
    assert exit_code == 1
    assert "readable by mallory" in capsys.readouterr().err
