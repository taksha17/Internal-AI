"""Ask retrieves first, then the on-prem runner writes under a citation check."""

from __future__ import annotations

from internal_ai.agentic import AgenticSearchResult
from internal_ai.domain import DatasetVersion
from internal_ai.ids import now_iso
from internal_ai.llm import Completion
from internal_ai.services.build import ask, served_embedder, written_answer


class FakeRunner:
    name = "fake-runner"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, prompt: str, system: str = "", **_: object) -> Completion:
        return Completion(text=self.text, model=self.name)


def _dataset(platform, released: bool = True) -> str:
    dataset_id = "ds_ask"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="tree",
            version=1,
            content_hash="abc",
            record_count=2,
            token_count=40,
            created_at=now_iso(),
            released=released,
            stats={"token_count_kind": "measured"},
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": "src/a.txt",
                "payload": "the sign out lives here\nreset the profile",
                "acl": ["engineering"],
                "meta": {"path": "src/a.txt"},
            },
            {
                "record_key": "src/b.txt",
                "payload": "reconcile the ledger\nclose the books",
                "acl": ["engineering"],
                "meta": {"path": "src/b.txt"},
            },
        ],
    )
    return dataset_id


def _result() -> AgenticSearchResult:
    return AgenticSearchResult(
        question="where is the sign out",
        answer="Found 2 supporting locations across 1 files:\n- src/a.txt:1 the sign out lives here",
        citations=[
            {"path": "src/a.txt", "line": 1, "text": "the sign out lives here"},
            {"path": "src/b.txt", "line": 12, "text": "reconcile the ledger"},
        ],
        sufficient_evidence=True,
    )


def test_a_moved_line_is_stored_as_the_hit_list(platform) -> None:
    dataset_id = _dataset(platform)
    result = _result()
    written = written_answer(
        platform,
        result,
        dataset_id,
        adapter=FakeRunner("The sign out is in src/b.txt:1 and it clears the profile."),
    )
    assert written.citation_passed is False
    assert written.answer == "src/a.txt:1\nsrc/b.txt:12"
    assert "src/b.txt" in "\n".join(written.mismatches)
    events = platform.store.list_audit(action="ask.written", limit=5)
    assert events and str(events[-1].outcome) == "failure"


def test_a_correct_line_is_stored_as_the_written_answer(platform) -> None:
    dataset_id = _dataset(platform)
    written = written_answer(
        platform,
        _result(),
        dataset_id,
        adapter=FakeRunner("The sign out is in src/a.txt:1, which clears the profile."),
    )
    assert written.citation_passed is True
    assert written.answer.startswith("The sign out is in src/a.txt:1")
    assert written.written_by == "fake-runner"
    events = platform.store.list_audit(action="ask.written", limit=5)
    assert events and str(events[-1].outcome) == "success"


def test_ask_without_a_promoted_model_returns_keyword_hits(platform, enroll_person) -> None:
    dataset_id = _dataset(platform)
    engineer = enroll_person(platform, "Ada Engineer", ["ml_lead"], ["engineering"])
    assert served_embedder(platform) is None
    result = ask(platform, engineer, dataset_id, "where is the sign out")
    assert result.files
    assert result.written_by == ""
    assert result.steps[0].strategy == "initial"


def test_ask_refuses_an_unreleased_dataset(platform, enroll_person) -> None:
    import pytest

    dataset_id = _dataset(platform, released=False)
    engineer = enroll_person(platform, "Ada Engineer", ["ml_lead"], ["engineering"])
    with pytest.raises(PermissionError, match="released datasets only"):
        ask(platform, engineer, dataset_id, "where is the sign out")
