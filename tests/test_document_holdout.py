"""A held-out document is held out whole, or the test set is training text.

The RFC corpus splits paragraphs, so a held-out line's neighbours are in the
training text and the model can answer by having read the next paragraph. A
document-level holdout fixes that, and this is the switch for it.
"""

from __future__ import annotations

from internal_ai.data.records import Record
from internal_ai.data.splits import make_splits

# Each record gets a vocabulary of its own, so the near-duplicate guard has no
# reason to drop a held-out line and the split is what the test is measuring.
WORDS = [f"term{number}" for number in range(600)]


def _text(index: int, line: int) -> str:
    start = (index * 37 + line * 11) % 600
    return " ".join(WORDS[(start + step * 13) % 600] for step in range(12)) + "."


def _documents(count: int, per_document: int) -> list[Record]:
    return [
        Record(
            id=f"{index}-{line}",
            record_key=f"rfc9{index:03d}.txt:{line}",
            text=_text(index, line),
            source_id="rfc",
        )
        for index in range(count)
        for line in range(per_document)
    ]


def _document(record: Record) -> str:
    return record.record_key.rsplit(":", 1)[0]


def test_a_document_never_spans_two_splits() -> None:
    records = _documents(count=12, per_document=6)

    result = make_splits(records, group_of=_document)

    homes: dict[str, set[str]] = {}
    for name, split_records in result.splits.items():
        for record in split_records:
            homes.setdefault(_document(record), set()).add(name)
    assert homes
    assert all(len(names) == 1 for names in homes.values()), homes
    assert result.stats["grouped_by"] == "document"
    assert len(result.stats["groups"]) == 12
    assert sum(result.stats["groups"].values()) == 12 * 6


def test_a_named_document_is_held_out_whole() -> None:
    records = _documents(count=10, per_document=5)

    result = make_splits(records, group_of=_document, test_groups={"rfc9003.txt"})

    test_links = {record.record_key for record in result.splits["test"]}
    train_links = {record.record_key for record in result.splits["train"]}
    assert {f"rfc9003.txt:{line}" for line in range(5)} <= test_links
    assert not any(key.startswith("rfc9003.txt:") for key in train_links)
    assert result.stats["test_groups"] == ["rfc9003.txt"]


def test_document_grouping_is_stable_across_runs() -> None:
    records = _documents(count=8, per_document=4)

    first = make_splits(records, group_of=_document)
    second = make_splits(records, group_of=_document)

    assert first.stats["counts"] == second.stats["counts"]
    assert {name: [r.record_key for r in items] for name, items in first.splits.items()} == {
        name: [r.record_key for r in items] for name, items in second.splits.items()
    }


def test_the_default_split_still_splits_by_record() -> None:
    """An existing released dataset was split this way; its hash cannot move."""
    records = _documents(count=8, per_document=4)

    result = make_splits(records)

    assert "grouped_by" not in result.stats
    assert result.stats["seed"] == "forge-v1"
