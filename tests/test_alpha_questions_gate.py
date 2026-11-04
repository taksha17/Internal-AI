"""The Y1.5 set is filed by the script and frozen only by a person.

The script used to approve its own ``eval_set_freeze`` gate and pass the owner's
standing instruction as the approver's note. A gate decided by the agent that
wrote the questions is not a gate: the point of asking a person is that the
person can say no. So the script now files the set, opens the gate, and stops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from internal_ai.context import Platform
from internal_ai.domain import DatasetVersion, GateStatus
from internal_ai.ids import now_iso
from internal_ai.people import PeopleRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import alpha_y15_questions


def _install(tmp_path: Path, *, expert: bool = True) -> tuple[Path, Path]:
    root = tmp_path / "install"
    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        records = [
            {
                "record_key": citation,
                "payload": answer,
                "source_id": "rfc",
                "acl": [],
                "sensitivity": "internal",
                "split": "test",
            }
            for citation, _query, answer in alpha_y15_questions.QUESTIONS
        ]
        platform.store.add_dataset(
            DatasetVersion(
                id="ds_y15",
                name="rfc",
                version=1,
                content_hash="hash-y15",
                record_count=len(records),
                token_count=len(records),
                created_at=now_iso(),
                released=True,
            )
        )
        platform.store.add_dataset_records("ds_y15", records)
        if expert:
            PeopleRegistry(platform.store).new_person(
                "Pat Reviewer", "Domain Expert", ["domain_expert"]
            )
    finally:
        platform.close()
    return root, tmp_path / "eval.json"


def _run(root: Path, out: Path) -> int:
    return alpha_y15_questions.main(["--root", str(root), "--dataset", "ds_y15", "--out", str(out)])


def test_the_set_is_filed_unfrozen_with_the_gate_waiting_for_a_person(
    tmp_path: Path, capsys
) -> None:
    root, out = _install(tmp_path)

    assert _run(root, out) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["frozen"] is False
    assert result["items"] == len(alpha_y15_questions.QUESTIONS)
    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        eval_set = platform.store.get_eval_set(result["eval_set_id"])
        assert eval_set is not None
        assert eval_set.frozen is False
        assert not eval_set.approved_by
        gate = next(item for item in platform.gates.pending() if item.id == result["gate_id"])
        assert gate.status == GateStatus.PENDING
    finally:
        platform.close()


def test_the_requester_holds_the_expert_role_rather_than_the_agent(tmp_path: Path, capsys) -> None:
    root, out = _install(tmp_path)

    assert _run(root, out) == 0

    result = json.loads(capsys.readouterr().out)
    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        gate = platform.store.get_gate(result["gate_id"])
        assert gate is not None
        people = {person["id"]: person for person in PeopleRegistry(platform.store).list_people()}
        requester = people[gate.requested_by]
        assert "domain_expert" in requester["roles"]
    finally:
        platform.close()


def test_with_no_expert_the_set_is_not_filed_at_all(tmp_path: Path) -> None:
    """Filing the set first would leave a set nobody can freeze."""
    root, out = _install(tmp_path, expert=False)

    assert _run(root, out) == 1

    assert not out.exists()
    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        assert platform.store.list_eval_sets() == []
        assert platform.gates.pending() == []
    finally:
        platform.close()
