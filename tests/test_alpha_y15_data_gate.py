"""The corpus script asks for a release; it does not sign one.

``dataset_release`` is a human gate. The script used to approve every required
approver by proxy with the owner's standing instruction as the note and then
release, which meant the record said people had decided something nobody had
been asked. It now builds, splits, measures and asks — and releases only when a
decided gate is already there, which the pipeline checks itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from programme import record_ready_programme

from internal_ai.domain import GateStatus
from internal_ai.people import PeopleRegistry
from internal_ai.services.actor import Actor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import alpha_y15_data


def _person(platform, role: str) -> dict:
    name = alpha_y15_data.ROSTER[role][0]
    people = {person["name"]: person for person in PeopleRegistry(platform.store).list_people()}
    return people[name]


def _run(root: Path, corpus: Path, out: Path, *extra: str) -> int:
    return alpha_y15_data.main(
        [
            "--root",
            str(root),
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--name",
            "rfc",
            *extra,
        ]
    )


def test_the_run_asks_for_the_release_and_leaves_the_dataset_unreleased(
    platform, corpus_path: Path, tmp_path: Path, capsys
) -> None:
    root = platform.settings.root
    out = tmp_path / "dataset.json"

    assert _run(root, corpus_path, out) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["released"] is False
    assert result["language_review"]["sampled"] > 0
    stored = platform.store.get_dataset(result["dataset_id"])
    assert stored is not None
    assert stored.released is False
    gate = platform.store.get_gate(result["release_gate"])
    assert gate is not None
    assert gate.status == GateStatus.PENDING


def test_the_requester_is_the_data_owner_account(
    platform, corpus_path: Path, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "dataset.json"

    assert _run(platform.settings.root, corpus_path, out) == 0

    result = json.loads(capsys.readouterr().out)
    gate = platform.store.get_gate(result["release_gate"])
    assert gate is not None
    assert gate.requested_by == _person(platform, "data_owner")["id"]


def test_release_refuses_while_the_gate_is_still_pending(
    platform, corpus_path: Path, tmp_path: Path, capsys
) -> None:
    """The second run may release, but only on a gate somebody decided."""
    out = tmp_path / "dataset.json"
    assert _run(platform.settings.root, corpus_path, out) == 0
    first = json.loads(capsys.readouterr().out)

    assert _run(platform.settings.root, corpus_path, out, "--release") == 1

    stored = platform.store.get_dataset(first["dataset_id"])
    assert stored is not None
    assert stored.released is False


def test_a_decided_gate_releases_the_dataset(
    platform, corpus_path: Path, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "dataset.json"
    assert _run(platform.settings.root, corpus_path, out) == 0
    first = json.loads(capsys.readouterr().out)
    record_ready_programme(
        platform, Actor.from_person(_person(platform, "ml_lead"), authenticated=True)
    )
    gate = platform.store.get_gate(first["release_gate"])
    assert gate is not None
    for role in gate.required_approvers:
        person = _person(platform, role)
        platform.gates.decide(gate.id, person["name"], approved=True, note="read it", evidence={})

    assert _run(platform.settings.root, corpus_path, out, "--release") == 0

    stored = platform.store.get_dataset(first["dataset_id"])
    assert stored is not None
    assert stored.released is True
