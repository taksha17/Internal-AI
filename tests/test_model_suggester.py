"""The suggester names the work a source can carry, and never overreaches.

A recommendation to train is only ever made on evidence the scan actually
measured: a label column for a classifier, and volume for a model trained from
scratch. Everything unready stays visible and says what it would need.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.data.suggest import (
    DECIDE,
    GENERATE,
    SEARCH,
    label_columns,
    suggest_model_types,
)
from internal_ai.services import build

TOKEN = re.compile(r"/console/sources/inspect/([A-Za-z0-9_-]+)")


def kinds(options: list[dict]) -> list[str]:
    return [option["kind"] for option in options]


def entry(options: list[dict], kind: str) -> dict:
    return next(option for option in options if option["kind"] == kind)


def prose(**over: object) -> dict:
    summary = {
        "kind": "tree",
        "files": 40,
        "text_files": 40,
        "formats": {".md": 30, ".txt": 10},
        "label_columns": [],
    }
    summary.update(over)
    return summary


def test_a_folder_of_prose_recommends_search_and_never_generate() -> None:
    options = suggest_model_types(prose())

    assert kinds(options)[0] == SEARCH
    assert entry(options, SEARCH)["ready"] is True
    assert entry(options, DECIDE)["ready"] is False
    assert entry(options, GENERATE)["ready"] is False
    assert "keyword search wins" in entry(options, GENERATE)["why"]
    assert "a column that holds the answer" in entry(options, DECIDE)["needs"]


def test_a_label_column_is_what_makes_decide_the_best_fit() -> None:
    options = suggest_model_types(prose(formats={".csv": 40}, label_columns=["label"]))

    assert kinds(options)[0] == DECIDE
    assert entry(options, DECIDE)["ready"] is True
    assert "label" in entry(options, DECIDE)["why"]
    assert entry(options, DECIDE)["needs"] == ""


def test_a_table_without_a_label_column_does_not_outrank_search() -> None:
    options = suggest_model_types(prose(formats={".csv": 40}, label_columns=[]))

    assert kinds(options)[0] == SEARCH
    assert entry(options, DECIDE)["ready"] is False
    assert "what the answer is" in entry(options, DECIDE)["why"]
    assert entry(options, DECIDE)["needs"]


def test_only_volume_makes_a_generative_model_the_best_fit() -> None:
    small = suggest_model_types(prose(records=98))
    assert entry(small, GENERATE)["ready"] is False
    assert kinds(small)[0] == SEARCH

    large = suggest_model_types(prose(records=600))
    assert entry(large, GENERATE)["ready"] is True
    assert kinds(large)[0] == GENERATE


def test_a_source_with_nothing_usable_still_names_search_first() -> None:
    options = suggest_model_types({"kind": "tree", "files": 0, "text_files": 0, "formats": {}})

    assert kinds(options)[0] == SEARCH
    assert not any(option["ready"] for option in options)


def test_every_option_says_why_and_an_unready_one_says_what_it_needs() -> None:
    options = suggest_model_types(prose())

    assert [option["rank"] for option in options] == [1, 2, 3]
    assert all(option["why"] for option in options)
    assert all(option["ready"] or option["needs"] for option in options)


@pytest.mark.parametrize(
    ("suffix", "text", "expected"),
    [
        (".csv", "ticket,subject,label\n1,vpn,network\n", ["label"]),
        (".csv", "ticket,subject\n1,vpn\n", []),
        (".tsv", "ticket\tstatus\n1\topen\n", ["status"]),
        (".jsonl", '{"text": "vpn", "intent": "network"}\n', ["intent"]),
        (".jsonl", '{"text": "vpn"}\n', []),
        (".jsonl", "not json\n", []),
        (".md", "label: network\n", []),
        (".csv", "a,b," + "c" * 3_000 + "\n", []),
    ],
)
def test_a_label_column_is_read_from_the_first_row_only(
    suffix: str, text: str, expected: list[str]
) -> None:
    assert label_columns(suffix, text) == expected


def test_the_scan_reports_the_formats_and_the_label_columns_it_saw(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "note.md").write_text("reset the vpn\n")
    (root / "tickets.csv").write_text("ticket,subject,label\n1,vpn,network\n")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))

    result = build.inspect_source(platform, str(root))

    assert result.formats == {".csv": 1, ".md": 1}
    assert result.label_columns == ["label"]
    assert kinds(suggest_model_types(result.to_dict()))[0] == DECIDE


def test_the_page_shows_what_the_data_can_carry_after_a_guided_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    for index in range(30):
        (root / f"note-{index}.txt").write_text(f"ticket {index}: reset the vpn\n")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        dana = enroll(app.state.platform, "Dana Owner", ["data_owner"])
        assert sign_in(client, dana.name).status_code == 303
        guided = client.post(
            "/console/guided",
            data={"csrf_token": session_csrf(client), "path": str(root)},
            follow_redirects=False,
        )
        assert guided.status_code == 303

        tracker = client.get(guided.headers["location"])

        assert tracker.status_code == 200
        assert "What this data can carry" in tracker.text
        assert "best fit" in tracker.text
        assert "search" in tracker.text


def test_a_refused_source_is_not_told_what_to_build(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(approved))
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "note.txt").write_text("reset the vpn\n")

    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        lead = enroll(app.state.platform, "Lena Lead", ["program_lead", "data_owner"])
        assert sign_in(client, lead.name).status_code == 303
        page = client.get("/console/sources", params={"inspect": str(outside)})
        match = TOKEN.search(page.text)
        if match:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                body = client.get(f"/console/sources/inspect/{match.group(1)}").text
                if "What the platform sees" in body:
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("inspection did not finish in time")
        else:
            body = page.text

        assert "This source would be refused" in body
        assert "What to build from this" not in body
