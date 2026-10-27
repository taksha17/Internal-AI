"""Source inspection runs in the background and reports a real percentage."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.services import build

pytestmark = pytest.mark.filterwarnings("ignore:Using `httpx` with `starlette.testclient`")

TOKEN = re.compile(r"/console/sources/inspect/([A-Za-z0-9_-]+)")


def corpus(root: Path, files: int = 25) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(files):
        (root / f"note-{index}.txt").write_text(f"ticket {index}: reset the vpn\n")
    return root


def wait_for_result(client: TestClient, token: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    body = ""
    while time.monotonic() < deadline:
        body = client.get(f"/console/sources/inspect/{token}").text
        if "What the platform sees" in body:
            return body
        time.sleep(0.02)
    raise AssertionError(f"inspection did not finish in time: {body[:300]}")


def test_inspection_reports_files_read_as_it_goes(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = corpus(tmp_path / "data")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))
    seen: list[tuple[int, str]] = []
    result = build.inspect_source(platform, str(root), progress=lambda p, d: seen.append((p, d)))

    assert result.files == 25
    assert result.text_files == 25
    assert result.problems == []
    percentages = [percent for percent, _ in seen]
    assert percentages == sorted(percentages)
    assert seen[0] == (0, "listing files")
    assert seen[-1] == (100, "done")
    assert max(percentages) == 100
    assert any("of 25 files read" in detail for _, detail in seen)
    assert any(0 < percent < 90 for percent in percentages)


def test_a_path_outside_the_approved_folders_is_refused_before_it_is_read(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(approved))
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    secret = outside / "note.txt"
    secret.write_text("api_key: 'sk-live-not-a-real-token'\n")
    seen: list[tuple[int, str]] = []

    result = build.inspect_source(platform, str(outside), progress=lambda p, d: seen.append((p, d)))

    assert result.files == 0
    assert result.secret_hits == {}
    assert result.problems == [
        "the path is outside the approved data folders: " + str(approved.resolve())
    ]
    assert seen == [(100, "done")]
    assert secret.read_text().startswith("api_key")


def test_a_link_inside_an_approved_folder_cannot_read_outside_it(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("api_key: 'sk-live-not-a-real-token'\n")
    link = approved / "notes.txt"
    link.symlink_to(secret)
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(approved))

    result = build.inspect_source(platform, str(approved))

    assert result.secret_hits == {}
    assert any("link points outside" in problem for problem in result.problems)
    assert not result.acceptable


def test_a_weight_file_renamed_as_text_is_refused(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    (approved / "notes.txt").write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(approved))

    result = build.inspect_source(platform, str(approved))

    assert any("weight" in problem for problem in result.problems)
    assert not result.acceptable


def test_sensitive_files_are_left_out_and_the_rest_remains(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from internal_ai.data.tree import build_tree_index

    approved = tmp_path / "approved"
    (approved / "tickets").mkdir(parents=True)
    (approved / "tickets" / "note.txt").write_text("reset the vpn\n")
    fhir = approved / "fhir notes"
    fhir.mkdir()
    (fhir / "note.txt").write_text("patient chart\n")
    (approved / "model.pt").write_bytes(b"not-weights")
    (approved / "keys.txt").write_text("api_key: 'sk-live-not-a-real-token'\n")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(approved))

    result = build.inspect_source(platform, str(approved))

    assert result.acceptable
    assert result.text_files == 1
    assert result.secret_hits
    assert "model.pt" in result.weight_files
    indexed = build_tree_index(approved, source_id="src")
    assert [item.text for item in indexed.files] == ["reset the vpn\n"]


def test_health_and_message_folders_are_excluded(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "approved"
    folder = approved / "fhir notes"
    folder.mkdir(parents=True)
    (folder / "note.txt").write_text("ticket 1\n")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(approved))

    result = build.inspect_source(platform, str(folder))

    assert any("fhir" in problem for problem in result.problems)
    assert not result.acceptable


def test_console_inspects_without_blocking_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = corpus(tmp_path / "data", files=40)
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        unsigned = client.get(
            "/console/sources/inspect", params={"inspect": str(root)}, follow_redirects=False
        )
        assert unsigned.status_code == 303
        assert (
            client.get("/console/sources/inspect/unknown", follow_redirects=False).status_code
            == 303
        )

        lead = enroll(app.state.platform, "Lena Lead", ["program_lead", "data_owner"])
        sign_in(client, lead.name)

        page = client.get("/console/sources")
        assert 'hx-get="/console/sources/inspect"' in page.text
        assert 'hx-target="#inspect-live"' in page.text

        started = client.get("/console/sources/inspect", params={"inspect": str(root)})
        assert started.status_code == 200
        assert "<html" not in started.text.lower()

        match = TOKEN.search(started.text)
        if match:
            assert "<progress" in started.text
            assert "inspect-live" in started.text
            assert "What the platform sees" not in started.text
            finished = wait_for_result(client, match.group(1))
        else:
            finished = started.text
        assert "What the platform sees" in finished
        assert "of 25 files read" not in finished
        assert str(root) in finished

        assert client.get("/console/sources/inspect/unknown").status_code == 404


def test_the_plain_link_still_starts_an_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = corpus(tmp_path / "data", files=5)
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        lead = enroll(app.state.platform, "Lena Lead", ["program_lead", "data_owner"])
        sign_in(client, lead.name)
        page = client.get("/console/sources", params={"inspect": str(root)})
        assert page.status_code == 200
        assert "inspect-live" in page.text
        assert "hx-get" in page.text
        match = TOKEN.search(page.text)
        if match:
            assert "What the platform sees" in wait_for_result(client, match.group(1))
        else:
            assert "What the platform sees" in page.text
