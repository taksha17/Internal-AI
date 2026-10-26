"""Inspection skips dependencies and the platform's own filter rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.data.filters import parse_filters
from internal_ai.data.tree import build_tree_index
from internal_ai.services import build
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import PermissionDenied

pytestmark = pytest.mark.filterwarnings("ignore:Using `httpx` with `starlette.testclient`")


def tree(root: Path) -> Path:
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "one.txt").write_text("reset the vpn from the portal\n")
    (root / "notes" / "two.md").write_text("how to file an expense\n")
    (root / "node_modules" / "react-native").mkdir(parents=True)
    (root / "node_modules" / "react-native" / "model.bin").write_bytes(b"\x00weights")
    (root / ".caches" / "hub").mkdir(parents=True)
    (root / ".caches" / "hub" / "model.bin").write_bytes(b"\x00weights")
    (root / "vendor").mkdir()
    (root / "vendor" / "bundle.bin").write_bytes(b"\x00weights")
    return root


def test_dependency_folders_are_not_reported_as_weights(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tree(tmp_path / "repo")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))
    inspection = build.inspect_source(platform, str(root))

    assert inspection.weight_files == ["vendor/bundle.bin"]
    assert inspection.files == 3
    assert "node_modules" in inspection.ignored_folders
    assert ".caches" in inspection.ignored_folders


def test_filters_drop_folders_and_file_types_from_the_preview(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tree(tmp_path / "repo")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))
    owner = enroll(platform, "Dana Owner", ["data_owner"])
    build.set_inspection_filters(platform, owner, "vendor", ".bin, .md")

    inspection = build.inspect_source(platform, str(root))
    assert inspection.acceptable, inspection.problems
    assert inspection.files == 1
    assert inspection.text_files == 1
    assert inspection.ignored_files == 1
    assert "vendor" in inspection.ignored_folders
    assert inspection.weight_files == []


def test_weight_files_are_left_out_instead_of_refused(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "osmOS"
    (root / "resources" / "llm" / "models").mkdir(parents=True)
    (root / "notes.txt").write_text("reset the vpn from the portal\n")
    (root / "resources" / "llm" / "models" / "Qwen-Q4_K_M.gguf").write_bytes(b"\x00weights")
    (root / "snapshot_blob.bin").write_bytes(b"\x00blob")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))

    preview = build.inspect_source(platform, str(root))
    assert preview.acceptable, preview.problems
    assert preview.weight_files == [
        "resources/llm/models/Qwen-Q4_K_M.gguf",
        "snapshot_blob.bin",
    ]
    assert (preview.files, preview.text_files) == (3, 1)

    index = build_tree_index(root)
    assert [item.relative_path for item in index.files] == ["notes.txt"]
    assert index.stats()["weight_files"] == [
        "resources/llm/models/Qwen-Q4_K_M.gguf",
        "snapshot_blob.bin",
    ]

    owner = enroll(platform, "Dana Owner", ["data_owner"])
    source = build.register_source(
        platform, owner, str(root), "osmos", "learn from the docs", ["engineering"]
    )
    assert source["scan"]["weight_files"] == [
        "resources/llm/models/Qwen-Q4_K_M.gguf",
        "snapshot_blob.bin",
    ]


def test_a_weight_file_on_its_own_is_still_refused(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "stories15M-q4_0.gguf"
    weights.write_bytes(b"\x00weights")
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(tmp_path))

    inspection = build.inspect_source(platform, str(weights))
    assert inspection.weight_files == [weights.name]
    assert not inspection.acceptable
    assert "pretrained weights are never used" in inspection.problems[0]


def test_a_data_owner_can_add_a_folder_for_inspection(platform, tmp_path: Path) -> None:
    outside = tmp_path / "company-docs"
    outside.mkdir()
    (outside / "handbook.md").write_text("how to file an expense\n")
    owner = enroll(platform, "Dana Owner", ["data_owner"])

    refused = build.inspect_source(platform, str(outside))
    assert not refused.acceptable
    assert "outside the approved data folders" in refused.problems[0]

    build.set_extra_data_roots(platform, owner, str(outside))
    accepted = build.inspect_source(platform, str(outside))
    assert accepted.acceptable, accepted.problems
    assert accepted.text_files == 1
    assert str(outside) in [str(item) for item in build.data_roots(platform)]

    events = platform.audit.recent(limit=5, action="data.approved_roots_changed")
    assert events and events[0].actor == owner.id
    assert json.loads(json.dumps(events[0].details))["after"] == [str(outside)]


def test_only_data_owners_may_add_a_data_folder(platform, tmp_path: Path) -> None:
    expert = enroll(platform, "Dex Expert", ["domain_expert"])
    agent = Actor(
        id="data_engineer",
        name="data_engineer",
        roles=frozenset({"data_owner", "ml_lead"}),
        groups=frozenset(),
        authenticated=True,
        kind="agent",
    )
    with pytest.raises(PermissionDenied, match="needs data_owner"):
        build.set_extra_data_roots(platform, expert, str(tmp_path))
    with pytest.raises(PermissionDenied, match="needs data_owner"):
        build.set_extra_data_roots(platform, agent, str(tmp_path))
    assert build.extra_data_roots(platform) == []


def test_a_data_folder_must_be_an_existing_absolute_folder(platform, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello\n")
    with pytest.raises(ValueError, match="absolute path"):
        build.parse_data_roots("relative/folder")
    with pytest.raises(ValueError, match="does not exist"):
        build.parse_data_roots(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="not a folder"):
        build.parse_data_roots(str(tmp_path / "notes.txt"))
    with pytest.raises(ValueError, match="too broad"):
        build.parse_data_roots("/")
    many = [tmp_path / f"folder{index}" for index in range(11)]
    for folder in many:
        folder.mkdir()
    with pytest.raises(ValueError, match="at most"):
        build.parse_data_roots(", ".join(str(folder) for folder in many))

    owner = enroll(platform, "Dana Owner", ["data_owner"])
    build.set_extra_data_roots(platform, owner, f"{tmp_path}, {tmp_path}")
    assert build.extra_data_roots(platform) == [tmp_path.resolve()]
    build.set_extra_data_roots(platform, owner, "")
    assert build.extra_data_roots(platform) == []


def test_tree_secrets_are_left_out_instead_of_refused(
    platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "notes.txt").write_text("reset the vpn from the portal\n")
    (root / "config.py").write_text('password = "supersecretvalue"\n')
    monkeypatch.setenv("INTERNAL_AI_DATA_ROOTS", str(root))

    inspection = build.inspect_source(platform, str(root))
    assert inspection.acceptable, inspection.problems
    assert inspection.secret_files == ["config.py"]
    assert inspection.secret_hits == {"password_assignment": 1}
    assert inspection.files == 2
    assert inspection.text_files == 1

    index = build_tree_index(root)
    assert [item.relative_path for item in index.files] == ["notes.txt"]
    assert index.secret_files == ["config.py"]
    assert index.stats()["secret_files"] == ["config.py"]

    owner = enroll(platform, "Dana Owner", ["data_owner"])
    source = build.register_source(
        platform, owner, str(root), "repo", "learn from resolved tickets", ["engineering"]
    )
    assert source["gate_id"]


def test_only_data_owners_may_change_the_filters(platform) -> None:
    owner = enroll(platform, "Dana Owner", ["data_owner"])
    expert = enroll(platform, "Dex Expert", ["domain_expert"])
    agent = Actor(
        id="data_engineer",
        name="data_engineer",
        roles=frozenset({"data_owner", "ml_lead"}),
        groups=frozenset(),
        authenticated=True,
        kind="agent",
    )
    with pytest.raises(PermissionDenied, match="needs data_owner"):
        build.set_inspection_filters(platform, expert, "vendor", ".bin")
    with pytest.raises(PermissionDenied, match="needs data_owner"):
        build.set_inspection_filters(platform, agent, "vendor", ".bin")
    assert build.inspection_filters(platform).folders == frozenset()

    build.set_inspection_filters(platform, owner, "vendor, .tmp", "bin")
    saved = build.inspection_filters(platform)
    assert saved.folders == frozenset({"vendor", ".tmp"})
    assert saved.suffixes == frozenset({".bin"})
    events = platform.audit.recent(limit=5, action="data.inspection_filters_changed")
    assert events and events[0].actor == owner.id
    assert json.loads(json.dumps(events[0].details))["after"]["suffixes"] == [".bin"]


def test_filter_text_is_validated() -> None:
    assert parse_filters(" vendor , .tmp ", "bin, .ONNX").folders == frozenset({"vendor", ".tmp"})
    assert parse_filters("", "bin").suffixes == frozenset({".bin"})
    with pytest.raises(ValueError, match="folder name, not a path"):
        parse_filters("deps/node_modules", "")
    with pytest.raises(ValueError, match="file extension"):
        parse_filters("", "not an extension")
    with pytest.raises(ValueError, match="at most"):
        parse_filters(", ".join(f"dir{index}" for index in range(41)), "")


def test_console_saves_filters_and_refuses_the_wrong_role(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        owner = enroll(app.state.platform, "Dana Owner", ["data_owner"])
        sign_in(client, owner.name)
        page = client.post(
            "/console/sources/filters",
            data={
                "csrf_token": csrf_from(client.get("/console/sources").text),
                "folders": "vendor",
                "suffixes": ".bin",
            },
        )
        assert "inspection filters saved" in page.text
        assert 'value="vendor"' in page.text
        assert 'value=".bin"' in page.text

        expert = enroll(app.state.platform, "Dex Expert", ["domain_expert"])
        sign_in(client, expert.name)
        refused = client.post(
            "/console/sources/filters",
            data={
                "csrf_token": csrf_from(client.get("/console/sources").text),
                "folders": "vendor",
                "suffixes": ".bin",
            },
        )
        assert "needs data_owner or ml_lead" in refused.text
        assert build.inspection_filters(app.state.platform).folders == frozenset({"vendor"})

        stale = client.post("/console/sources/filters", data={"folders": "x", "suffixes": ".y"})
        assert stale.status_code == 403


def test_console_saves_data_folders_and_refuses_the_wrong_role(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    docs = tmp_path / "company-docs"
    docs.mkdir()
    with TestClient(app) as client:
        owner = enroll(app.state.platform, "Dana Owner", ["data_owner"])
        sign_in(client, owner.name)
        page = client.post(
            "/console/sources/roots",
            data={"csrf_token": csrf_from(client.get("/console/sources").text), "roots": str(docs)},
        )
        assert "data folders saved" in page.text
        assert str(docs) in page.text
        assert str(docs) in [str(item) for item in build.data_roots(app.state.platform)]

        expert = enroll(app.state.platform, "Dex Expert", ["domain_expert"])
        sign_in(client, expert.name)
        refused = client.post(
            "/console/sources/roots",
            data={"csrf_token": csrf_from(client.get("/console/sources").text), "roots": "/tmp"},
        )
        assert "needs data_owner or ml_lead" in refused.text
        assert build.extra_data_roots(app.state.platform) == [docs.resolve()]
