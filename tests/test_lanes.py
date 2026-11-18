"""A device label selects its lane, and a label the table does not list is refused."""

from __future__ import annotations

import pytest
from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.compute.lanes import LaneRefused, lane_for
from internal_ai.compute.scheduler import Job, JobSpec, KubernetesBackend
from internal_ai.services.compute_advice import select_lane
from internal_ai.services.permissions import PermissionDenied


def _job(label: str = "") -> Job:
    return Job(
        id="job_lane",
        spec=JobSpec(
            name="nightly",
            command=["python", "-c", "print(1)"],
            agent="infra_engineer",
            device="gpu",
            device_label=label,
        ),
        created_at="now",
    )


def test_each_vendor_has_its_own_resource_and_an_unknown_label_is_refused() -> None:
    assert lane_for("nvidia.com/gpu").collective == "nccl"
    assert lane_for("amd.com/gpu").resource == "amd.com/gpu"
    assert lane_for("gpu.intel.com/i915").image == "internal-ai-train:xpu"
    with pytest.raises(LaneRefused, match="unknown device label"):
        lane_for("mystery.gpu")


def test_a_named_lane_is_what_the_manifest_requests() -> None:
    backend = KubernetesBackend(namespace="internal-ai", image="python:3.12-slim")
    unnamed = backend.render_job(_job())
    assert 'nvidia.com/gpu: "1"' in unnamed
    assert "image: python:3.12-slim" in unnamed
    named = backend.render_job(_job("amd.com/gpu"))
    assert 'amd.com/gpu: "1"' in named
    assert "nvidia.com/gpu" not in named
    assert "image: internal-ai-train:rocm" in named


def test_finance_cannot_select_a_lane(platform) -> None:
    infra = enroll(platform, "Ivy Infra", ["infra_engineer"])
    books = enroll(platform, "Finn Books", ["finance"])
    assert select_lane(platform, infra, "amd.com/gpu") == "amd.com/gpu"
    events = platform.store.list_audit(action="compute.lane_selected")
    assert events[0].details["collective"] == "rccl"
    with pytest.raises(PermissionDenied):
        select_lane(platform, books, "nvidia.com/gpu")


def test_the_compute_page_lists_the_lanes_and_records_a_choice(tmp_path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted")
    with TestClient(app) as client:
        enroll(app.state.platform, "Ivy Infra", ["infra_engineer"])
        assert sign_in(client, "Ivy Infra").status_code == 303
        page = client.get("/console/compute")
        assert "amd.com/gpu" in page.text
        assert "gpu.intel.com/i915" in page.text
        chosen = client.post(
            "/console/compute/lane",
            data={"label": "gpu.intel.com/i915", "csrf_token": csrf_from(page.text)},
        )
        assert chosen.status_code == 200
        assert "lane gpu.intel.com/i915 selected" in chosen.text
