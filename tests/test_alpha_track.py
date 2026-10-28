"""Alpha may load one pinned base. The from-scratch track may not."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from internal_ai.domain import DatasetVersion, GateType, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.training.alpha.card import alpha_card
from internal_ai.training.alpha.citations import AskHit, score_answer
from internal_ai.training.alpha.pin import FALLBACK, PRIMARY, RECIPE
from internal_ai.training.alpha.promote import register_alpha, request_alpha_promotion, serve_alpha
from internal_ai.training.alpha.stream import lora_step, stream_lora_file
from internal_ai.workflows import model_card_markdown

ROOT = Path(__file__).resolve().parents[1] / "src" / "internal_ai"
HITS = [
    AskHit("Acmy_Website/next.config.ts", 36),
    AskHit("Acmy_Website/src/app/api/payments/razorpay/create-order/route.ts", 8),
    AskHit("Acmy_Website/docs/PRD.md", 16),
]


def dataset(platform, dataset_id: str, released: bool) -> None:
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="company",
            version=1,
            content_hash=f"hash-{dataset_id}",
            record_count=1,
            token_count=1,
            created_at=now_iso(),
            released=released,
        )
    )


def test_only_the_alpha_package_names_a_published_loader() -> None:
    banned = ("from_pretrained", "AutoModel", "snapshot_download", "safe_open", "from safetensors")
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if "training/alpha" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}:{token}")
    assert offenders == []


def test_recipe_matches_the_card_that_trained_on_this_gpu() -> None:
    assert RECIPE.rank == 8
    assert RECIPE.alpha == 16
    assert RECIPE.max_seq_len == 64
    assert RECIPE.quantization is None
    assert RECIPE.stream_from_disk is True
    assert PRIMARY.repo_id == "Qwen/Qwen3-4B"
    assert PRIMARY.license == "Apache-2.0"
    assert FALLBACK.repo_id == "Qwen/Qwen2.5-3B"


def test_alpha_card_names_the_base_and_scratch_card_does_not() -> None:
    card = alpha_card(PRIMARY, RECIPE, "ds_1", "hash", "abc")
    model = ModelVersion(
        id="mdl_alpha",
        name="qwen3-4b",
        version=1,
        content_hash="h",
        created_at=now_iso(),
        card=card,
    )
    text = model_card_markdown(model)
    assert "Pretrained weights used: True" in text
    assert "Qwen/Qwen3-4B" in text
    assert PRIMARY.revision in text
    scratch = ModelVersion(
        id="mdl_scratch",
        name="new-volume-3",
        version=1,
        content_hash="h",
        created_at=now_iso(),
        card={"trained_from_scratch": True, "pretrained_weights_used": False},
    )
    scratch_text = model_card_markdown(scratch)
    assert "Pretrained weights used: False" in scratch_text
    assert "Base model: none" in scratch_text


def test_a_line_number_cannot_move_onto_another_file() -> None:
    mixed = (
        "The script is in Acmy_Website/src/app/api/payments/razorpay/"
        "create-order/route.ts at line 36."
    )
    report = score_answer(mixed, HITS)
    assert report.passed is False
    assert any("next.config.ts" in item for item in report.mismatches)
    clean = "The host is allowed in Acmy_Website/next.config.ts at line 36."
    assert score_answer(clean, HITS).passed is True


def test_a_citation_to_a_file_that_was_never_a_hit_is_refused() -> None:
    hits = [AskHit("app.ts", 10)]
    report = score_answer("See src/app.ts at line 10.", hits)
    assert report.passed is False
    assert any("src/app.ts" in item for item in report.mismatches)


def test_a_hit_path_that_prefixes_another_hit_is_not_a_false_failure() -> None:
    hits = [AskHit("a/b.ts", 5), AskHit("a/b.ts.bak", 7)]
    report = score_answer("See a/b.ts.bak at line 7.", hits)
    assert report.passed is True, report.mismatches


def test_streamed_lora_step_uses_rank_8_and_drops_the_matrix(tmp_path: Path) -> None:
    path = tmp_path / "layer.pt"
    torch.save({"w": torch.ones(6, 4)}, path)
    adapter = stream_lora_file(path, "w")
    assert adapter["rank"] == 8
    assert adapter["max_seq_len"] == 64
    assert adapter["quantization"] is None
    assert adapter["A"].shape == (8, 4)
    assert adapter["B"].shape == (6, 8)
    with pytest.raises(ValueError, match="4-bit"):
        lora_step(torch.ones(2, 2), type(RECIPE)(quantization="4bit"))


def test_the_streamed_step_learns_a_rank_8_correction() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 4)
    adapter = lora_step(weight, RECIPE, steps=100)
    assert adapter["A"].abs().sum() > 0, "the adapter never left its zero initialisation"
    assert adapter["B"].abs().sum() > 0, "the adapter never learned anything"
    assert adapter["loss_last"] < adapter["loss_first"]


def test_alpha_promotion_is_requested_and_serving_waits(
    platform, enroll_person, tmp_path: Path
) -> None:
    person = enroll_person(platform, "Elena Alpha", ["ml_lead"])
    dataset(platform, "ds_alpha", released=True)
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    model = register_alpha(platform, PRIMARY, adapter, "ds_alpha", person)
    assert model.card["pretrained_weights_used"] is True
    gate = request_alpha_promotion(
        platform,
        person,
        model.id,
        "See Acmy_Website/next.config.ts at line 36.",
        HITS,
    )
    assert gate.status.value == "pending"
    assert gate.requested_by == person.id
    with pytest.raises(PermissionError, match="approved"):
        serve_alpha(platform, model.id, tmp_path / "runner")
    stored = platform.store.get_gate(gate.id)
    assert stored is not None
    stored.status = type(stored.status)("approved")
    platform.store.update_gate(stored)
    manifest = serve_alpha(platform, model.id, tmp_path / "runner")
    body = manifest.read_text(encoding="utf-8")
    assert model.card["adapter_sha256"] in body
    assert "local-runner" in body
    assert str(GateType.MODEL_PROMOTION) == "model_promotion"


def test_registering_an_alpha_model_needs_the_register_model_capability(
    platform, enroll_person, tmp_path: Path
) -> None:
    from internal_ai.services.permissions import PermissionDenied

    dataset(platform, "ds_alpha", released=True)
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    outsider = enroll_person(platform, "Rae Outsider", ["data_engineer"])
    with pytest.raises(PermissionDenied):
        register_alpha(platform, PRIMARY, adapter, "ds_alpha", outsider)
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    model = register_alpha(platform, PRIMARY, adapter, "ds_alpha", lead)
    assert model.card["pretrained_weights_used"] is True
    assert model.status == "candidate"


def test_an_alpha_model_may_name_only_the_pinned_base(
    platform, enroll_person, tmp_path: Path
) -> None:
    from dataclasses import replace

    dataset(platform, "ds_alpha", released=True)
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    forged = replace(PRIMARY, repo_id="Qwen/Qwen3-4B-finetuned", revision="0" * 40)
    with pytest.raises(PermissionError, match="pinned"):
        register_alpha(platform, forged, adapter, "ds_alpha", lead)


def test_an_alpha_model_needs_a_released_dataset(platform, enroll_person, tmp_path: Path) -> None:
    dataset(platform, "ds_draft", released=False)
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    lead = enroll_person(platform, "Mo Lead", ["ml_lead"])
    with pytest.raises(PermissionError, match="released"):
        register_alpha(platform, PRIMARY, adapter, "ds_draft", lead)
