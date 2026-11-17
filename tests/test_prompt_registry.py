"""The prompt id a model carries.

Training, evaluation and serving share one function, so they cannot disagree. What
they could still do is serve a model that was trained in an older wording, which is
why the text itself is hashed rather than a version label trusted.
"""

from __future__ import annotations

from internal_ai.training.alpha import pairs


def test_the_fingerprint_is_stable_and_hex() -> None:
    first = pairs.prompt_fingerprint()

    assert first == pairs.prompt_fingerprint()
    assert len(first) == 64
    assert int(first, 16) >= 0


def test_the_fingerprint_follows_the_text_not_the_label(monkeypatch) -> None:
    before = pairs.prompt_fingerprint()
    monkeypatch.setattr(pairs, "INSTRUCTION", pairs.INSTRUCTION + " Be brief.")

    assert pairs.prompt_fingerprint() != before, (
        "an edit that forgets to bump PROMPT_VERSION has to change the id anyway"
    )


def test_a_different_candidate_layout_is_a_different_prompt() -> None:
    selection = pairs.prompt_for_candidates("q", [("k", "text")])

    assert "[k] text" in selection
    assert selection != pairs.prompt_for("q")


def test_the_card_records_the_prompt_that_made_the_pairs(platform, enroll_person, tmp_path) -> None:
    import json

    import torch

    from internal_ai.training.alpha.pin import PRIMARY
    from internal_ai.training.alpha.promote import register_alpha

    person = enroll_person(platform, "Elena Alpha", ["ml_lead"])
    from internal_ai.domain import DatasetVersion

    platform.store.add_dataset(
        DatasetVersion(
            id="ds_prompt",
            name="alpha",
            version=1,
            content_hash="h",
            record_count=10,
            token_count=100,
            created_at="2026-01-01T00:00:00+00:00",
            released=True,
        )
    )
    adapter = tmp_path / "adapter.pt"
    torch.save({"A": torch.zeros(1)}, adapter)
    (tmp_path / "reproducibility.json").write_text(json.dumps({}), encoding="utf-8")

    model = register_alpha(platform, PRIMARY, adapter, "ds_prompt", person)

    assert model.card["prompt"]["id"] == pairs.prompt_fingerprint()
    assert model.card["prompt"]["version"] == pairs.PROMPT_VERSION
