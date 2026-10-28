"""The model is checked against its spec before it is trained.

A mis-shaped layer or a NaN in the initial weights is cheap to catch and
expensive to discover from a finished run, so shapes and finiteness are checked
rather than logged, and a first loss that is not finite is named as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from internal_ai.data.records import Record
from internal_ai.data.tokenizer import Tokenizer, train_tokenizer
from internal_ai.training import trainer as trainer_module
from internal_ai.training.model import (
    TinyTransformer,
    TransformerConfig,
    non_finite_parameters,
    spec_violations,
)
from internal_ai.training.trainer import TrainConfig, Trainer

TEXTS = [
    "the vpn client keeps dropping the tunnel on the home network",
    "the invoice total is wrong and needs a credit note raised today",
]


def spec(**overrides: int) -> TransformerConfig:
    values = {
        "vocab_size": 200,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "d_ff": 64,
        "max_seq_len": 64,
        "dropout": 0.0,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_a_model_built_from_its_spec_has_no_violations() -> None:
    assert spec_violations(TinyTransformer(spec()), spec()) == []
    assert non_finite_parameters(TinyTransformer(spec())) == []


def test_a_shape_that_disagrees_with_the_spec_is_named() -> None:
    built = TinyTransformer(spec(vocab_size=200))
    wanted = spec(vocab_size=300)

    violations = spec_violations(built, wanted)

    assert any(
        "token_embedding.weight is 200.32 but the spec says 300.32" in line for line in violations
    )


def test_a_layer_count_that_disagrees_with_the_spec_is_named() -> None:
    violations = spec_violations(TinyTransformer(spec(n_layers=1)), spec(n_layers=3))

    assert violations == ["the model has 1 attention layers; the spec says 3"]


def test_a_nan_weight_is_found_before_training_starts() -> None:
    model = TinyTransformer(spec())
    with __import__("torch").no_grad():
        model.token_embedding.weight[0, 0] = float("nan")

    assert non_finite_parameters(model) == ["token_embedding.weight"]


def build_trainer(platform, tmp_path: Path) -> tuple[Trainer, Tokenizer, list[Record]]:
    tokenizer = train_tokenizer(TEXTS * 4, vocab_size=200)
    records = [
        Record(
            id=str(index), record_key=str(index), text=text, source_id="src", acl=["engineering"]
        )
        for index, text in enumerate(TEXTS * 4)
    ]
    return (
        Trainer(
            platform.store,
            platform.audit,
            platform.budget,
            tmp_path / "artifacts",
            root=tmp_path,
        ),
        tokenizer,
        records,
    )


def config(**overrides: object) -> TrainConfig:
    values = {
        "steps": 2,
        "batch_size": 2,
        "seq_len": 32,
        "max_seq_len": 32,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 64,
        "log_interval": 1,
        "checkpoint_interval": 1,
        "objective": "masked",
    }
    values.update(overrides)
    return TrainConfig(**values)


def test_a_mis_wired_model_is_refused_and_audited(platform, tmp_path, monkeypatch) -> None:
    trainer, tokenizer, records = build_trainer(platform, tmp_path)
    poisoned = TinyTransformer(spec())
    with __import__("torch").no_grad():
        poisoned.norm.weight[0] = float("inf")
    monkeypatch.setattr(trainer_module, "build_model", lambda config, seed=0: poisoned)

    outcome = trainer.train(
        config(), records, tokenizer, dataset_id="ds_spec", dataset_hash="h", resume=False
    )

    assert outcome.status == "blocked_by_spec"
    assert outcome.steps == 0
    assert any("non-finite weights: norm.weight" in alert for alert in outcome.alerts)
    event = platform.store.list_audit(action="train.model_spec_failed")[0]
    assert event.details["non_finite"] == ["norm.weight"]
    assert not platform.store.list_audit(action="train.run_started")


def test_a_first_loss_that_is_not_finite_is_named(platform, tmp_path, monkeypatch) -> None:
    trainer, tokenizer, records = build_trainer(platform, tmp_path)

    class NaNModel(TinyTransformer):
        def forward(self, input_ids, labels=None):  # type: ignore[override]
            logits, loss = super().forward(input_ids, labels)
            if loss is not None:
                return logits, loss * float("nan")
            return logits, loss

    monkeypatch.setattr(trainer_module, "build_model", lambda config, seed=0: NaNModel(config))

    outcome = trainer.train(
        config(), records, tokenizer, dataset_id="ds_nan", dataset_hash="h", resume=False
    )

    assert outcome.status == "diverged"
    assert outcome.steps == 0
    assert any("the first loss is not finite" in alert for alert in outcome.alerts)
    assert np.isnan(outcome.final_loss)


def test_a_healthy_run_still_trains_after_the_check(platform, tmp_path) -> None:
    trainer, tokenizer, records = build_trainer(platform, tmp_path)

    outcome = trainer.train(
        config(), records, tokenizer, dataset_id="ds_ok", dataset_hash="h", resume=False
    )

    assert outcome.status == "completed", outcome.alerts
    assert outcome.steps == 2
    assert not platform.store.list_audit(action="train.model_spec_failed")
