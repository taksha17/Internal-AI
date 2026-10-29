"""A held-out batch is logged during training and the checkpoint can reproduce it."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from internal_ai.console.app import loss_points
from internal_ai.data.records import Record
from internal_ai.data.tokenizer import train_tokenizer
from internal_ai.training.model import TransformerConfig
from internal_ai.training.trainer import TrainConfig, Trainer

TEXTS = [
    "the vpn client keeps dropping the tunnel on the home network",
    "the invoice total is wrong and needs a credit note raised today",
    "restart the worker if the queue stops moving overnight",
    "the support rota is published on monday morning",
]


def _records(texts: list[str]) -> list[Record]:
    return [
        Record(
            id=str(index), record_key=str(index), text=text, source_id="src", acl=["engineering"]
        )
        for index, text in enumerate(texts)
    ]


def _config(**overrides: object) -> TrainConfig:
    values = {
        "name": "val-tiny",
        "steps": 4,
        "batch_size": 2,
        "seq_len": 32,
        "max_seq_len": 32,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 64,
        "dropout": 0.0,
        "log_interval": 1,
        "checkpoint_interval": 4,
        "objective": "masked",
        "seed": 17,
    }
    values.update(overrides)
    return TrainConfig(**values)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_a_validation_split_is_logged_on_more_than_one_row(platform, tmp_path: Path) -> None:
    tokenizer = train_tokenizer(TEXTS * 2, vocab_size=200)
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
    outcome = trainer.train(
        _config(),
        _records(TEXTS),
        tokenizer,
        dataset_id="ds_val",
        dataset_hash="h",
        validation_records=_records(TEXTS[:2]),
        resume=False,
    )

    rows = [
        row
        for row in _rows(Path(outcome.metrics_path))
        if isinstance(row.get("validation_loss"), float)
    ]
    assert len(rows) > 1
    assert outcome.checkpoint_reloaded is True


def test_a_dataset_without_a_validation_split_says_so(platform, tmp_path: Path) -> None:
    tokenizer = train_tokenizer(TEXTS, vocab_size=200)
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
    outcome = trainer.train(
        _config(steps=2),
        _records(TEXTS),
        tokenizer,
        dataset_id="ds_noval",
        dataset_hash="h",
        resume=False,
    )

    rows = _rows(Path(outcome.metrics_path))
    assert rows
    assert all(row["validation_loss"] == "not_applicable" for row in rows)
    assert outcome.checkpoint_reloaded is False


def test_a_rising_validation_loss_stops_the_run(platform, tmp_path, monkeypatch) -> None:
    tokenizer = train_tokenizer(TEXTS, vocab_size=200)
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
    seen = {"n": 0}

    def rising(self, model, batches, config):
        seen["n"] += 1
        return float(seen["n"])

    monkeypatch.setattr(Trainer, "_validation_loss", rising)
    outcome = trainer.train(
        _config(steps=6),
        _records(TEXTS),
        tokenizer,
        dataset_id="ds_rise",
        dataset_hash="h",
        validation_records=_records(TEXTS[:2]),
        resume=False,
    )

    assert outcome.status == "diverged"
    assert any("rose for three logs at step" in alert for alert in outcome.alerts)


def test_corrupted_checkpoint_weights_do_not_match_the_log(platform, tmp_path: Path) -> None:
    tokenizer = train_tokenizer(TEXTS * 2, vocab_size=200)
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
    outcome = trainer.train(
        _config(steps=2),
        _records(TEXTS),
        tokenizer,
        dataset_id="ds_corrupt",
        dataset_hash="h",
        validation_records=_records(TEXTS[:2]),
        resume=False,
    )
    assert outcome.checkpoint_reloaded is True
    path = Path(outcome.checkpoint_path)
    state = torch.load(path, weights_only=True)
    with torch.no_grad():
        state["model_state"]["norm.weight"].add_(3)
    torch.save(state, path)
    config = _config(steps=2)
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    )
    logged = next(
        row["validation_loss"]
        for row in reversed(_rows(Path(outcome.metrics_path)))
        if isinstance(row.get("validation_loss"), float)
    )
    sequences = trainer.encode_records(_records(TEXTS[:2]), tokenizer, config.seq_len)
    assert (
        trainer._logged_checkpoint_matches(
            path, logged, [sequences[: config.batch_size]], config, model_config
        )
        is False
    )


def test_the_model_page_chart_draws_both_series(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "loss": 1.0, "validation_loss": 1.2}),
                json.dumps({"step": 1, "loss": 0.8, "validation_loss": 0.9}),
            ]
        ),
        encoding="utf-8",
    )

    chart = str(loss_points(str(path)))

    assert chart.count("<polyline") == 2
