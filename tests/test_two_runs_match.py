"""Two runs with the same seed and the same corpus produce the same loss."""

from __future__ import annotations

import json
from pathlib import Path

from internal_ai.data.records import Record
from internal_ai.data.tokenizer import train_tokenizer
from internal_ai.training.trainer import TrainConfig, Trainer

TEXTS = [
    "the vpn client keeps dropping the tunnel on the home network",
    "the invoice total is wrong and needs a credit note raised today",
    "restart the worker if the queue stops moving overnight",
    "the support rota is published on monday morning",
]


def _run(platform, root: Path, seed: int) -> list[float]:
    tokenizer = train_tokenizer(TEXTS, vocab_size=200)
    records = [
        Record(
            id=str(index), record_key=str(index), text=text, source_id="src", acl=["engineering"]
        )
        for index, text in enumerate(TEXTS)
    ]
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, root / "artifacts", root=root
    )
    outcome = trainer.train(
        TrainConfig(
            name="match-tiny",
            steps=6,
            batch_size=2,
            seq_len=32,
            max_seq_len=32,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
            log_interval=1,
            checkpoint_interval=6,
            objective="masked",
            seed=seed,
        ),
        records,
        tokenizer,
        dataset_id="ds_match",
        dataset_hash="h",
        resume=False,
    )
    rows = [
        json.loads(line)
        for line in Path(outcome.metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [float(row["loss"]) for row in rows]


def test_two_runs_with_the_same_seed_match(platform, tmp_path: Path) -> None:
    left = _run(platform, tmp_path / "left", seed=17)
    right = _run(platform, tmp_path / "right", seed=17)

    assert len(left) == 6
    assert all(abs(a - b) <= 1e-5 for a, b in zip(left, right, strict=True))
