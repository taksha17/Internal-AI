"""A stopped run continues on the next example, and matches a run that never stopped."""

from __future__ import annotations

import json
from pathlib import Path

from internal_ai.data.records import Record
from internal_ai.data.tokenizer import Tokenizer, train_tokenizer
from internal_ai.training.trainer import TrainConfig, Trainer

TEXTS = [
    "the vpn client keeps dropping the tunnel on the home network",
    "the invoice total is wrong and needs a credit note raised today",
    "restart the worker if the queue stops moving overnight",
    "the support rota is published on monday morning",
    "rotate the api key before the quarter ends on friday",
    "export the audit log before the retention window closes",
    "check the shard count before you start the nightly build",
    "open a ticket when the invoice total looks wrong again",
]


def _trainer(platform, root: Path) -> tuple[Trainer, Tokenizer, list[Record]]:
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
    return trainer, tokenizer, records


def _config(**overrides: object) -> TrainConfig:
    values = {
        "name": "resume-tiny",
        "steps": 8,
        "batch_size": 2,
        "seq_len": 32,
        "max_seq_len": 32,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 64,
        "dropout": 0.0,
        "log_interval": 1,
        "checkpoint_interval": 1,
        "objective": "masked",
        "seed": 17,
    }
    values.update(overrides)
    return TrainConfig(**values)


def _losses(path: Path) -> dict[int, dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return {int(row["step"]): row for row in rows}


def test_a_stopped_run_resumes_on_the_next_row_and_matches_a_continuous_run(
    platform, tmp_path: Path
) -> None:
    stopped, tokenizer, records = _trainer(platform, tmp_path / "stopped")
    stopped.train(
        _config(steps=4),
        records,
        tokenizer,
        dataset_id="ds_resume",
        dataset_hash="h",
        run_id="run_resume",
        resume=False,
    )
    outcome = stopped.train(
        _config(steps=8),
        records,
        tokenizer,
        dataset_id="ds_resume",
        dataset_hash="h",
        run_id="run_resume",
        resume=True,
    )
    control, _, _ = _trainer(platform, tmp_path / "control")
    control.train(
        _config(steps=8),
        records,
        tokenizer,
        dataset_id="ds_resume",
        dataset_hash="h",
        run_id="run_control",
        resume=False,
    )

    assert outcome.resumed_from == 4
    assert outcome.steps == 8
    left = _losses(tmp_path / "stopped" / "artifacts" / "runs" / "run_resume" / "metrics.jsonl")
    right = _losses(tmp_path / "control" / "artifacts" / "runs" / "run_control" / "metrics.jsonl")
    assert left[4]["data_offset"]["index"] != left[3]["data_offset"]["index"]
    assert left[4]["data_offset"]["dataset_id"] == "ds_resume"
    assert left[4]["data_offset"]["split"] == "train"
    assert abs(left[4]["loss"] - right[4]["loss"]) <= 1e-5
