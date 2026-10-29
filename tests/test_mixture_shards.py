"""A dataset records its folder mixture, an ablation, and shard files the trainer names."""

from __future__ import annotations

import json
from pathlib import Path

from internal_ai.data.mixture import mixture_rows, write_shards
from internal_ai.data.records import Record
from internal_ai.data.tokenizer import train_tokenizer
from internal_ai.domain import DatasetVersion
from internal_ai.training.trainer import TrainConfig, Trainer
from internal_ai.workflows import run_mixture_ablation


def test_two_folders_store_ratios_that_sum_to_one() -> None:
    records = [
        Record(id="1", record_key="billing/a", text="invoice total", source_id="s", tokens=10),
        Record(id="2", record_key="billing/b", text="credit note", source_id="s", tokens=10),
        Record(id="3", record_key="support/c", text="vpn reset", source_id="s", tokens=20),
    ]
    rows = mixture_rows(records)
    assert [row["folder"] for row in rows] == ["billing", "support"]
    assert abs(sum(row["ratio"] for row in rows) - 1) <= 0.001


def test_an_ablation_stores_two_losses(platform) -> None:
    dataset_id = "ds_mix"
    platform.store.add_dataset(
        DatasetVersion(
            id=dataset_id,
            name="mix",
            version=1,
            content_hash="h",
            record_count=4,
            token_count=4,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    platform.store.add_dataset_records(
        dataset_id,
        [
            {
                "record_key": "billing/one",
                "payload": "the invoice total is wrong and needs a credit note today",
            },
            {
                "record_key": "billing/two",
                "payload": "export the billing file before the quarter closes on friday",
            },
            {
                "record_key": "support/one",
                "payload": "reset the vpn profile before you sign in on monday morning",
            },
            {
                "record_key": "support/two",
                "payload": "restart the worker if the queue stops moving overnight",
            },
        ],
    )
    results = run_mixture_ablation(platform, dataset_id)
    assert len(results) == 2
    assert all(isinstance(item["validation_loss"], float) for item in results)
    stored = platform.store.get_dataset(dataset_id)
    assert stored is not None and len(stored.stats["ablations"]) == 2


def test_the_trainer_names_the_first_shard(platform, tmp_path) -> None:
    texts = [
        "reset the vpn profile before you sign in again today",
        "the billing export runs every friday at six",
        "open a ticket when the invoice total looks wrong",
        "restart the worker if the queue stops moving",
    ]
    tokenizer = train_tokenizer(texts, vocab_size=80)
    rows = [{"record_key": f"k{index}", "payload": text} for index, text in enumerate(texts)]
    directory = tmp_path / "shards"
    write_shards(rows[:2], tokenizer, directory)
    # A second call would overwrite shard-00000. Write the rest beside it.
    extra = directory / "shard-00001.jsonl"
    with extra.open("w", encoding="utf-8") as handle:
        for row in rows[2:]:
            ids = tokenizer.content_ids(tokenizer.encode(row["payload"]))
            handle.write(json.dumps({"ids": ids, "record_key": row["record_key"]}) + "\n")
    trainer = Trainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    )
    outcome = trainer.train(
        TrainConfig(
            steps=2,
            batch_size=2,
            seq_len=16,
            max_seq_len=16,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
            log_interval=1,
            checkpoint_interval=2,
            objective="causal",
            seed=17,
            vocab_rationale="shard test",
        ),
        [
            Record(id="1", record_key="k0", text=texts[0], source_id="s"),
            Record(id="2", record_key="k1", text=texts[1], source_id="s"),
        ],
        tokenizer,
        dataset_id="ds_shards",
        dataset_hash="h",
        resume=False,
        shards_dir=directory,
    )
    first = json.loads(Path(outcome.metrics_path).read_text(encoding="utf-8").splitlines()[0])
    assert first["data_offset"]["shard"] == "shard-00000.jsonl"
