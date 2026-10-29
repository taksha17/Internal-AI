"""Tests for training subset selection based on advice marks."""

from __future__ import annotations

from pathlib import Path

from internal_ai.data.records import Record
from internal_ai.data.tokenizer import Tokenizer, train_tokenizer
from internal_ai.training.trainer import (
    TrainConfig,
    Trainer,
    _boilerplate_ratio,
    _calculate_record_quality_score,
    select_training_subset,
)


def _make_record(text: str, language: str = "en", record_id: str = "r1") -> Record:
    """Helper to create a test record."""
    return Record(
        id=record_id,
        record_key=record_id,
        text=text,
        source_id="test_source",
        sensitivity="internal",
        acl=["engineering"],
        tokens=len(text.split()),
        meta={"language": language},
    )


def _make_tokenizer() -> Tokenizer:
    """Create a simple tokenizer for testing."""
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
        "test": 6,
        "good": 7,
        "bad": 8,
        "the": 9,
        "and": 10,
    }
    return Tokenizer(vocab=vocab, merges=[])


def test_boilerplate_ratio_detects_markers() -> None:
    """Boilerplate ratio should be higher for text with legal markers."""
    boilerplate_text = (
        "Copyright 2026. All rights reserved. Terms of service apply. Privacy policy."
    )
    clean_text = "This is a normal document with regular content about engineering topics."

    bp_score = _boilerplate_ratio(boilerplate_text)
    clean_score = _boilerplate_ratio(clean_text)

    assert bp_score > clean_score
    assert bp_score >= 0.8  # High boilerplate
    assert clean_score == 0.0


def test_quality_score_prefers_english_clean_text() -> None:
    """English, clean, well-sized text should score higher than boilerplate."""
    config = TrainConfig()
    tokenizer = _make_tokenizer()

    good_record = _make_record(
        "This is a well written technical document about software engineering and system design patterns that developers use daily.",
        language="en",
        record_id="good",
    )
    bad_record = _make_record(
        "Copyright. All rights reserved. Terms of service. Privacy policy. Unsubscribe here. Click here.",
        language="unknown",
        record_id="bad",
    )

    good_score = _calculate_record_quality_score(good_record, tokenizer, 128, config)
    bad_score = _calculate_record_quality_score(bad_record, tokenizer, 128, config)

    assert good_score > bad_score
    assert good_score > 0.8  # High quality
    assert bad_score < 0.7  # Lower quality due to boilerplate


def test_subset_selection_no_limit_returns_all() -> None:
    """When train_subset_size is None, all records should be returned."""
    config = TrainConfig(train_subset_size=None)
    tokenizer = _make_tokenizer()
    records = [
        _make_record(f"Document number {i} with some content.", record_id=f"r{i}")
        for i in range(10)
    ]

    selected, stats = select_training_subset(records, config, tokenizer, 128)

    assert len(selected) == 10
    assert stats["subset_used"] == "all"
    assert stats["selected_count"] == 10


def test_subset_selection_quality_strategy_selects_top_n() -> None:
    """Quality strategy should select the top N highest-scoring records."""
    config = TrainConfig(train_subset_size=3, train_subset_strategy="quality")
    tokenizer = _make_tokenizer()

    records = [
        _make_record(
            "Copyright all rights reserved terms of service.", language="unknown", record_id="bad1"
        ),
        _make_record(
            "This is excellent technical content about machine learning systems.",
            language="en",
            record_id="good1",
        ),
        _make_record(
            "Another great document about distributed systems and databases.",
            language="en",
            record_id="good2",
        ),
        _make_record(
            "Privacy policy unsubscribe click here.", language="unknown", record_id="bad2"
        ),
        _make_record(
            "One more high quality engineering document about cloud infrastructure.",
            language="en",
            record_id="good3",
        ),
    ]

    selected, stats = select_training_subset(records, config, tokenizer, 128)

    assert len(selected) == 3
    assert stats["selection_method"] == "quality"
    assert stats["selected_count"] == 3
    assert stats["total_available"] == 5
    # All selected should be English high-quality
    assert all(r.meta.get("language") == "en" for r in selected)


def test_subset_selection_random_strategy() -> None:
    """Random strategy should select random subset of requested size."""
    config = TrainConfig(train_subset_size=4, train_subset_strategy="random", seed=42)
    tokenizer = _make_tokenizer()

    records = [_make_record(f"Document {i} content here.", record_id=f"r{i}") for i in range(10)]

    selected, stats = select_training_subset(records, config, tokenizer, 128)

    assert len(selected) == 4
    assert stats["selection_method"] == "random"
    # Random selection with seed should be deterministic
    selected2, _ = select_training_subset(records, config, tokenizer, 128)
    assert [r.id for r in selected] == [r.id for r in selected2]


def test_subset_selection_threshold_strategy() -> None:
    """Threshold strategy should include records meeting minimum quality threshold."""
    config = TrainConfig(
        train_subset_size=100,  # Large limit so threshold is the binding constraint
        train_subset_strategy="threshold",
        train_subset_threshold=0.7,  # Reasonable threshold based on new scoring
    )
    tokenizer = _make_tokenizer()

    records = [
        _make_record(
            "High quality technical documentation about software architecture and design patterns for developers.",
            language="en",
            record_id="hq1",
        ),
        _make_record(
            "Another excellent document about machine learning and neural networks in production systems.",
            language="en",
            record_id="hq2",
        ),
        _make_record(
            "Copyright. All rights reserved. Terms of service.", language="unknown", record_id="lq1"
        ),
        _make_record("Short text.", language="unknown", record_id="lq2"),
    ]

    selected, stats = select_training_subset(records, config, tokenizer, 128)

    assert stats["selection_method"] == "threshold"
    # Should select only the high-quality ones (those above 0.7 threshold)
    assert len(selected) >= 2
    assert all(r.meta.get("language") == "en" for r in selected)


def test_subset_size_larger_than_available() -> None:
    """Requesting more records than available should return all available."""
    config = TrainConfig(train_subset_size=100, train_subset_strategy="quality")
    tokenizer = _make_tokenizer()

    records = [_make_record(f"Document {i}.", record_id=f"r{i}") for i in range(5)]

    selected, stats = select_training_subset(records, config, tokenizer, 128)

    assert len(selected) == 5  # Only 5 available
    assert stats["selected_count"] == 5
    assert stats["total_available"] == 5


def test_quality_score_handles_empty_text() -> None:
    """Quality scoring should handle edge cases gracefully."""
    config = TrainConfig()
    tokenizer = _make_tokenizer()

    empty_record = _make_record("", record_id="empty")
    score = _calculate_record_quality_score(empty_record, tokenizer, 128, config)

    # Empty text should get a low but valid score
    assert 0.0 <= score <= 1.0


def test_train_config_subset_parameters() -> None:
    """TrainConfig should properly serialize and deserialize subset parameters."""
    config = TrainConfig(
        train_subset_size=500,
        train_subset_strategy="random",
        train_subset_threshold=0.75,
        min_quality_score=0.6,
    )

    config_dict = config.to_dict()
    assert config_dict["train_subset_size"] == 500
    assert config_dict["train_subset_strategy"] == "random"
    assert config_dict["train_subset_threshold"] == 0.75
    assert config_dict["min_quality_score"] == 0.6

    # Round-trip
    restored = TrainConfig.from_dict(config_dict)
    assert restored.train_subset_size == 500
    assert restored.train_subset_strategy == "random"


def _run_config() -> TrainConfig:
    return TrainConfig(
        name="subset-tiny",
        steps=1,
        batch_size=2,
        seq_len=32,
        max_seq_len=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        log_interval=1,
        checkpoint_interval=1,
        objective="masked",
        seed=17,
        train_subset_size=2,
        train_subset_strategy="quality",
    )


def test_a_real_run_trains_on_the_selected_subset_only(platform, tmp_path, monkeypatch) -> None:
    """The subset is chosen before encoding, so validation keeps every record."""
    good = [
        "the vpn client keeps dropping the tunnel on the home network",
        "the invoice total is wrong and needs a credit note raised today",
    ]
    boilerplate = "Copyright 2026 sample. All rights reserved. Terms of service apply."
    records = [_make_record(text, record_id=f"good-{index}") for index, text in enumerate(good)]
    records += [_make_record(boilerplate, record_id=f"junk-{index}") for index in range(6)]
    validation = [_make_record("check the shard count before the nightly build", record_id="val")]

    tokenizer = train_tokenizer([item.text for item in records + validation], vocab_size=300)
    trainer = Trainer(
        platform.store,
        platform.audit,
        platform.budget,
        tmp_path / "artifacts",
        root=tmp_path,
    )

    seen: list[list[str]] = []
    original = Trainer.encode_records

    def spy(self, rows, token, seq_len):
        seen.append([row.record_key for row in rows])
        return original(self, rows, token, seq_len)

    monkeypatch.setattr(Trainer, "encode_records", spy)
    outcome = trainer.train(
        _run_config(),
        records,
        tokenizer,
        dataset_id="ds_subset",
        dataset_hash="h",
        resume=False,
        validation_records=validation,
    )

    assert outcome.steps == 1
    assert outcome.subset["selected_count"] == 2
    assert outcome.subset["total_available"] == 8
    assert seen[0] == ["good-0", "good-1"]
    assert seen[1] == ["val"], "validation is not part of the training subset"
    events = platform.store.list_audit(action="train.subset_selected")
    assert events
    assert events[-1].details["selected_count"] == 2
    assert events[-1].details["total_available"] == 8
    assert events[-1].details["selection_method"] == "quality"


def test_the_outcome_and_card_record_how_many_records_trained(platform, tmp_path) -> None:
    from internal_ai.domain import DatasetVersion
    from internal_ai.ids import now_iso
    from internal_ai.training.trainer import TrainingOutcome
    from internal_ai.workflows import register_model

    platform.store.add_dataset(
        DatasetVersion(
            id="ds_card",
            name="tickets",
            version=1,
            content_hash="abc",
            record_count=8,
            token_count=10_000,
            created_at=now_iso(),
            released=True,
            stats={},
        )
    )
    outcome = TrainingOutcome(
        run_id="run_subset",
        status="completed",
        steps=2,
        tokens_seen=64,
        final_loss=1.0,
        cost_usd=0.0,
        checkpoint_path="",
        metrics_path="",
        model_path=str(tmp_path / "m.pt"),
        subset={
            "subset_used": "selected",
            "selected_count": 3,
            "total_available": 8,
            "selection_method": "quality",
            "average_score": 0.9,
        },
    )
    model = register_model(
        platform,
        "subset-card",
        Path(outcome.model_path),
        outcome,
        "ds_card",
        actor="ml_engineer",
    )
    assert model.card["training_records_used"] == 3
    assert model.card["training_subset"]["total_available"] == 8


def test_the_cli_and_console_accept_a_subset() -> None:
    from internal_ai.cli import build_parser

    args = build_parser().parse_args(
        [
            "train",
            "run",
            "--dataset-id",
            "ds",
            "--tokenizer",
            "tok.json",
            "--subset-size",
            "500",
            "--subset-strategy",
            "threshold",
            "--subset-threshold",
            "0.7",
        ]
    )
    assert args.subset_size == 500
    assert args.subset_strategy == "threshold"
    assert args.subset_threshold == 0.7
    template = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "internal_ai"
        / "console"
        / "templates"
        / "models.html"
    ).read_text(encoding="utf-8")
    assert 'name="subset_size"' in template
    assert "training_subset.selected_count" in template
