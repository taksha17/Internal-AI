"""High-level workflows that connect the data plane, training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.data.pipeline import DataPipeline, PipelineConfig
from internal_ai.data.records import Record, Source
from internal_ai.data.splits import SplitResult, make_splits
from internal_ai.data.tokenizer import Tokenizer, coverage_report, train_tokenizer
from internal_ai.domain import AuditOutcome, DatasetVersion, ModelVersion
from internal_ai.ids import content_hash, new_id, now_iso
from internal_ai.training.classifier import ClassifierConfig, ClassifierOutcome
from internal_ai.training.embedder import TorchEmbedder
from internal_ai.training.trainer import TrainConfig, TrainingOutcome


@dataclass(slots=True)
class CorpusBuild:
    dataset_id: str
    content_hash: str
    records: int
    tokens: int
    stats: dict[str, Any]
    lineage: dict[str, Any]


def build_corpus(
    platform: Platform,
    source_paths: list[Path],
    name: str,
    version: int = 1,
    approved: bool = True,
    owner: str = "data_owner",
    actor: str = "data_engineer",
    config: PipelineConfig | None = None,
) -> CorpusBuild:
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
    sources = [
        Source(
            source_id=path.stem,
            path=path,
            sensitivity="internal",
            acl=["engineering"],
            approved=approved,
            owner=owner,
        )
        for path in source_paths
    ]
    pipeline_config = config or PipelineConfig(name=name, version=version)
    result = pipeline.run(sources, pipeline_config, actor=actor)
    return CorpusBuild(
        dataset_id=result.dataset.id,
        content_hash=result.dataset.content_hash,
        records=result.dataset.record_count,
        tokens=result.dataset.token_count,
        stats=result.stats,
        lineage=result.lineage,
    )


def split_corpus(platform: Platform, dataset_id: str, seed: str = "forge-v1") -> SplitResult:
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if dataset.released:
        raise PermissionError("a released dataset is immutable; build a new version to re-split")
    rows = platform.store.dataset_records(dataset_id)
    records = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            sensitivity=row["sensitivity"],
            acl=row["acl"],
            tokens=row["tokens"],
            meta=row["meta"],
        )
        for row in rows
    ]
    result = make_splits(records, seed=seed)
    assignments: dict[str, str] = {}
    for split_name, split_records in result.splits.items():
        for record in split_records:
            assignments[record.record_key] = split_name
    platform.store.set_record_splits(dataset_id, assignments)
    lineage = dict(dataset.lineage)
    lineage["splits"] = result.stats
    dataset.lineage = lineage
    dataset.stats = {**dataset.stats, "splits": result.stats}
    platform.store.conn.execute(
        "UPDATE datasets SET lineage = ?, stats = ? WHERE id = ?",
        (
            json.dumps(lineage, sort_keys=True, default=str),
            json.dumps(dataset.stats, sort_keys=True, default=str),
            dataset_id,
        ),
    )
    platform.audit.record(
        actor="data_engineer",
        action="data.splits_created",
        subject=dataset_id,
        details=result.stats,
    )
    return result


def train_tokenizer_for(
    platform: Platform,
    dataset_id: str,
    vocab_size: int = 4_000,
    jargon: list[str] | None = None,
    actor: str = "data_engineer",
) -> tuple[Tokenizer, dict[str, Any]]:
    rows = platform.store.dataset_records_by_split(dataset_id, "train")
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    if not rows:
        raise ValueError(f"dataset {dataset_id} has no records")
    texts = [row["payload"] for row in rows]
    tokenizer = train_tokenizer(texts, vocab_size=vocab_size)
    by_source: dict[str, list[str]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row["payload"])
    coverage = coverage_report(tokenizer, by_source, jargon=jargon)
    path = platform.settings.models_dir / f"tokenizer-{dataset_id}.json"
    tokenizer.save(path)
    coverage["tokenizer_path"] = str(path)
    coverage["dataset_id"] = dataset_id
    coverage["trained_on_split"] = "train"
    coverage["trained_on_documents"] = len(texts)
    platform.audit.record(
        actor=actor,
        action="tokenizer.trained",
        subject=dataset_id,
        details={"vocab_size": tokenizer.vocab_size, "coverage": coverage},
    )
    measured = DataPipeline(
        platform.store, platform.audit, platform.settings.root
    ).measure_token_count(dataset_id, tokenizer, actor=actor)
    coverage["tokens"] = measured["tokens"]
    coverage["token_count_estimate"] = measured["estimate"]
    return tokenizer, coverage


def train_embedding_model(
    platform: Platform,
    dataset_id: str,
    tokenizer: Tokenizer,
    config: TrainConfig | None = None,
    actor: str = "ml_engineer",
    split: str = "train",
    pair_mode: str = "paraphrase",
) -> TrainingOutcome:
    from internal_ai.training.trainer import Trainer

    rows = platform.store.dataset_records_by_split(dataset_id, split)
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    records = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            sensitivity=row["sensitivity"],
            acl=row["acl"],
            tokens=row["tokens"],
            meta=row["meta"],
        )
        for row in rows
    ]
    validation = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            sensitivity=row["sensitivity"],
            acl=row["acl"],
            tokens=row["tokens"],
            meta=row["meta"],
        )
        for row in platform.store.dataset_records_by_split(dataset_id, "validation")
    ]
    from internal_ai.training.ownership import require_released_dataset

    dataset = require_released_dataset(platform.store.get_dataset(dataset_id))
    trainer = Trainer(
        platform.store,
        platform.audit,
        platform.budget,
        platform.settings.artifacts_dir,
        root=platform.settings.root,
    )
    train_config = config or TrainConfig()
    pairs = build_pair_dataset(platform, dataset_id, split=split, mode=pair_mode)
    return trainer.train(
        train_config,
        records,
        tokenizer,
        dataset_id=dataset_id,
        dataset_hash=dataset.content_hash if dataset else "unknown",
        actor=actor,
        pairs=pairs,
        validation_records=validation,
    )


def evaluate_embedding_model(
    platform: Platform,
    model_path: Path,
    eval_set_id: str,
    tokenizer: Tokenizer,
    dataset_id: str,
    margin: float = 0.0,
    actor: str = "evaluation",
    model_id: str | None = None,
    judged_by: str | None = None,
) -> dict[str, Any]:
    from internal_ai.brief import load as load_brief
    from internal_ai.evals.harness import EvalHarness

    harness = EvalHarness(platform.store, platform.audit)
    eval_set = platform.store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise KeyError(f"unknown eval set: {eval_set_id}")
    if not eval_set.frozen:
        raise PermissionError(
            f"eval set {eval_set_id} is not frozen; evaluate against frozen sets only (FR-14)"
        )
    corpus_rows = platform.store.dataset_records(dataset_id)
    corpus = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            sensitivity=row["sensitivity"],
            acl=row["acl"],
            tokens=row["tokens"],
            meta=row["meta"],
        )
        for row in corpus_rows
    ]
    training_records = [
        record for record, row in zip(corpus, corpus_rows, strict=True) if row["split"] == "train"
    ]
    embedder = TorchEmbedder(model_path, tokenizer)
    outcome = harness.evaluate_retrieval(
        embedder,
        eval_set,
        corpus,
        training_records=training_records,
        margin=margin,
        actor=actor,
        brief=load_brief(platform.store),
    )
    report = harness.record_report(
        model_id or model_path.stem, eval_set_id, outcome, actor=actor, judged_by=judged_by
    )
    return {
        "report_id": report.id,
        "model_id": report.model_id,
        "passed": report.passed,
        "metrics": report.metrics,
        "baseline_metrics": report.baseline_metrics,
        "checks": report.checks,
        "notes": report.notes,
    }


def register_model(
    platform: Platform,
    name: str,
    model_path: Path,
    outcome: TrainingOutcome,
    dataset_id: str,
    intended_use: str = "internal semantic search and retrieval",
    actor: str = "ml_engineer",
) -> ModelVersion:
    existing = platform.store.list_models(name=name)
    version = max((item.version for item in existing), default=0) + 1
    dataset = platform.store.get_dataset(dataset_id)
    card = {
        "intended_use": intended_use,
        "data_sources": [
            item["source_id"] for item in (dataset.lineage.get("sources", []) if dataset else [])
        ],
        "dataset_id": dataset_id,
        "dataset_hash": dataset.content_hash if dataset else None,
        "training_config": outcome.reproducibility.get("config", {}),
        "commit": outcome.reproducibility.get("commit"),
        "seed": outcome.reproducibility.get("seed"),
        "eval_results": {},
        "known_limitations": [
            "Trained only on internal corpus v1; no public data was used.",
            "Embedding quality is bounded by corpus size and vocabulary coverage.",
            "Not evaluated on languages outside the corpus.",
        ],
        "trained_from_scratch": True,
        "pretrained_weights_used": False,
        "registered_by": actor,
        "created_at": now_iso(),
    }
    model = ModelVersion(
        id=new_id("mdl"),
        name=name,
        version=version,
        content_hash=content_hash({"path": str(model_path), "run": outcome.run_id}),
        created_at=now_iso(),
        dataset_ids=[dataset_id],
        config=outcome.reproducibility.get("config", {}),
        status="candidate",
        card=card,
        path=str(model_path),
    )
    platform.store.add_model(model)
    platform.audit.record(
        actor=actor,
        action="model.registered",
        subject=model.id,
        details={"name": name, "version": version, "run_id": outcome.run_id},
    )
    return model


def promote_model(
    platform: Platform,
    model_id: str,
    gate_approved: bool,
    approver: str = "ml_lead",
    actor: str = "evaluation",
) -> ModelVersion:
    model = platform.store.get_model(model_id)
    if model is None:
        raise KeyError(f"unknown model: {model_id}")
    if not gate_approved:
        raise PermissionError("model promotion requires an approved model_promotion gate (FR-15)")
    report = latest_frozen_pass(platform, model_id)
    if report is None:
        raise PermissionError(
            "model promotion requires a passing eval report on a frozen eval set (FR-15)"
        )
    builder = str(model.card.get("registered_by") or "")
    if report.judged_by and report.judged_by in {builder, approver}:
        raise PermissionError(
            "the newest report was judged by the person who built or is approving this model; "
            "a different ml_lead has to run that evaluation"
        )
    model.card["eval_results"] = report.metrics
    model.card["baseline_metrics"] = report.baseline_metrics
    model.card["eval_report_id"] = report.id
    model.card["promoted_by"] = approver
    model.card["promoted_at"] = now_iso()
    platform.store.update_model_card(model_id, model.card)
    platform.store.set_model_status(model_id, "promoted", report.id)
    platform.audit.record(
        actor=approver,
        actor_kind="human",
        action="model.promoted",
        subject=model_id,
        details={
            "eval_report_id": report.id,
            "eval_set_id": report.eval_set_id,
            "passed": report.passed,
        },
    )
    return platform.store.get_model(model_id)


def latest_frozen_pass(platform: Platform, model_id: str) -> Any:
    """The newest report counts only when it passed on a set that is still frozen.

    A classifier run records a report too, and its file name can be chosen to
    match a registered model. That report has no frozen eval set, so it must
    not satisfy promotion. A report that failed its safety checks must not
    satisfy it either: the human gate approves a promotion, it does not
    re-run the PII and leakage checks.
    """
    reports = platform.store.list_eval_reports(model_id)
    if not reports or not reports[-1].passed:
        return None
    report = reports[-1]
    eval_set = platform.store.get_eval_set(report.eval_set_id)
    if eval_set is None or not eval_set.frozen:
        return None
    return report


def model_card_markdown(model: ModelVersion) -> str:
    card = model.card
    lines = [
        f"# Model card: {model.name} v{model.version}",
        "",
        f"- Status: {model.status}",
        f"- Intended use: {card.get('intended_use', 'unspecified')}",
        f"- Trained from scratch: {card.get('trained_from_scratch', True)}",
        f"- Pretrained weights used: {card.get('pretrained_weights_used', False)}",
        f"- Base model: {card.get('base_repo_id', 'none')}",
        f"- Base revision: {card.get('base_revision', 'none')}",
        f"- Base license: {card.get('base_license', 'none')}",
        f"- Dataset: {card.get('dataset_id')} (hash {card.get('dataset_hash')})",
        f"- Data sources: {', '.join(card.get('data_sources', [])) or 'unknown'}",
        f"- Commit: {card.get('commit')}",
        f"- Seed: {card.get('seed')}",
        "",
        "## Evaluation results",
        "",
    ]
    for key, value in card.get("eval_results", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Known limitations", ""])
    for limitation in card.get("known_limitations", []):
        lines.append(f"- {limitation}")
    lines.extend(["", "## Intended use", "", card.get("intended_use", "unspecified"), ""])
    return "\n".join(lines)


def _paraphrase_query(text: str, variant: int = 0) -> str | None:
    """Rewrite a document into a user-style query using internal synonyms."""
    from internal_ai.data.samples import TERM_SYNONYMS

    lowered = text.lower()
    matches = [(term, synonym) for term, synonym in TERM_SYNONYMS.items() if term in lowered]
    if not matches:
        return None
    if variant == 0:
        parts = [synonym for _, synonym in matches[:1]]
        return "how do I resolve " + parts[0]
    parts = [synonym for _, synonym in matches[:2]]
    return "troubleshooting " + " and ".join(parts)


def build_pair_dataset(
    platform: Platform, dataset_id: str, split: str = "train", mode: str = "paraphrase"
) -> list[dict[str, str]]:
    """Query-document pairs for contrastive training, built from internal text.

    Mirrors the AI engineer task-data builder (PRD 7.7): queries are written
    against internal resolved tickets and documents, never generated by a
    third-party model. In paraphrase mode the query is a user-style rewrite
    built from internal synonyms; in title mode it is the document's own first
    line, the task a developer performs when they search for a module by its
    header. Title mode exists because real engineering corpora do not contain
    the support-ticket vocabulary the paraphrase builder depends on.
    """
    rows = platform.store.dataset_records_by_split(dataset_id, split)
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    pairs: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["record_key"]):
        if mode == "title":
            lines = row["payload"].strip().splitlines()
            first_line = lines[0].strip() if lines else ""
            if len(first_line) < 8:
                continue
            pairs.append(
                {
                    "query": " ".join(first_line.split()[:12]),
                    "document": row["payload"],
                    "record_key": row["record_key"],
                    "variant": 0,
                }
            )
            continue
        for variant in (0, 1):
            query = _paraphrase_query(row["payload"], variant=variant)
            if not query:
                continue
            pairs.append(
                {
                    "query": query,
                    "document": row["payload"],
                    "record_key": row["record_key"],
                    "variant": variant,
                }
            )
    return pairs


def derive_eval_items(
    platform: Platform, dataset_id: str, count: int = 24, mode: str = "paraphrase"
) -> list[dict[str, Any]]:
    """Build query-document pairs from the held-out split so relevance is grounded.

    In paraphrase mode the query is a user-style rewrite of the document. In
    title mode the query is the document's own first line, which is the task a
    developer performs when they search for a module by its header.
    """
    rows = platform.store.dataset_records_by_split(dataset_id, "test")
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    rows = sorted(rows, key=lambda row: row["record_key"])
    items: list[dict[str, Any]] = []
    for row in rows:
        if mode == "title":
            lines = row["payload"].strip().splitlines()
            first_line = lines[0].strip() if lines else ""
            if len(first_line) < 8:
                continue
            query = " ".join(first_line.split()[:12])
            items.append(
                {
                    "query": query,
                    "relevant_ids": [row["record_key"]],
                    "kind": "retrieval",
                    "source_id": row["source_id"],
                }
            )
            if len(items) >= count:
                break
            continue
        for variant in (0, 1):
            query = _paraphrase_query(row["payload"], variant=variant)
            if not query:
                continue
            items.append(
                {
                    "query": query,
                    "paraphrase": query,
                    "relevant_ids": [row["record_key"]],
                    "kind": "retrieval",
                    "source_id": row["source_id"],
                }
            )
        if len(items) >= count:
            break
    return items[:count]


def train_classifier_model(
    platform: Platform,
    dataset_id: str,
    encoder_path: Path,
    tokenizer: Tokenizer,
    config: ClassifierConfig | None = None,
    actor: str = "ml_engineer",
    split: str = "train",
) -> ClassifierOutcome:
    """Train a task classifier over frozen from-scratch encoder embeddings."""
    from internal_ai.data.labels import build_labeled_dataset
    from internal_ai.training.classifier import ClassifierTrainer
    from internal_ai.training.ownership import require_released_dataset

    require_released_dataset(platform.store.get_dataset(dataset_id))
    rows = platform.store.dataset_records_by_split(dataset_id, split)
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    records = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            sensitivity=row["sensitivity"],
            acl=row["acl"],
            tokens=row["tokens"],
            meta=row["meta"],
        )
        for row in rows
    ]
    labelled = build_labeled_dataset(records)
    baseline = [item["label"] for item in labelled]
    trainer = ClassifierTrainer(
        platform.store,
        platform.audit,
        platform.budget,
        platform.settings.artifacts_dir,
        root=platform.settings.root,
    )
    train_config = config or ClassifierConfig()
    return trainer.train(
        train_config,
        encoder_path,
        tokenizer,
        labelled,
        baseline,
        actor=actor,
    )


def evaluate_classifier_model(
    platform: Platform,
    model_path: Path,
    tokenizer: Tokenizer,
    dataset_id: str,
    split: str = "test",
    actor: str = "evaluation",
    min_accuracy: float = 0.6,
) -> dict[str, Any]:
    """Evaluate a classifier on held-out data against the manual labelling.

    The rule labels stand in for the manual process, so comparing the model
    against them would be circular. The comparison baseline is therefore the
    majority class, and the rule agreement is reported as a reference figure.
    A true rules-versus-model comparison needs independently labelled data.
    """
    from internal_ai.brief import describe_verdict
    from internal_ai.brief import load as load_brief
    from internal_ai.data.labels import build_labeled_dataset, label_counts
    from internal_ai.evals.harness import EvalHarness, brief_targets
    from internal_ai.training.classifier import predict_classifier

    rows = platform.store.dataset_records_by_split(dataset_id, split)
    if not rows:
        rows = platform.store.dataset_records(dataset_id)
    records = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            sensitivity=row["sensitivity"],
            acl=row["acl"],
            tokens=row["tokens"],
            meta=row["meta"],
        )
        for row in rows
    ]
    labelled = build_labeled_dataset(records)
    if not labelled:
        raise ValueError(f"dataset {dataset_id} has no labelled records")
    predictions = predict_classifier(model_path, tokenizer, [item["text"] for item in labelled])
    truth = [item["label"] for item in labelled]
    counts = label_counts(labelled)
    majority_label = max(counts, key=lambda label: (counts[label], label))
    majority_baseline = [majority_label] * len(truth)
    harness = EvalHarness(platform.store, platform.audit)
    outcome = harness.evaluate_classifier(predictions, truth, actor=actor)
    baseline_outcome = harness.evaluate_classifier(majority_baseline, truth, actor=actor)
    per_class = _per_class_report(predictions, truth, sorted(counts))
    baseline_per_class = _per_class_report(majority_baseline, truth, sorted(counts))
    model_accuracy = outcome.metrics["accuracy"]
    baseline_accuracy = baseline_outcome.metrics["accuracy"]
    beat_baseline = model_accuracy > baseline_accuracy
    enough_accuracy = model_accuracy >= min_accuracy
    decision = beat_baseline and enough_accuracy
    outcome.passed = decision
    outcome.baseline_metrics = baseline_outcome.metrics
    outcome.notes = (
        "classifier beats the majority-class baseline and clears the accuracy floor"
        if decision
        else "classifier does not beat the majority-class baseline or misses the floor"
    )
    targets = brief_targets(load_brief(platform.store), outcome.metrics)
    if targets is not None:
        outcome.checks["brief_targets"] = targets
        if not targets["passed"]:
            decision = False
            outcome.passed = False
            outcome.notes += "; short of the brief: " + "; ".join(
                describe_verdict(verdict) for verdict in targets["targets"] if not verdict["met"]
            )
    report = harness.record_report(
        f"classifier:{model_path.stem}",
        f"classifier::{dataset_id}",
        outcome,
        actor=actor,
    )
    return {
        "report_id": report.id,
        "passed": decision,
        "accuracy": model_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "baseline_description": "majority class baseline",
        "rule_agreement": model_accuracy,
        "rule_agreement_note": (
            "the rule labels are the reference labelling, so rule agreement equals "
            "accuracy; a true rules-versus-model comparison needs independent labels"
        ),
        "min_accuracy": min_accuracy,
        "per_class": per_class,
        "baseline_per_class": baseline_per_class,
        "label_distribution": counts,
        "brief_targets": targets,
        "notes": outcome.notes,
    }


def _per_class_report(
    predictions: list[str], truth: list[str], labels: list[str]
) -> dict[str, Any]:
    from internal_ai.evals.metrics import precision_recall_f1

    return {label: precision_recall_f1(predictions, truth, positive=label) for label in labels}


def build_tree_corpus(
    platform: Platform,
    index: Any,
    name: str,
    version: int = 1,
    acl: list[str] | None = None,
    actor: str = "data_engineer",
    sensitivity: str = "internal",
) -> CorpusBuild:
    """Version a file tree as a dataset with path metadata and lineage."""

    records = index.records(acl=acl or ["engineering", "support"], sensitivity=sensitivity)
    for record in records:
        record.sensitivity = sensitivity
    dataset = DatasetVersion(
        id=new_id("ds"),
        name=name,
        version=version,
        content_hash=content_hash(
            {"files": sorted(item.content_hash for item in index.files), "name": name}
        ),
        record_count=len(records),
        token_count=sum(record.tokens for record in records),
        created_at=now_iso(),
        lineage={
            "sources": [
                {
                    "source_id": index.source_id,
                    "path": str(index.root),
                    "content_hash": content_hash(sorted(item.content_hash for item in index.files)),
                    "owner": "data_owner",
                    "sensitivity": sensitivity,
                    "records_ingested": len(records),
                    "approved": True,
                }
            ],
            "steps": [{"step": "ingest_tree", **index.stats()}],
            "config": {"name": name, "version": version, "acl": acl or []},
            "config_hash": content_hash({"name": name, "version": version}),
            "code_commit": _git_commit(platform.settings.root),
            "created_at": now_iso(),
        },
        stats={"steps": [{"step": "ingest_tree", **index.stats()}]},
        released=False,
    )
    platform.store.add_dataset(dataset)
    platform.store.add_dataset_records(dataset.id, [record.to_row() for record in records])
    platform.audit.record(
        actor=actor,
        action="dataset.version_created",
        subject=dataset.id,
        outcome=AuditOutcome.SUCCESS,
        details={
            "name": name,
            "version": version,
            "content_hash": dataset.content_hash,
            "records": dataset.record_count,
            "tokens": dataset.token_count,
            "root": str(index.root),
            "lineage_coverage": "100%",
        },
    )
    return CorpusBuild(
        dataset_id=dataset.id,
        content_hash=dataset.content_hash,
        records=dataset.record_count,
        tokens=dataset.token_count,
        stats={"steps": dataset.stats["steps"]},
        lineage=dataset.lineage,
    )


def _git_commit(root: Path) -> str:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
