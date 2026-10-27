"""Task-specific classifier trained on frozen from-scratch embeddings.

The encoder is trained once on the internal corpus; the classifier is a linear
head over its pooled embeddings. Both parts are trained from scratch on
internal data, and the rules-based labeller is the baseline to beat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.budget import BudgetManager
from internal_ai.data.tokenizer import Tokenizer
from internal_ai.domain import AuditOutcome
from internal_ai.ids import new_id, now_iso
from internal_ai.store import Store

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required to train classifiers") from exc

from internal_ai.training.model import PAD_ID, TinyTransformer, TransformerConfig


@dataclass(slots=True)
class ClassifierConfig:
    name: str = "router-v1"
    epochs: int = 40
    learning_rate: float = 0.05
    seed: int = 23
    cost_cap_usd: float = 2.0
    max_seq_len: int = 256

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "cost_cap_usd": self.cost_cap_usd,
            "max_seq_len": self.max_seq_len,
        }


@dataclass(slots=True)
class ClassifierOutcome:
    run_id: str
    status: str
    epochs: int
    final_loss: float
    accuracy: float
    baseline_accuracy: float
    model_path: str
    labels: list[str] = field(default_factory=list)
    class_report: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "model_path": self.model_path,
            "labels": self.labels,
            "class_report": self.class_report,
            "alerts": self.alerts,
        }


class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.linear(vectors)


class ClassifierTrainer:
    """Trains a linear head over frozen encoder embeddings."""

    def __init__(
        self,
        store: Store,
        audit: AuditLog,
        budget: BudgetManager,
        artifacts_dir: Path,
        root: Path | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.budget = budget
        self.artifacts_dir = artifacts_dir
        self.root = root or Path.cwd()

    def _embed(
        self,
        model: TinyTransformer,
        tokenizer: Tokenizer,
        texts: list[str],
        max_seq_len: int,
    ) -> torch.Tensor:
        encoded = [list(tokenizer.encode(text)[: max_seq_len - 1]) or [1] for text in texts]
        width = max(len(row) for row in encoded)
        padded = [row + [PAD_ID] * (width - len(row)) for row in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long)
        mask = (input_ids != PAD_ID).unsqueeze(-1)
        with torch.no_grad():
            return model.embed(input_ids, mask=mask)

    def train(
        self,
        config: ClassifierConfig,
        encoder_path: Path,
        tokenizer: Tokenizer,
        labelled: list[dict[str, Any]],
        baseline_predictions: list[str],
        actor: str = "ml_engineer",
    ) -> ClassifierOutcome:
        run_id = new_id("cls")
        run_dir = self.artifacts_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        labels = sorted({item["label"] for item in labelled})
        label_index = {label: index for index, label in enumerate(labels)}
        if len(labels) < 2:
            raise ValueError("a classifier needs at least two classes")

        payload = torch.load(encoder_path, weights_only=True, map_location="cpu")
        encoder_config = TransformerConfig.from_dict(payload["model_config"])
        encoder = TinyTransformer(encoder_config)
        encoder.load_state_dict(payload["model_state"])
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

        torch.manual_seed(config.seed)
        features = self._embed(
            encoder, tokenizer, [item["text"] for item in labelled], config.max_seq_len
        )
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        features = (features - mean) / std
        targets = torch.tensor([label_index[item["label"]] for item in labelled], dtype=torch.long)
        weights = self._class_weights(targets, len(labels))
        head = LinearHead(features.size(1), len(labels))
        optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)
        alerts: list[str] = []
        loss_value = float("nan")
        for epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = F.cross_entropy(logits, targets, weight=weights)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
            if not torch.isfinite(torch.tensor(loss_value)):
                alerts.append(f"divergence detected at epoch {epoch}")
                break

        with torch.no_grad():
            predictions = [labels[index] for index in head(features).argmax(dim=1).tolist()]
        truth = [item["label"] for item in labelled]
        accuracy = sum(
            1 for prediction, label in zip(predictions, truth, strict=False) if prediction == label
        ) / len(truth)
        baseline_accuracy = (
            sum(
                1
                for prediction, label in zip(baseline_predictions, truth, strict=False)
                if prediction == label
            )
            / len(truth)
            if truth
            else 0.0
        )
        class_report = self._class_report(predictions, truth, labels)
        model_path = self.artifacts_dir / "models" / config.name / f"{run_id}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": encoder.state_dict(),
                "encoder_config": encoder_config.to_dict(),
                "head_state": head.state_dict(),
                "labels": labels,
                "config": config.to_dict(),
                "feature_mean": mean.squeeze(0),
                "feature_std": std.squeeze(0),
            },
            model_path,
        )
        spend = self._spend(config, len(labelled))
        self.budget.record_spend(spend, agent=actor, task_id=None, notes=f"classifier:{run_id}")
        self.audit.record(
            actor=actor,
            action="classifier.trained",
            subject=run_id,
            outcome=AuditOutcome.SUCCESS if not alerts else AuditOutcome.FAILURE,
            details={
                "labels": labels,
                "accuracy": accuracy,
                "baseline_accuracy": baseline_accuracy,
                "alerts": alerts,
                "model_path": str(model_path),
            },
            cost_usd=spend,
        )
        return ClassifierOutcome(
            run_id=run_id,
            status="diverged" if alerts else "completed",
            epochs=config.epochs,
            final_loss=loss_value,
            accuracy=accuracy,
            baseline_accuracy=baseline_accuracy,
            model_path=str(model_path),
            labels=labels,
            class_report=class_report,
            alerts=alerts,
        )

    @staticmethod
    def _standardise(features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (features - mean) / std

    @staticmethod
    def _class_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        counts = torch.bincount(targets, minlength=num_classes).float()
        inverse = counts.clamp(min=1.0).reciprocal()
        weights = inverse / inverse.mean()
        return torch.where(counts > 0, weights, torch.zeros_like(weights))

    @staticmethod
    def _class_report(
        predictions: list[str], truth: list[str], labels: list[str]
    ) -> dict[str, Any]:
        from internal_ai.evals.metrics import precision_recall_f1

        report: dict[str, Any] = {}
        for label in labels:
            report[label] = precision_recall_f1(predictions, truth, positive=label)
        return report

    @staticmethod
    def _spend(config: ClassifierConfig, samples: int) -> float:
        return round(min(samples * config.epochs * 1e-6, config.cost_cap_usd), 6)


def predict_classifier(model_path: Path, tokenizer: Tokenizer, texts: list[str]) -> list[str]:
    """Load a trained classifier and score texts."""
    payload = torch.load(model_path, weights_only=True, map_location="cpu")
    encoder_config = TransformerConfig.from_dict(payload["encoder_config"])
    encoder = TinyTransformer(encoder_config)
    encoder.load_state_dict(payload["encoder_state"])
    encoder.eval()
    labels: list[str] = payload["labels"]
    head = LinearHead(encoder_config.d_model, len(labels))
    head.load_state_dict(payload["head_state"])
    head.eval()
    encoded = [list(tokenizer.encode(text)[:255]) or [1] for text in texts]
    width = max(len(row) for row in encoded)
    padded = [row + [PAD_ID] * (width - len(row)) for row in encoded]
    input_ids = torch.tensor(padded, dtype=torch.long)
    mask = (input_ids != PAD_ID).unsqueeze(-1)
    with torch.no_grad():
        vectors = encoder.embed(input_ids, mask=mask)
        mean = payload.get("feature_mean")
        std = payload.get("feature_std")
        if mean is not None and std is not None:
            vectors = (vectors - mean.unsqueeze(0)) / std.unsqueeze(0).clamp(min=1e-6)
        logits = head(vectors)
    return [labels[index] for index in logits.argmax(dim=1).tolist()]


def classifier_metadata(model_path: Path) -> dict[str, Any]:
    payload = torch.load(model_path, weights_only=True, map_location="cpu")
    return {
        "labels": payload["labels"],
        "config": payload["config"],
        "created_at": now_iso(),
        "trained_from_scratch": True,
        "pretrained_weights_used": False,
    }


def save_metadata(model_path: Path, metadata: dict[str, Any]) -> Path:
    path = model_path.with_suffix(".json")
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path
