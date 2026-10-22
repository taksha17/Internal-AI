"""Training loop with checkpointing, metric streaming, cost caps and resume.

Satisfies FR-10 to FR-13: reproducible runs, resumable checkpoints, streamed
metrics with divergence alerts, and a hard cost cap enforced during the run.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llmforge.audit import AuditLog
from llmforge.budget import BudgetManager
from llmforge.data.records import Record
from llmforge.data.tokenizer import Tokenizer
from llmforge.domain import AuditOutcome
from llmforge.ids import content_hash, new_id, now_iso
from llmforge.store import Store
from llmforge.training.cost import CostEstimate, estimate_cpu_run, estimate_gpu_run
from llmforge.training.model import PAD_ID, TinyTransformer, TransformerConfig, build_model

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for training") from exc


@dataclass(slots=True)
class TrainConfig:
    name: str = "embedding-v1"
    objective: str = "contrastive"
    steps: int = 100
    batch_size: int = 8
    seq_len: int = 128
    learning_rate: float = 3e-4
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mask_probability: float = 0.15
    temperature: float = 0.05
    seed: int = 17
    checkpoint_interval: int = 25
    log_interval: int = 5
    cost_cap_usd: float = 5.0
    device: str = "cpu"
    gpu_hour_usd: float = 2.0
    cpu_hour_usd: float = 0.35
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        known = {key: payload[key] for key in cls.__slots__ if key in payload}
        return cls(**known)


@dataclass(slots=True)
class TrainingOutcome:
    run_id: str
    status: str
    steps: int
    tokens_seen: int
    final_loss: float
    cost_usd: float
    checkpoint_path: str
    metrics_path: str
    model_path: str
    alerts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimate: dict[str, Any] = field(default_factory=dict)
    reproducibility: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit(root: Path) -> str:
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


class Trainer:
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

    def encode_records(
        self, records: list[Record], tokenizer: Tokenizer, seq_len: int
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        for record in records:
            ids = tokenizer.encode(record.text, max_tokens=seq_len - 1)
            if len(ids) < 8:
                continue
            sequences.append(ids)
        return sequences

    def build_batches(
        self, sequences: list[list[int]], batch_size: int, seed: int
    ) -> list[list[list[int]]]:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(sequences), generator=generator).tolist()
        shuffled = [sequences[index] for index in order]
        batches = []
        for start in range(0, len(shuffled), batch_size):
            chunk = shuffled[start : start + batch_size]
            if chunk:
                batches.append(chunk)
        return batches

    def _pair_batches(
        self, pairs: list[dict[str, str]], tokenizer: Tokenizer, config: TrainConfig
    ) -> list[list[tuple[list[int], list[int]]]]:
        encoded: list[tuple[list[int], list[int]]] = []
        for pair in pairs:
            anchor = tokenizer.encode(pair["query"])[: max(config.seq_len // 2, 8)]
            positive = tokenizer.encode(pair["document"])[: config.seq_len]
            if len(anchor) < 3 or len(positive) < 8:
                continue
            encoded.append((anchor, positive))
        if not encoded:
            return []
        generator = torch.Generator().manual_seed(config.seed)
        order = torch.randperm(len(encoded), generator=generator).tolist()
        shuffled = [encoded[index] for index in order]
        return [
            shuffled[start : start + config.batch_size]
            for start in range(0, len(shuffled), config.batch_size)
            if shuffled[start : start + config.batch_size]
        ]

    def _make_batch(
        self, batch: list[Any], config: TrainConfig
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch and isinstance(batch[0], tuple):
            anchors = [pair[0] for pair in batch]
            positives = [pair[1] for pair in batch]
            width = max(len(row) for row in anchors + positives)
            padded_anchors = [row + [PAD_ID] * (width - len(row)) for row in anchors]
            padded_positives = [row + [PAD_ID] * (width - len(row)) for row in positives]
            anchor_ids = torch.tensor(padded_anchors, dtype=torch.long)
            positive_ids = torch.tensor(padded_positives, dtype=torch.long)
            return anchor_ids, positive_ids, anchor_ids
        if config.objective == "contrastive":
            return self._make_contrastive_batch(batch, config)
        width = max(len(row) for row in batch)
        padded = [row + [PAD_ID] * (width - len(row)) for row in batch]
        input_ids = torch.tensor(padded, dtype=torch.long)
        if config.objective == "causal":
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            return inputs, targets, input_ids
        masked = input_ids.clone()
        labels = input_ids.clone()
        probability = torch.full(labels.shape, config.mask_probability)
        mask = torch.bernoulli(probability).bool()
        mask = mask & (input_ids != PAD_ID)
        replace = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & mask
        masked[replace] = 1
        random_tokens = torch.randint(
            low=2, high=config.max_seq_len, size=input_ids.shape, dtype=torch.long
        )
        masked[mask & ~replace] = random_tokens[mask & ~replace]
        labels[~mask] = -100
        return masked, labels, input_ids

    def _make_contrastive_batch(
        self, batch: list[list[int]], config: TrainConfig
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Anchor and positive pairs with in-batch negatives (InfoNCE)."""
        anchors = [row[: max(len(row) // 3, 4)] for row in batch]
        positives = [row[max(len(row) // 3, 4) :] for row in batch]
        width = max(len(row) for row in anchors + positives)
        padded_anchors = [row + [PAD_ID] * (width - len(row)) for row in anchors]
        padded_positives = [row + [PAD_ID] * (width - len(row)) for row in positives]
        anchor_ids = torch.tensor(padded_anchors, dtype=torch.long)
        positive_ids = torch.tensor(padded_positives, dtype=torch.long)
        return anchor_ids, positive_ids, anchor_ids

    def train(
        self,
        config: TrainConfig,
        records: list[Record],
        tokenizer: Tokenizer,
        dataset_id: str,
        dataset_hash: str,
        actor: str = "ml_engineer",
        resume: bool = True,
        pairs: list[dict[str, str]] | None = None,
    ) -> TrainingOutcome:
        run_id = new_id("run")
        run_dir = self.artifacts_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.jsonl"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.artifacts_dir / "models" / config.name
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        os.environ["PYTHONHASHSEED"] = str(config.seed)

        sequences = self.encode_records(records, tokenizer, config.seq_len)
        if not sequences and not pairs:
            raise ValueError("no usable sequences in the dataset")
        batches = self.build_batches(sequences, config.batch_size, config.seed)
        if pairs and config.objective == "contrastive":
            batches = self._pair_batches(pairs, tokenizer, config)
        tokens_per_step = sum(
            len(part)
            for batch in batches
            for item in batch
            for part in (item if isinstance(item, tuple) else (item,))
        ) / max(len(batches), 1)
        total_tokens = int(tokens_per_step * config.steps)

        model_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            causal=config.objective == "causal",
        )
        model = build_model(model_config, seed=config.seed)
        parameters = model.parameter_count()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        estimate = self._estimate(config, parameters, total_tokens)
        reproducibility = {
            "commit": _git_commit(self.root),
            "config": config.to_dict(),
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "tokenizer_hash": content_hash(tokenizer.vocab),
            "seed": config.seed,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
        }
        (run_dir / "reproducibility.json").write_text(
            json.dumps(reproducibility, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        self.audit.record(
            actor=actor,
            action="train.run_started",
            subject=run_id,
            outcome=AuditOutcome.REQUESTED,
            details={
                "config": config.to_dict(),
                "estimate": estimate.to_dict(),
                "parameters": parameters,
                "dataset_id": dataset_id,
            },
        )
        if estimate.estimated_cost_usd > config.cost_cap_usd:
            self.audit.record(
                actor=actor,
                action="train.cost_cap_exceeded",
                subject=run_id,
                outcome=AuditOutcome.DENIED,
                details={
                    "estimate_usd": estimate.estimated_cost_usd,
                    "cap_usd": config.cost_cap_usd,
                },
            )
            return TrainingOutcome(
                run_id=run_id,
                status="blocked_by_cost_cap",
                steps=0,
                tokens_seen=0,
                final_loss=float("nan"),
                cost_usd=0.0,
                checkpoint_path="",
                metrics_path=str(metrics_path),
                model_path="",
                alerts=[
                    f"estimated cost {estimate.estimated_cost_usd:.4f} exceeds "
                    f"cap {config.cost_cap_usd:.4f}"
                ],
                estimate=estimate.to_dict(),
                reproducibility=reproducibility,
            )

        start_step = 0
        last_checkpoint = checkpoint_dir / "latest.pt"
        if resume and last_checkpoint.exists():
            state = torch.load(last_checkpoint, weights_only=False)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            start_step = int(state["step"])
            self.audit.record(
                actor=actor,
                action="train.resumed",
                subject=run_id,
                details={"from_step": start_step},
            )

        model.train()
        alerts: list[str] = []
        warnings: list[str] = []
        loss_history: list[float] = []
        ema_loss: float | None = None
        spike_streak = 0
        tokens_seen = 0
        start_time = time.monotonic()
        step = start_step
        final_loss = float("nan")
        step_costs: list[float] = []

        while step < config.steps:
            batch_index = step % len(batches)
            batch = batches[batch_index]
            inputs, targets, _ = self._make_batch(batch, config)
            lr = self._lr(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if config.objective == "contrastive":
                loss = self._contrastive_loss(model, inputs, targets, config)
            else:
                _, loss = model(inputs, targets)
            if not torch.isfinite(loss):
                alerts.append(f"divergence detected: non-finite loss at step {step}")
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            loss_value = float(loss.detach())
            loss_history.append(loss_value)
            ema_loss = loss_value if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_value
            tokens_seen += int(inputs.numel())
            elapsed = max(time.monotonic() - start_time, 1e-6)
            throughput = tokens_seen / elapsed
            final_loss = loss_value

            if len(loss_history) > 10 and ema_loss is not None:
                if loss_value > 2 * ema_loss + 0.1:
                    spike_streak += 1
                    if spike_streak == 1:
                        warnings.append(
                            f"loss spike at step {step}: {loss_value:.4f} vs ema {ema_loss:.4f}"
                        )
                    if spike_streak >= 3:
                        alerts.append(
                            f"persistent divergence at step {step}: {spike_streak} consecutive "
                            f"spikes, loss {loss_value:.4f} vs ema {ema_loss:.4f}"
                        )
                        break
                else:
                    spike_streak = 0
            if (
                config.objective == "contrastive"
                and step > config.steps // 2
                and loss_value < 0.05
                and len(loss_history) > 20
            ):
                warnings.append(
                    f"loss collapse at step {step}: {loss_value:.4f}; the model may be "
                    "memorising pairs instead of generalising"
                )
            step_cost = self._step_cost(config, tokens_per_step, elapsed, step + 1)
            step_costs.append(step_cost)
            if sum(step_costs) > config.cost_cap_usd:
                alerts.append(
                    f"cost cap {config.cost_cap_usd:.4f} USD reached at step {step}; run stopped"
                )
                break

            if step % config.log_interval == 0:
                self._write_metric(
                    metrics_path,
                    {
                        "step": step,
                        "loss": round(loss_value, 6),
                        "ema_loss": round(ema_loss or 0.0, 6),
                        "grad_norm": round(float(grad_norm), 6),
                        "learning_rate": lr,
                        "tokens_per_second": round(throughput, 2),
                        "tokens_seen": tokens_seen,
                        "cost_usd": round(sum(step_costs), 6),
                        "timestamp": now_iso(),
                    },
                )
            if (step + 1) % config.checkpoint_interval == 0 or step + 1 == config.steps:
                self._checkpoint(checkpoint_dir, model, optimizer, step + 1, config, dataset_hash)
            step += 1

        checkpoint_path = str(checkpoint_dir / "latest.pt")
        self._checkpoint(checkpoint_dir, model, optimizer, step, config, dataset_hash)
        model_path = model_dir / f"{run_id}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": model_config.to_dict(),
                "tokenizer_vocab_size": tokenizer.vocab_size,
            },
            model_path,
        )
        spend = sum(step_costs)
        self.budget.record_spend(spend, agent=actor, task_id=None, notes=f"train:{run_id}")
        self.audit.record(
            actor=actor,
            action="train.run_finished",
            subject=run_id,
            outcome=AuditOutcome.FAILURE if alerts else AuditOutcome.SUCCESS,
            details={
                "steps": step,
                "tokens_seen": tokens_seen,
                "final_loss": final_loss,
                "alerts": alerts,
                "warnings": warnings,
                "cost_usd": spend,
            },
            cost_usd=spend,
        )
        return TrainingOutcome(
            run_id=run_id,
            status="diverged"
            if alerts
            else ("completed_with_warnings" if warnings else "completed"),
            steps=step,
            tokens_seen=tokens_seen,
            final_loss=final_loss,
            cost_usd=spend,
            checkpoint_path=checkpoint_path,
            metrics_path=str(metrics_path),
            model_path=str(model_path),
            alerts=alerts,
            warnings=warnings,
            estimate=estimate.to_dict(),
            reproducibility=reproducibility,
        )

    def _contrastive_loss(
        self,
        model: TinyTransformer,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        config: TrainConfig,
    ) -> torch.Tensor:
        anchor_mask = (anchors != PAD_ID).unsqueeze(-1)
        positive_mask = (positives != PAD_ID).unsqueeze(-1)
        anchor_vectors = model.embed(anchors, mask=anchor_mask)
        positive_vectors = model.embed(positives, mask=positive_mask)
        logits = anchor_vectors @ positive_vectors.T / max(config.temperature, 1e-3)
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    def _lr(self, config: TrainConfig, step: int) -> float:
        from llmforge.training.model import cosine_schedule

        warmup = max(config.warmup_steps, config.steps // 10)
        return cosine_schedule(step, config.steps, config.learning_rate, warmup)

    def _estimate(self, config: TrainConfig, parameters: int, tokens: int) -> CostEstimate:
        if config.device == "cpu":
            return estimate_cpu_run(
                parameters,
                tokens,
                tokens_per_second=20_000.0,
                cpu_hour_usd=config.cpu_hour_usd,
            )
        return estimate_gpu_run(parameters, tokens, gpu_hour_usd=config.gpu_hour_usd)

    def _step_cost(
        self, config: TrainConfig, tokens_per_step: float, elapsed: float, steps_done: int
    ) -> float:
        rate = config.cpu_hour_usd if config.device == "cpu" else config.gpu_hour_usd
        tokens_per_second = max(tokens_per_step * steps_done / max(elapsed, 1e-6), 1.0)
        seconds = tokens_per_step / tokens_per_second
        return seconds / 3600.0 * rate

    def _write_metric(self, path: Path, payload: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _checkpoint(
        self,
        checkpoint_dir: Path,
        model: TinyTransformer,
        optimizer: Any,
        step: int,
        config: TrainConfig,
        dataset_hash: str,
    ) -> None:
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "dataset_hash": dataset_hash,
            "commit": _git_commit(self.root),
        }
        torch.save(payload, checkpoint_dir / "latest.pt")
        torch.save(payload, checkpoint_dir / f"step_{step:06d}.pt")
