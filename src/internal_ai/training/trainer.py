"""Training loop with checkpointing, metric streaming, cost caps and resume.

Satisfies FR-10 to FR-13: reproducible runs, resumable checkpoints, streamed
metrics with divergence alerts, and a hard cost cap enforced during the run.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.budget import BudgetManager
from internal_ai.data.records import Record
from internal_ai.data.tokenizer import Tokenizer
from internal_ai.domain import AuditOutcome
from internal_ai.ids import content_hash, new_id, now_iso
from internal_ai.store import Store
from internal_ai.training.cost import CostEstimate, estimate_cpu_run, estimate_gpu_run
from internal_ai.training.model import (
    PAD_ID,
    TinyTransformer,
    TransformerConfig,
    build_model,
    non_finite_parameters,
    spec_violations,
)

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for training") from exc

TOKENS_PER_PARAMETER_FLOOR = 20


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
    eval_interval: int = 10
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
    vocab_rationale: str = ""

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
    token_budget: dict[str, Any] = field(default_factory=dict)
    resumed_from: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit(root: Path) -> str:
    """The code commit this run used.

    The data root is often outside the checkout, so the package's own directory is
    tried as well: a run started in a temporary data directory still records the
    commit of the code that executed it.
    """
    for candidate in (root, Path(__file__).resolve().parents[3]):
        try:
            proc = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return "unknown"


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
            ids = tokenizer.content_ids(tokenizer.encode(record.text))[: seq_len - 1]
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

    @staticmethod
    def _batch_position(batches: list[Any], index: int) -> dict[str, Any]:
        """Which batch a step reads, named by where it sits and what it holds.

        The position is a function of the step today, but a resumed run is only
        continuing the same data if the batch it reads next is the one the
        checkpoint left off before. Naming both lets a resume prove that rather
        than assume it.
        """
        batch = batches[index % len(batches)]
        rows = [
            [
                int(token)
                for part in (item if isinstance(item, tuple) else (item,))
                for token in part
            ]
            for item in batch
        ]
        return {"batch_index": index % len(batches), "batch_hash": content_hash(rows)}

    def _pair_batches(
        self, pairs: list[dict[str, str]], tokenizer: Tokenizer, config: TrainConfig
    ) -> list[list[tuple[list[int], list[int]]]]:
        encoded: list[tuple[list[int], list[int]]] = []
        for pair in pairs:
            anchor = tokenizer.content_ids(tokenizer.encode(pair["query"]))[
                : max(config.seq_len // 2, 8)
            ]
            positive = tokenizer.content_ids(tokenizer.encode(pair["document"]))[: config.seq_len]
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
        run_id: str | None = None,
        validation_records: list[Record] | None = None,
    ) -> TrainingOutcome:
        """Train, or continue, one run.

        ``run_id`` names the run directory. A caller that wants to resume a killed
        run passes the id it started with; without it every call is a new run and
        its checkpoint directory starts empty.
        """
        run_id = run_id or new_id("run")
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
        val_sequences = self.encode_records(validation_records or [], tokenizer, config.seq_len)
        val_batches = (
            self.build_batches(val_sequences, config.batch_size, config.seed + 1)
            if val_sequences
            else []
        )
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
        blocked = self._refuse_mis_wired(run_id, actor, model, model_config, metrics_path)
        if blocked is not None:
            return blocked
        parameters = model.parameter_count()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        estimate = self._estimate(config, parameters, total_tokens)
        token_budget = self._token_budget(dataset_id, parameters)
        reproducibility = {
            "commit": _git_commit(self.root),
            "config": config.to_dict(),
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "tokenizer_hash": content_hash(tokenizer.vocab),
            "seed": config.seed,
            "parameter_count": parameters,
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
                "token_budget": token_budget,
            },
        )
        if not token_budget["meets_floor"]:
            self.audit.record(
                actor=actor,
                action="train.token_budget_short",
                subject=run_id,
                outcome=AuditOutcome.REQUESTED,
                details=token_budget,
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
                token_budget=token_budget,
            )

        start_step = 0
        resumed_from = 0
        last_checkpoint = checkpoint_dir / "latest.pt"
        if resume and last_checkpoint.exists():
            state = torch.load(last_checkpoint, weights_only=True)
            model.load_state_dict(state["model_state"])
            broken = self._refuse_mis_wired(run_id, actor, model, model_config, metrics_path)
            if broken is not None:
                broken.alerts.append(f"resume refused from checkpoint step {state['step']}")
                return broken
            start_step = int(state["step"])
            refused = self._refuse_moved_data(
                run_id, actor, state.get("data_position") or {}, batches, start_step, metrics_path
            )
            if refused is not None:
                return refused
            position = self._batch_position(batches, start_step)
            optimizer.load_state_dict(state["optimizer_state"])
            resumed_from = start_step
            self.audit.record(
                actor=actor,
                action="train.resumed",
                subject=run_id,
                details={"from_step": start_step, **position},
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
            position = self._batch_position(batches, step)
            batch = batches[position["batch_index"]]
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
                if not loss_history:
                    alerts.append(
                        f"the first loss is not finite at step {step} "
                        f"({float(loss.detach())}): the model is mis-wired, the inputs are out of "
                        "range, or the learning rate is too high"
                    )
                else:
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
                line = {
                    "step": step,
                    "batch_index": position["batch_index"],
                    "batch_hash": position["batch_hash"],
                    "loss": round(loss_value, 6),
                    "ema_loss": round(ema_loss or 0.0, 6),
                    "grad_norm": round(float(grad_norm), 6),
                    "learning_rate": lr,
                    "tokens_per_second": round(throughput, 2),
                    "tokens_seen": tokens_seen,
                    "cost_usd": round(sum(step_costs), 6),
                    "timestamp": now_iso(),
                }
                if val_batches and step % config.eval_interval == 0:
                    val_loss = self._validation_loss(model, val_batches, config)
                    line["val_loss"] = round(val_loss, 6)
                self._write_metric(metrics_path, line)
            if (step + 1) % config.checkpoint_interval == 0 or step + 1 == config.steps:
                self._checkpoint(
                    checkpoint_dir,
                    model,
                    optimizer,
                    step + 1,
                    config,
                    dataset_hash,
                    data_position=self._batch_position(batches, step + 1),
                )
                if val_batches:
                    alerts.extend(
                        self._verify_checkpoint(
                            run_id,
                            actor,
                            checkpoint_dir / "latest.pt",
                            model,
                            model_config,
                            val_batches,
                            config,
                            step + 1,
                        )
                    )
            step += 1

        checkpoint_path = str(checkpoint_dir / "latest.pt")
        self._checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            step,
            config,
            dataset_hash,
            data_position=self._batch_position(batches, step),
        )
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
            token_budget=token_budget,
            resumed_from=resumed_from,
        )

    def _validation_loss(
        self, model: TinyTransformer, batches: list[Any], config: TrainConfig
    ) -> float:
        """Mean loss on held-out batches, measured on a copy.

        The copy keeps the masking on its own seed, and the training model and its
        random stream are left as they were, so a held-out number can be compared
        between steps and between memory and a reloaded checkpoint.
        """
        frozen = copy.deepcopy(model)
        frozen.eval()
        totals: list[float] = []
        with torch.random.fork_rng():
            torch.manual_seed(config.seed + 991)
            with torch.no_grad():
                for batch in batches:
                    inputs, targets, _ = self._make_batch(batch, config)
                    if config.objective == "contrastive":
                        loss = self._contrastive_loss(frozen, inputs, targets, config)
                    else:
                        _, loss = frozen(inputs, targets)
                    if loss is not None and torch.isfinite(loss):
                        totals.append(float(loss.detach()))
        return sum(totals) / len(totals) if totals else float("nan")

    def _verify_checkpoint(
        self,
        run_id: str,
        actor: str,
        path: Path,
        model: TinyTransformer,
        model_config: TransformerConfig,
        val_batches: list[Any],
        config: TrainConfig,
        step: int,
    ) -> list[str]:
        """Reload a checkpoint and re-measure, so the file is shown to hold the model."""
        in_memory = self._validation_loss(model, val_batches, config)
        state = torch.load(path, weights_only=True)
        with torch.random.fork_rng():
            reloaded = TinyTransformer(model_config)
        reloaded.load_state_dict(state["model_state"])
        restored = self._validation_loss(reloaded, val_batches, config)
        delta = abs(in_memory - restored)
        matches = delta <= 1e-5
        self.audit.record(
            actor=actor,
            action="train.checkpoint_verified" if matches else "train.checkpoint_mismatch",
            subject=run_id,
            outcome=AuditOutcome.SUCCESS if matches else AuditOutcome.FAILURE,
            details={
                "step": step,
                "val_loss": round(in_memory, 6),
                "reloaded_val_loss": round(restored, 6),
                "delta": round(delta, 8),
            },
        )
        if matches:
            return []
        return [
            f"the checkpoint at step {step} reloaded to val loss {restored:.4f} against "
            f"{in_memory:.4f} in memory; the file does not hold the model"
        ]

    def _refuse_moved_data(
        self,
        run_id: str,
        actor: str,
        recorded: dict[str, Any],
        batches: list[Any],
        start_step: int,
        metrics_path: Path,
    ) -> TrainingOutcome | None:
        """Resume only onto the batch the checkpoint left off before.

        A run continued on different data is a different run wearing the same id,
        which is how a broken resume reads as a successful one.
        """
        if not recorded:
            return None
        expected = self._batch_position(batches, start_step)
        if recorded == expected:
            return None
        self.audit.record(
            actor=actor,
            action="train.resume_position_mismatch",
            subject=run_id,
            outcome=AuditOutcome.DENIED,
            details={"recorded": recorded, "expected": expected, "step": start_step},
        )
        return TrainingOutcome(
            run_id=run_id,
            status="blocked_by_data_position",
            steps=0,
            tokens_seen=0,
            final_loss=float("nan"),
            cost_usd=0.0,
            checkpoint_path="",
            metrics_path=str(metrics_path),
            model_path="",
            alerts=[
                f"step {start_step} resumes on batch {expected['batch_index']} "
                f"({expected['batch_hash']}) but the checkpoint was written before batch "
                f"{recorded.get('batch_index')} ({recorded.get('batch_hash')})"
            ],
        )

    def _refuse_mis_wired(
        self,
        run_id: str,
        actor: str,
        model: TinyTransformer,
        config: TransformerConfig,
        metrics_path: Path,
    ) -> TrainingOutcome | None:
        """A model that does not match its spec, or holds a NaN, never trains.

        Shapes and finite weights cost milliseconds to check and are expensive to
        discover from a finished run that produced a model nobody can trust.
        """
        violations = spec_violations(model, config)
        non_finite = non_finite_parameters(model)
        if not violations and not non_finite:
            return None
        self.audit.record(
            actor=actor,
            action="train.model_spec_failed",
            subject=run_id,
            outcome=AuditOutcome.DENIED,
            details={
                "violations": violations,
                "non_finite": non_finite,
                "spec": config.to_dict(),
                "parameters": model.parameter_count(),
            },
        )
        alerts = list(violations)
        if non_finite:
            alerts.append("non-finite weights: " + ", ".join(non_finite))
        return TrainingOutcome(
            run_id=run_id,
            status="blocked_by_spec",
            steps=0,
            tokens_seen=0,
            final_loss=float("nan"),
            cost_usd=0.0,
            checkpoint_path="",
            metrics_path=str(metrics_path),
            model_path="",
            alerts=alerts,
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
        from internal_ai.training.model import cosine_schedule

        warmup = max(config.warmup_steps, config.steps // 10)
        return cosine_schedule(step, config.steps, config.learning_rate, warmup)

    def _token_budget(self, dataset_id: str, parameters: int) -> dict[str, Any]:
        """Tokens in the corpus against the parameters that will learn from them.

        Twenty tokens per parameter is the PRD's planning floor, not a promise of
        quality; a run below it is recorded as short rather than refused, because a
        small demonstration is a legitimate thing to run.
        """
        dataset = self.store.get_dataset(dataset_id)
        tokens = dataset.token_count if dataset else 0
        measured = bool(dataset and dataset.stats.get("token_count_kind") == "measured")
        per_parameter = round(tokens / parameters, 2) if parameters else 0.0
        return {
            "dataset_tokens": tokens,
            "parameters": parameters,
            "tokens_per_parameter": per_parameter,
            "floor": TOKENS_PER_PARAMETER_FLOOR,
            "meets_floor": tokens >= TOKENS_PER_PARAMETER_FLOOR * parameters,
            "count_kind": "measured" if measured else "estimate",
        }

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
        data_position: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "dataset_hash": dataset_hash,
            "data_position": data_position or {},
            "commit": _git_commit(self.root),
        }
        torch.save(payload, checkpoint_dir / "latest.pt")
        torch.save(payload, checkpoint_dir / f"step_{step:06d}.pt")
