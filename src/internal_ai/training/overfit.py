"""Can the from-scratch model learn at all?

A handful of short sentences, trained until the loss collapses, is the cheapest
proof that the implementation is wired correctly: if a model cannot memorise
eight lines, a longer run will not help. It is deterministic, CPU-only and takes
seconds, so it belongs before a real run rather than after it.
"""

from __future__ import annotations

from typing import Any

import torch

from internal_ai.context import Platform
from internal_ai.data.tokenizer import train_tokenizer
from internal_ai.domain import AuditOutcome
from internal_ai.services.actor import Actor
from internal_ai.services.permissions import Action, require
from internal_ai.training.model import TransformerConfig, build_model

CORPUS = (
    "reset the vpn profile before you sign in again",
    "the billing export runs every friday at six",
    "open a ticket when the invoice total looks wrong",
    "restart the worker if the queue stops moving",
    "the support rota is published on monday morning",
    "rotate the api key before the quarter ends",
    "export the audit log before the retention window closes",
    "check the shard count before you start the build",
)
STEPS = 160
LEARNING_RATE = 3e-3
SEQ_LEN = 24
COLLAPSE_RATIO = 0.25
PASS_LOSS = 1.0


def overfit_check(platform: Platform, actor: Actor, steps: int = STEPS) -> dict[str, Any]:
    """Train one tiny batch until it is memorised, and record what happened."""
    require(platform.audit, actor, Action.TRAIN_MODEL, "overfit_check")
    tokenizer = train_tokenizer(list(CORPUS), vocab_size=240)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        causal=True,
    )
    torch.manual_seed(17)
    model = build_model(config, seed=17)
    sequences = []
    for line in CORPUS:
        ids = tokenizer.encode(line, max_tokens=SEQ_LEN - 1)
        padded = ids + [0] * (SEQ_LEN - len(ids))
        sequences.append(padded[:SEQ_LEN])
    inputs = torch.tensor([row[:-1] for row in sequences], dtype=torch.long)
    targets = torch.tensor([row[1:] for row in sequences], dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses: list[float] = []
    model.train()
    for _step in range(steps):
        optimizer.zero_grad()
        _logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    first = losses[0]
    final = losses[-1]
    collapsed = final <= PASS_LOSS and final <= first * COLLAPSE_RATIO
    report: dict[str, Any] = {
        "passed": collapsed,
        "steps": steps,
        "corpus": len(CORPUS),
        "vocab_size": tokenizer.vocab_size,
        "parameters": model.parameter_count(),
        "first_loss": round(first, 4),
        "final_loss": round(final, 4),
        "ratio": round(final / first, 4) if first else None,
        "target": {"final_loss_at_most": PASS_LOSS, "ratio_at_most": COLLAPSE_RATIO},
    }
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="train.overfit_check",
        subject="from-scratch model",
        outcome=AuditOutcome.SUCCESS if collapsed else AuditOutcome.FAILURE,
        details=report,
    )
    return report
