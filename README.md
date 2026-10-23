# Internal-AI

**Internal-AI** is an agent-driven in-house LLM build platform. It designs,
trains, evaluates, and deploys **from-scratch** models on company data — no
pretrained weights, no third-party model output, no raw records sent to an
external API. Built so a data-rich company can move off third-party AI
providers (Claude, ChatGPT, etc.) while keeping every byte of its proprietary
corpus inside company boundaries.

## What it is
- A multi-agent engineering organisation (orchestrator + role agents:
  data, ML, evaluation, infra, security, program lead, …) supervised by a
  small group of humans.
- A control plane: orchestrator, dependency-ordered task graph, approval
  gates with dual control, per-agent/job/weekly budget caps and kill
  switches, an immutable hash-chained audit log, and full data lineage.
- A data plane: ingestion, exact/near-duplicate removal, PII/secret scrubbing,
  quality/language filtering, versioned immutable datasets, tokenizer trained
  from scratch, and leakage-safe train/eval splits.
- A training plane: from-scratch transformer, reproducible runs (commit +
  config + dataset hash + seed), checkpointing, resume, cost caps, metric
  streaming, and divergence/loss-spike detectors.
- An eval plane: frozen eval sets approved by humans, independent evaluation
  agent, promotion gates, model cards.
- A serving plane: authenticated inference + permission-aware retrieval API
  with per-user rate limits and rollback.

## What it is NOT
- A general-purpose conversational assistant competitive with frontier models.
- A distillation pipeline (training on third-party model output is prohibited).
- Fully autonomous operation (humans own decisions, approvals, and quality).

## Repository contents
- `src/internal_ai/` — the platform implementation (Python package `internal_ai`):
  orchestrator, role agents, gates, budgets, audit log, data pipeline,
  training, evals, serving API, CLI, and a 73-test suite. Platform docs live in
  `docs/` and `AGENTS.md`.
- `PRD Agent-Driven In-House LLM Build Platform.md` — the product requirements
  document (v1.0): scope, phasing, agent specifications, functional/non-functional
  requirements, gates, and success metrics.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"

.venv/bin/python -m internal_ai.cli --root var_demo demo   # Phase 0 loop end to end
```

The full model path (sample corpus → build dataset → split → train tokenizer →
train from scratch → freeze eval set → evaluate → register → promote → serve)
is exercised by the test suite; see `docs/runbook.md` for day-to-day commands
and `docs/tester-guide.md` for a test sheet against a running instance.

## Documentation
- `docs/tester-guide.md` — test sheet for the control-plane and serving APIs,
  the agentic search path and the governance refusals that must hold.
- `docs/runbook.md` — operating commands, including the budget model.
- `docs/architecture.md` — how the planes fit together.
- `docs/prd-traceability.md` — requirement to code and test mapping.
- `STATUS.md` — what is done, what is open, what is blocked.

## License
Apache 2.0 — see `LICENSE`.
