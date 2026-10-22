# LLM-Forge

The platform described in `../PRD Agent-Driven In-House LLM Build Platform.md`: a
multi-agent engineering organisation that designs, builds, trains, evaluates and
deploys **from-scratch** models on company data, supervised by a small group of
humans.

LLM-Forge is the control plane around that organisation. It is not a model and
not a replacement for ML expertise. It is a force multiplier for the 2 to 4
experienced humans named in PRD section 16.

## What is implemented

| PRD area | Where | Status |
| --- | --- | --- |
| Orchestrator, task graph, FR-1 to FR-4 | `llmforge/engine.py`, `llmforge/planner.py` | working |
| Ten role agents, section 7, least privilege | `llmforge/roles.py`, `llmforge/agent.py` | working |
| Task-scoped sub-agents, section 6.1 | `llmforge/subagents.py` | working, 47 sub-agents |
| Gates and approvals, section 14 | `llmforge/gates.py` | working |
| Budgets and kill switches, section 15 | `llmforge/budget.py` | working |
| Alerting, section 15 | `llmforge/alerts.py` | working |
| Immutable audit log, FR-20 and FR-21 | `llmforge/audit.py` | working |
| Policy decisions P1 to P5 | `llmforge/policy.py` | working, awaiting human sign-off |
| Data pipeline, FR-5 to FR-9 | `llmforge/data/` | working |
| Retention and erasure, section 11 | `llmforge/data/governance.py` | working |
| Tokenizer trained from scratch, FR-9 | `llmforge/data/tokenizer.py` | working |
| Training from scratch, FR-10 to FR-13 | `llmforge/training/` | working, CPU scale |
| Classifier and extractor models, G2 | `llmforge/training/classifier.py` | working |
| Compute plane, quotas and caps, C4 | `llmforge/compute/scheduler.py` | local backend, cluster backend is the integration point |
| Evaluation and promotion, FR-14 to FR-16 | `llmforge/evals/` | working |
| Serving, FR-17 to FR-19 | `llmforge/serving/` | working |

Nothing in this repository loads pretrained weights, and nothing trains on the
output of a third-party model. The transformer in `llmforge/training/model.py`
is randomly initialised and trained only on company data.

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # training only

# Phase 0 exit criteria: agents run an end-to-end loop with gates and audit
.venv/bin/python -m llmforge.cli demo
```

The demo submits the PRD milestone plan, approves it as a human, dispatches
every task through its role agent and a separate reviewer, opens and clears the
section 14 gates, and verifies the audit chain.

## The full model path

```bash
export LLMFORGE_HOME=./var

# 1. a pilot corpus (replace with approved internal sources)
.venv/bin/python -m llmforge.cli sample --out ./var/samples --records 150

# 2. clean, scrub, deduplicate and version it
.venv/bin/python -m llmforge.cli data build \
    --source ./var/samples/corpus.jsonl --name corpus-v1

# 3. split it, with a leakage guard
.venv/bin/python -m llmforge.cli data split --dataset-id <dataset_id>

# 4. train the tokenizer from scratch
.venv/bin/python -m llmforge.cli tokenizer train \
    --dataset-id <dataset_id> --vocab-size 3000 \
    --jargon ledgerline,backorderqueue,fulfilmenthub

# 5. train the embedding model from scratch
.venv/bin/python -m llmforge.cli train run \
    --dataset-id <dataset_id> --tokenizer <tokenizer.json> \
    --name embedding-v1 --objective contrastive --steps 150 --cost-cap 5.0

# 6. build and freeze an eval set through a human gate
.venv/bin/python -m llmforge.cli eval derive --dataset-id <dataset_id> --name retrieval-v1
.venv/bin/python -m llmforge.cli gates request --gate-type eval_set_freeze \
    --title "Freeze retrieval-v1" --subject <eval_set_id> --by data_scientist
.venv/bin/python -m llmforge.cli gates approve --gate-id <gate_id> --by ml_lead
.venv/bin/python -m llmforge.cli eval freeze --id <eval_set_id> --by ml_lead --gate-id <gate_id>

# 7. evaluate against the frozen set, then promote through the promotion gate
.venv/bin/python -m llmforge.cli model register --name embedding-v1 \
    --model-path <checkpoint.pt> --dataset-id <dataset_id>
.venv/bin/python -m llmforge.cli eval run --model-path <checkpoint.pt> --model-id <model_id> \
    --eval-set <eval_set_id> --tokenizer <tokenizer.json> --dataset-id <dataset_id>
.venv/bin/python -m llmforge.cli gates request --gate-type model_promotion \
    --title "Promote embedding-v1" --subject <model_id> --by evaluation
.venv/bin/python -m llmforge.cli model promote --id <model_id> --by ml_lead --gate-id <gate_id>
```

## Services

```bash
.venv/bin/python -m llmforge.cli serve --port 8080          # control plane
LLMFORGE_SERVING_KEYS="k1:alice:support|engineering,k2:bob" \
  .venv/bin/python -m llmforge.cli serve-model --port 8081  # inference and retrieval
```

The control plane exposes `/tasks`, `/gates`, `/audit`, `/budget`, `/policy`,
`/plan`, `/run` and `/report`. The serving API exposes `/v1/search`,
`/v1/embed`, `/v1/models` and a rollback endpoint.

## Command surface

```
llmforge init | plan | run | board | report       control plane
llmforge gates list|request|approve|reject         approval gates
llmforge audit list|verify                          immutable audit log
llmforge budget status|set|kill                    caps and kill switches
llmforge policy list|decide|confirm                decisions P1 to P5
llmforge alerts list|ack                            spend, task and access alerts
llmforge data build|split|lineage|release|pii-audit|withdraw
llmforge tokenizer train                            tokenizer from scratch
llmforge train run                                  embedding model from scratch
llmforge classifier train|eval                      task classifier
llmforge eval derive|create|freeze|run              evaluation and promotion
llmforge model register|promote|card                model registry
llmforge job submit|run                             compute plane
llmforge serve | serve-model                        control plane and inference
llmforge demo                                       Phase 0 no-op loop
llmforge sample                                     synthetic corpus for local runs
```

## Operating rules encoded in the code

- **Evals are the referee.** The evaluation agent owns eval results. Held-out
  sets are frozen by human approval and are write-protected afterwards
  (`llmforge/evals/harness.py`).
- **Every builder has a separate reviewer.** Reviewer pairing is a role
  property, not a convention (`llmforge/roles.py:reviewer_for`).
- **Dual control.** A gate requester cannot approve their own gate, and agents
  cannot approve gates at all (`llmforge/gates.py`).
- **Least privilege.** Each role holds only the capabilities its mission needs.
  Tool calls are capability checked and audited (`llmforge/tools.py`).
- **No uncontrolled spend.** Per agent, per job, per project, per week and
  global caps, plus kill switches (`llmforge/budget.py`).
- **Loop detection.** Repeated failures escalate to a human instead of burning
  budget (`llmforge/engine.py`).
- **Full lineage.** Every dataset version records its sources, per-step
  statistics, config hash and code commit (`llmforge/data/pipeline.py`).

## Honest limits

- No GPU is present in this environment, so training runs at CPU scale. The
  cost model in `llmforge/training/cost.py` estimates GPU hours and cost for
  H100-class hardware using the PRD section 10.1 formula.
- The model in this repository is small and trained on synthetic data. It is a
  working pipeline, not a useful model. Quality comes from the internal corpus.
- Agents use a deterministic offline adapter by default. Set
  `ANTHROPIC_API_KEY` and use `--llm anthropic` to drive them with a real
  model. Per policy P2 that is for code and configuration work only; raw
  records must never reach an external API.

## Layout

```
src/llmforge/
  engine.py       orchestrator: dispatch, gates, budget, loop detection
  planner.py      goal to task graph
  roles.py        the ten role agents and their capabilities
  subagents.py    task-scoped sub-agents, the third level of the hierarchy
  agent.py        agent runtime, tool use, review pairing
  gates.py        approval gates and dual control
  budget.py       caps, spend, kill switches
  alerts.py       spend, task failure and data access alerting
  audit.py        hash chained immutable audit log
  policy.py       decisions P1 to P5
  store.py        SQLite persistence and state machine enforcement
  api.py          control plane HTTP API
  serving/        inference, retrieval, permissions, rollback
  data/           pipeline, splits, tokenizer, lineage, retention, labels
  training/       from scratch model, trainer, classifier, cost model
  compute/        job scheduler with quotas, priorities and cost caps
  evals/          metrics, frozen sets, promotion
```

## Tests

```bash
.venv/bin/python -m pytest        # 65 tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Further reading

- `docs/architecture.md` for the design and the reasoning behind it
- `docs/prd-traceability.md` for the requirement-by-requirement map
- `docs/runbook.md` for day-to-day operation
- `AGENTS.md` for the conventions agents follow in this repository
