# Architecture

Design notes for the platform. Where a product requirement states a principle,
this document names the code that enforces it. The requirements themselves are
internal and are not distributed with this repository.

## Principles and enforcement

| PRD principle | Enforcement |
| --- | --- |
| Three levels only: orchestrator, role agents, sub-agents | Roles are flat in `roles.py`; a role declares sub-agents as strings, never as nested agents that can spawn agents |
| Artifacts over chat | Agents communicate through `artifact` files, task events and tracker rows. The runtime prompt carries upstream artifact paths, not conversations |
| Evals are the referee | `evals/harness.py` owns eval results and frozen sets; every other role has read-only access to eval sets |
| Least privilege | `domain.Capability` per role, checked in `tools.py` before every tool call |
| Humans approve irreversible or expensive actions | `gates.py`, with the section 14 approval matrix and required evidence |
| Every builder has a separate reviewer | `roles.reviewer_for`, enforced again at review time in `agent.py` |

## Control plane

```
goal ──▶ Planner ──▶ TaskGraph ──▶ Orchestrator
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼
        Role agent                   Gate service               Budget manager
             │                     (human decisions)         (caps, kill switch)
             ▼
        Artifacts ──▶ Reviewer role ──▶ accepted / rework
```

The orchestrator owns no domain logic of its own. It decomposes, dispatches,
enforces gates and budgets, detects loops, and reports. Everything else lives
in a service that can be tested on its own.

### Task lifecycle

`proposed → approved → in_progress → in_review → accepted | rejected`, with
`blocked` and `escalated` for operational states. Transitions are validated in
`store.set_task_state` against `ALLOWED_TRANSITIONS`, so no path through the
CLI, the API or an agent can skip review. `accepted` and `rejected` are
terminal.

### Retry and escalation

A task that fails its build returns to `in_progress` and is retried. After
`max_attempts` the task is escalated: it moves to `escalated`, an audit event
is written, and it appears in the status report's risk list. A task that has
not changed state for `stale_minutes` is escalated on the same path.

Escalation is deliberately not a failure mode of the loop: the run continues
with the remaining tasks so one bad task does not stall the week.

### Gates

A gate request carries the required approver roles from the PRD section 14
matrix and the evidence that must be attached. A gate is approved when every
required approver has approved. Two rules are absolute:

- Agents cannot decide gates. `GateService.decide` raises for
  `approver_kind="agent"`.
- The requester cannot approve. This is the dual control rule and it is
  checked before any other gate logic.

Gated tasks are blocked at dispatch until the gate is approved. When a gate is
approved, `Orchestrator.release_unblocked` returns the task to the queue, so
approval unblocks work without restarting the orchestrator.

### Budgets

Spend is recorded against five scopes at once: global, project, agent, job and
week. `BudgetManager.check` returns a decision before a task is dispatched and
`record_spend` writes the same amount to every scope, so the weekly and
per-agent views never drift apart. The kill switch is a single flag in the key
value store; when set, the orchestrator stops dispatching and the CLI prints
that the switch is engaged.

### Audit log

Each event carries the actor, the action, the subject, the outcome, the
inputs, the cost and a hash of the previous event. The table has triggers that
reject `UPDATE` and `DELETE`, so history is append-only at the storage layer
rather than by convention. `AuditLog.verify` walks the chain and reports the
first broken sequence number.

## Data plane

```
sources ──▶ ingest ──▶ sensitivity exclusion ──▶ normalize ──▶ deduplicate
                                                                    │
                     release ◀── version + lineage ◀── quality filter
                                                                    │
                                                             PII scrub
```

The order matters. Exclusion happens before any content is read into the
corpus, so excluded records never reach a later stage. Scrubbing happens
before quality filtering so that statistics are computed on scrubbed text.

### Deduplication

Exact duplicates are removed by content hash. Near duplicates are found with
banded SimHash candidate generation followed by a Jaccard check over character
shingles. The two-stage design matters: a loose hash radius alone produces
false positives, and a tight radius alone misses real near duplicates.

### Lineage

Every dataset version stores its sources with content hashes, per-step
statistics, the pipeline config and its hash, and the code commit. Coverage is
therefore 100 percent by construction: a version cannot exist without the
record of how it was produced.

### Splits and leakage

Records are assigned to train, validation and test by a seeded hash of the
record key, so the split is reproducible without storing a shuffle order. The
leakage guard then removes any validation or test record that is an exact or
near duplicate of a training record, and reports how many it removed. Eval
sets are derived from the held-out split, so the contamination check in the
evaluation harness compares genuinely disjoint sets.

## Model and training

The company owns the model it improves. Training text comes from a released
internal dataset. A change is a new dataset version and a new run, then a
human promotion gate. A released dataset is not edited, and alpha output is
not training data for the from-scratch track.

Offline is the default (`config.offline_mode`). In that mode the Anthropic
adapter raises before any HTTP call, and the alpha fetcher refuses to
download a base. A base is accepted only by `training/alpha/pin.py`
`verify_snapshot`, which checks the pinned name, revision, and shard hashes
on disk. A lab sets `INTERNAL_AI_OFFLINE=0`.

The transformer in `training/model.py` is a standard encoder: token and
position embeddings, multi-head self attention, a feed-forward block and layer
normalisation, with an optional causal mask for language modelling. It is
randomly initialised. The from-scratch track has no path that accepts external
weights, and its model card records `pretrained_weights_used: false`. The alpha
track in `training/alpha/` is the only place a pinned open base may be loaded,
and that card records the base name, revision and license. The company owns
the adapter. The from-scratch language model is the later beta track.

### Objectives

- `contrastive`: anchor and positive pairs with in-batch negatives, InfoNCE
  loss. Pairs are built from internal text by the AI engineer's task-data
  builder, never from a third-party model.
- `mlm`: masked language modelling.
- `causal`: next-token prediction for the optional domain language model.

### Run health

The trainer watches for two failure modes that produce a model that looks
successful and is not:

- **Divergence**: non-finite loss stops the run immediately. A loss spike
  beyond the running average is a warning; three consecutive spikes are
  divergence and stop the run.
- **Collapse**: a contrastive loss that falls near zero late in training means
  the model is memorising pairs instead of learning to generalise. This is
  recorded as a warning on the run and on the report, and it is exactly the
  signal a human reviewer needs before trusting the result.

### Reproducibility

Every run writes `reproducibility.json` containing the commit, the full
config, the dataset id and content hash, the tokenizer hash, the seed and the
environment. Any reported result can be reproduced from those four things plus
the code.

### Cost

`training/cost.py` implements the PRD section 10.1 formula, `6 x parameters x
tokens`, against 400 sustained TFLOPs per H100-class GPU, with a 3x planning
multiplier for failures and ablations. The estimate is computed before the run
starts; if it exceeds the cap the run is refused. During the run the measured
throughput is used to keep accumulating spend, and the run stops if the cap is
reached.

## Evaluation

The harness compares a candidate against the keyword-search baseline on the
same frozen set, and reports both. A model is promotable only if it beats the
baseline and passes the safety checks:

- **contamination**: relevant eval documents against the training split
- **PII regurgitation**: retrieval probes for raw secrets in returned documents
- **baseline informativeness**: whether the baseline had any lexical signal, so
  a reviewer can see when a comparison was easy

The last check exists because a comparison against an uninformed baseline is
not evidence. The harness reports it rather than hiding it.

## Serving

Authentication is per-user API keys with group membership. Rate limiting is a
sliding window per user. Retrieval filters documents by the access-control
metadata inherited from the source system before scoring, so a user never
receives a document their group cannot read. Request logs record the user, the
endpoint, the latency, the status and a hash of the query, never the query
text or the document content.

Rollback changes an active-version pointer, so it takes effect in seconds
rather than a redeploy. The pointer change is audited with the previous model
id.

## Known gaps

- The sandbox in `sandbox.py` uses process isolation with a scrubbed
  environment, resource limits and a jailed working directory. It is not a
  container or a VM. Production needs container-level isolation with a
  network policy.
- The compute plane is not implemented. `BudgetManager` models cost and the
  trainer models runs, but job submission to Slurm or Kubernetes is a
  remaining workstream, matching the PRD's own separation between the
  platform and the compute decision in section 10.
- The experiment tracker is the SQLite store. The interface is narrow enough
  that MLflow or W&B can be added behind it, but that has not been done.
