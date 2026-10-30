<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="Logo/internal-ai-lockup-dark.png">
    <img src="Logo/internal-ai-lockup.png" width="720" height="216" alt="internal-ai. Private by design.">
  </picture>
</p>

A company trains its own model from random weights, on its own documents. People approve every gate. The corpus stays inside the boundary: no pretrained weights, no third-party model output, no raw records sent to an external API.

## What it does

The platform is a small engineering organisation of role agents, supervised by people.

- **Control.** An orchestrator walks a dependency-ordered task graph. Gates need a second person. Budgets cap spend per agent, job, and week. A kill switch stops new jobs. The audit log is append-only and hash-chained. Lineage records where every artifact came from.
- **Data.** Ingestion, exact and near-duplicate removal, PII and secret scrubbing, quality and language filters, immutable dataset versions, a tokenizer trained from scratch, and leakage-safe train and eval splits.
- **Training.** A from-scratch transformer. Each run records its commit, config, dataset hash, and seed. It checkpoints, resumes, caps cost, streams metrics, and stops on divergence or a loss spike.
- **Evaluation.** Frozen eval sets that only a person can approve, an independent evaluation agent, promotion gates, and model cards.
- **Serving.** Authenticated inference and permission-aware retrieval, with per-user rate limits and rollback.

It is not a general chat product competing with frontier models, and it is not a distillation pipeline. Agents prepare the work. People own the decisions.

## Run it

Create a virtualenv, install the CPU build of PyTorch, then the package:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"
```

The Phase 0 loop, end to end, on a demo root:

```bash
.venv/bin/python -m internal_ai.cli --root var_demo demo
```

The same path — sample corpus, dataset, split, tokenizer, training from scratch, a frozen eval set, evaluation, registration, promotion, serving — is what the test suite runs. To click through it, create the first account and open the console:

```bash
.venv/bin/python -m internal_ai.cli --root var_console people replace \
  --id ppl_priya_raghunathan --name "Ada Admin" --title "Program lead" \
  --roles program_lead --groups platform-admins
.venv/bin/python -m internal_ai.cli --root var_console console create-account \
  --person "Ada Admin"
.venv/bin/python -m internal_ai.cli --root var_console serve --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080/console](http://127.0.0.1:8080/console), sign in with the one-time password the command printed, and change it.

## Verified at scale

The demo exercises orchestration only. To see the platform under load,
`scripts/stress.py` builds a corpus, releases it through the real gates, trains
a model, evaluates it and races the guardrails:

```bash
.venv/bin/python scripts/stress.py --records 45000 --steps 80
```

One run, 2026-10-30, on a 16 GB Windows PC (AMD Ryzen 5 5600H, CPU-only
`torch` 2.14.1), took 36.4 minutes end to end and exited 0. Change 0.1 resized
the demo to a slim encoder on a ~45k-row synthetic corpus so it clears the
tokens-per-parameter floor the trainer enforces (`d_model` 64, `d_ff` 128, a
1500-vocab tokenizer); the run records the lineage commit that built it.

| Stage | Measured |
|---|---|
| Ingest, dedup, PII scrub | 41,294 records from 51,750 rows in 4m50s; 5,165,062 measured tokens; 6,201 exact + 141 near duplicates removed, 20,455 redactions, 4,102 excluded by sensitivity, 12 language-filtered |
| Leakage-safe split | 33,058 / 4,130 / 4,095 in 20m09s |
| Tokenizer from scratch | 1,500 of vocabulary over 33,058 documents in 33s |
| Language review, then release | a domain expert marked the 30-record sample; the `dataset_release` gate approved in 4m50s |
| Training, from random weights | stopped at step 70 of 80 by the divergence guard; 140,176 tokens, 1,653 tokens/s, loss 2.27, under $0.002, in 1m25s |
| Evaluation | 60 held-out queries, recall@10 0.000 against a 0.000 keyword baseline (MRR 0.0006 vs 0.0001); every safety check passed; promotion refused, in 4m30s |
| Audit | chain verified, 17 events |

What held:

- The script now runs green under every release gate at HEAD: the language
  review a domain expert files before release (change 14) and the code-commit
  lineage check a run needs a checkout to satisfy.
- The audit chain verified after the run, including both races below.
- The evaluation's safety checks all passed: train/eval contamination, PII
  regurgitation, document permission leak, instruction payloads, and both
  memorisation probes.
- The divergence guard stopped the run when validation loss rose for three
  logs, rather than shipping a model that looked fine and was not.

### What the run found

Three gaps this run left open, all closed since (changes 0.2 to 0.4):

1. **Budgets hold under concurrency.** This run's probe read 16 grants and
   $1.60 recorded against a $0.60 agent cap, because `check` and `record_spend`
   were separate steps. `BudgetManager.reserve` now takes the hold inside one
   `BEGIN IMMEDIATE` transaction and `settle` turns it into spend, so the same
   probe grants 12 and records $0.60. `tests/test_budget_reservations.py`.
2. **One signoff row per person per gate.** This run read up to seven decision
   rows for one approver on a single gate. `gate_decisions` now carries a unique
   index on (gate_id, approver), and `GateService.decide` reports the second
   write as a refusal, so eight simultaneous decisions leave one row. Distinct
   approvers still count, so no gate is decided by one person wearing two roles.
   `tests/test_gate_decision_uniqueness.py`.
3. **Serving embeds a dataset once per version.** `/v1/search` re-embedded every
   visible document per query, which is why this run did not measure serving at
   scale. `RetrievalService` now caches the document matrix against the released
   content hash and the model's identity, and applies the caller's ACLs to the
   ranking, so a warm dataset answers without re-embedding and a caller still
   never sees a document their group cannot read.
   `tests/test_retrieval_cache.py`.

This run's model was refused promotion, so none of it serves from that run; the
serving change is proven by the tests and becomes visible on the next passing
promotion.

Quality is equally blunt. The slim demo model diverged and reached a recall@10 of
0.000 on 60 held-out queries: it edges the keyword baseline on MRR and is far
short of the 0.02 bar the harness recorded, so promotion was refused and the
refusal was printed. That is the data-volume problem in section 10 of the PRD, not
a bug in the pipeline; the point of this run is that the harness runs green end to
end under the current gates, not that this synthetic corpus trains a good model.

## Read next

The implementation is the Python package `internal_ai` in [`src/internal_ai/`](src/internal_ai/). Conventions for people and agents are in [`AGENTS.md`](AGENTS.md). The product requirements are in [PRD Agent-Driven In-House LLM Build Platform.md](<PRD%20Agent-Driven%20In-House%20LLM%20Build%20Platform.md>).

- [Tester guide](docs/tester-guide.md) — the control plane, the serving API, the console, search, and the refusals.
- [Console](docs/phase-2-console.md) — how the browser client is built, and the safety rules it keeps.
- [Runbook](docs/runbook.md) — day-to-day commands, including the budget model.
- [Architecture](docs/architecture.md) — how the planes fit together.
- [Traceability](docs/prd-traceability.md) — requirement to code and test.
- [Status](STATUS.md) — what is done, what is open, what is blocked.

## License

[PolyForm Noncommercial 1.0.0](LICENSE). People, schools, nonprofits, and governments may use, change, and share the software for noncommercial purposes. A company needs a separate license from the owners to use it in its business or to sell something built on it.
