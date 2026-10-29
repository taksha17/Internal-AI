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
`scripts/stress.py` builds a corpus, releases it, trains a model, evaluates it,
races the guardrails and drives both servers over HTTP:

```bash
.venv/bin/python scripts/stress.py --records 5000 --steps 150
```

One run on a CPU-only laptop, 2026-10-29, with a 5,750-row synthetic corpus:

| Stage | Measured |
|---|---|
| Ingest, dedup, PII scrub | 4,642 records from 5,750 rows in 41s; 709 exact duplicates removed, 2,323 redactions, 399 records excluded by sensitivity |
| Leakage-safe split | 3,720 / 460 / 462 in 54s |
| Tokenizer from scratch | 3,610 tokens of vocabulary in 11s |
| Training, from random weights | 150 steps, 290,112 tokens, 1,149 tokens/s, loss 2.05, $0.03 |
| Evaluation | 60 held-out queries against 4,642 documents in 103s |
| Control plane | 13/13 tasks accepted and 279 audit events verified in the demo, in 0.6s |

That run is refused promotion, so the serving checks come from a smaller root
where the model clears its bar:

```bash
.venv/bin/python scripts/stress.py --root var_stress_serve --records 1200 \
  --steps 120 --brief-recall 0.02
```

What held under load:

- The audit chain verified after the run, including both races below.
- Dual control held: eight simultaneous decisions by one approver were never
  counted as two people, and no gate was decided by its requester.
- The serving rate limit is exact: 20 requests against a limit of 5 returned
  five `200`s and fifteen `429`s, each naming the limit and what remained.
- Document permissions held during the run: a caller whose groups cover none
  of a dataset's documents got `200` with no hits, not someone else's text.
- Password guessing is throttled before the account locks, and a valid sign-in
  from the same client waits behind that throttle.
- The model that missed the brief's bar was refused promotion, with the reason
  printed, even though it beat the keyword baseline.

### What the run found

Three gaps the run exposed:

1. **Retrieval cost grows with the corpus, per request.** `/v1/search` embeds
   every visible document for every query. Against about 1,100 documents one
   query takes 2.7s on this machine; 120 requests with 12 in flight all
   returned `200`, at a median of 52s and a worst case of 103s, or 0.2
   requests a second. `/v1/embed` is unaffected (7ms single, 14ms median under
   ten-way concurrency), because it embeds the query only. A vector index or
   an embedding cache per dataset is the missing piece.
2. **Budgets are a soft cap under concurrency.** `check` and `record_spend`
   are separate steps, so concurrent callers all pass the check before any
   spend is written. Twenty-four threads against a $0.60 agent cap recorded
   $1.83. A single spend path is capped; a burst is bounded only by how many
   calls are in flight.
3. **One approver can be recorded more than once.** Eight simultaneous
   decisions by the same person, repeated over ten gates, left duplicate
   decision rows in three to six rounds and up to four rows for one approver
   on a single gate. Distinct-approver counting still held, so no gate was
   decided by one person wearing two roles, but the signoff record is not
   unique and a reviewer sees more approvals than happened.

Quality is equally blunt. At 290k training tokens the from-scratch model reached
a recall@10 of 0.033 on 60 held-out queries: better than the keyword baseline
but short of the 0.05 bar the harness recorded, so promotion was refused and
the refusal was printed. A 1,200-record run reached 0.067 against a 0.02 bar and was
promoted. Neither outcome is a bug in the pipeline; 580k tokens of corpus is the
data-volume problem in section 10 of the PRD.

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
