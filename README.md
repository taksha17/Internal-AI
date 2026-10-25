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
