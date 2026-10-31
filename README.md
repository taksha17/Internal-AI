<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="Logo/internal-ai-lockup-dark.png">
    <img src="Logo/internal-ai-lockup.png" width="720" height="216" alt="internal-ai. Private by design.">
  </picture>
</p>

<p align="center"><strong>The control plane a company runs on its own machines to build, evaluate and serve a model it owns.</strong></p>

Most companies rent their intelligence: the data leaves, the weights stay on
someone else's hardware, and the model card does not belong to them. This is the
other arrangement. A company installs this platform on machines it controls,
points it at documents it already has, and ends up with a model whose training
data, evaluation and serving all stayed inside the boundary.

Agents do the engineering work. People take every irreversible decision.

## What it does

- **Control plane.** A dependency-ordered task graph, ten role agents with
  declared capabilities, gates that need a named second person, budgets at five
  scopes, a kill switch, an append-only hash-chained audit log, and lineage from
  every artifact back to the commit and dataset that produced it.
- **Data plane.** Ingestion, exact and near-duplicate removal, PII and secret
  scrubbing, quality and language filters, immutable dataset versions, a
  tokenizer trained from scratch, and leakage-safe splits.
- **Training.** A from-scratch transformer that records its commit, config,
  dataset hash and seed; checkpoints and resumes; caps cost; streams metrics;
  and stops itself on divergence or a loss spike.
- **Evaluation.** Frozen eval sets that only a person can approve, an
  independent evaluation agent, promotion gates, and model cards.
- **Serving.** Authenticated inference and permission-aware retrieval, per-user
  rate limits, and rollback.
- **Adaptation.** For companies that decide an existing open base is acceptable,
  a rank-8 adapter over one pinned base, streamed from disk, recorded on every
  model card by repo, revision and licence.

It is not a chat product competing with frontier models, and it is not a
distillation pipeline.

## Two tracks

| Track | Default | Starts from | Gate before any weight loads |
| --- | --- | --- | --- |
| From scratch | on | random initialisation on approved internal data | none: there is no code path that reads pretrained parameters |
| Alpha (adapter) | off | one pinned open base already on disk | `policy_decision` on `P6`; plus `base_model_exception` when the base licence is not permissive |

Provenance mode is a per-install setting, not a flag on a run. The default
`scratch_only` refuses every load. `open_weight_allowed` is decided through a
gate, recorded in the audit chain, and only then does the adapter track load a
base that is allowlisted and hash-pinned. Alpha output is never training data
for the from-scratch track.

## What stays inside

- No training on third-party model output. Distillation is a non-goal.
- Raw records never leave the boundary. Schemas, aggregate statistics, redacted
  samples, code and logs may.
- Frozen eval sets are write-protected; a change is a new version behind a new
  gate.
- Gates are human-only, and the requester cannot decide their own.
- `audit_events` is append-only and hash-chained, enforced by triggers.
- Every spend path goes through `BudgetManager` at every scope.
- Engaging the kill switch is immediate; releasing it needs a different person.

## Where those rules live

Change the rule in its own file, never around it.

| Rule | Home |
| --- | --- |
| Who may do what | [`roles.py`](src/internal_ai/roles.py), [`services/permissions.py`](src/internal_ai/services/permissions.py) |
| Every agent action | [`agent.py`](src/internal_ai/agent.py), [`tools.py`](src/internal_ai/tools.py) |
| Task states | [`store.py`](src/internal_ai/store.py) (`ALLOWED_TRANSITIONS`) |
| Spend | [`budget.py`](src/internal_ai/budget.py) |
| Datasets | [`data/`](src/internal_ai/data/) |
| Promotion | [`workflows.py`](src/internal_ai/workflows.py) |
| Gates | [`gates.py`](src/internal_ai/gates.py), `GateType` in [`domain.py`](src/internal_ai/domain.py) |
| Base models | [`training/alpha/pin.py`](src/internal_ai/training/alpha/pin.py) |
| Provenance mode | [`training/provenance.py`](src/internal_ai/training/provenance.py), policy `P6` |

## Run it

Python 3.12. CPU is enough for the demo path.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev,train]"
```

The whole loop — sample corpus, dataset, split, tokenizer, training from random
weights, a frozen eval set, evaluation, registration, promotion, serving — on a
throwaway root:

```bash
.venv/bin/python -m internal_ai.cli --root var_demo demo
```

The same path is what the test suite runs.

To click through it, create the first account and start the console:

```bash
.venv/bin/python -m internal_ai.cli --root var_console people replace \
  --id ppl_priya_raghunathan --name "Ada Admin" --title "Program lead" \
  --roles program_lead --groups platform-admins
.venv/bin/python -m internal_ai.cli --root var_console console create-account \
  --person "Ada Admin"
.venv/bin/python -m internal_ai.cli --root var_console serve --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080/console](http://127.0.0.1:8080/console), sign in with
the one-time password the command printed, and change it.

## The console

One FastAPI process serves Jinja pages and the JSON API. htmx is vendored under
`src/internal_ai/console/static/`: no CDN, no Node build, no frontend framework,
no inline script and no inline style, enforced by the content security policy
and by tests. The console is a third client of the same services as the CLI and
the API, with the same governance — it is not a separate back office.

## Verified at scale

`scripts/stress.py` builds a corpus, releases it through the real gates, trains a
model, evaluates it and races the guardrails. One run on a 16 GB CPU-only
workstation (AMD Ryzen 5 5600H) took 36.4 minutes end to end and exited 0:

| Stage | Measured |
| --- | --- |
| Ingest, dedup, PII scrub | 41,294 records from 51,750 rows in 4m50s; 6,201 exact and 141 near duplicates removed; 20,455 redactions |
| Leakage-safe split | 33,058 / 4,130 / 4,095 in 20m09s |
| Tokenizer from scratch | 1,500 of vocabulary over 33,058 documents in 33s |
| Release | a domain expert reviewed the sample, then the `dataset_release` gate approved |
| Training | stopped at step 70 of 80 by the divergence guard; 140,176 tokens at 1,653 tokens/s |
| Evaluation | recall@10 0.000 against a 0.000 keyword baseline; every safety check passed; promotion refused |
| Audit | chain verified, 17 events |

Two things that run is quoted for. The audit chain verified after two deliberate
races, and the harness refused to promote. A slim model on a synthetic corpus
does not beat keyword search, and the right behaviour is to say so: that is the
data-volume problem, not a bug in the pipeline. The serving, budget and gate
races it exposed are closed, with tests:
[`test_budget_reservations.py`](tests/test_budget_reservations.py),
[`test_gate_decision_uniqueness.py`](tests/test_gate_decision_uniqueness.py),
[`test_retrieval_cache.py`](tests/test_retrieval_cache.py).

## Repository layout

```
src/internal_ai/     control plane, data, training, evals, serving, console
tests/               the suite, including the refusal tests
scripts/             stress, probe and secret-scan tools
docs/                architecture, runbook, tester guide, console build notes
Logo/                the marks
```

## Documentation

- [Architecture](docs/architecture.md) — the planes, the task lifecycle, the
  enforcement behind each principle.
- [Runbook](docs/runbook.md) — day-to-day commands, including the budget model.
- [Tester guide](docs/tester-guide.md) — the control plane, the serving API, the
  console, search, and the refusals.
- [Console](docs/phase-2-console.md) — how the browser client is built and the
  safety rules it keeps.
- [Design](DESIGN.md) and [product](PRODUCT.md) — the console as built, and the
  product the platform is meant to be.
- [Agent conventions](AGENTS.md) and [contributors](CONTRIBUTORS.md).
- [Licensing](LICENSING.md) — what needs a commercial licence — and
  [third-party notices](THIRD_PARTY_NOTICES.md).
- [Promotion margin](docs/promotion.md) — the bar a candidate must clear before
  it may serve traffic.

## Not built yet

Stated plainly, because a platform that claims more than it does is worse than
one that claims less:

- The local answer path is a reference generator. It recomputes the prefix for
  every token and streams each layer from disk, so it is not a serving engine for
  a real 4B base on CPU.
- There is no rebase runner; comparing candidates across bases needs a second
  pinned base.
- Supervision sets cannot be released yet: their token counts are estimated.
- The sandbox is subprocess-level. Container isolation with a network policy is
  still open.
- The scheduler runs jobs locally. Slurm and Kubernetes are integration points,
  not finished backends.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). One change is one branch and one pull
request; the owner merges to `main`. Every change leaves `pytest`,
`ruff check .` and `ruff format --check .` green.

## Security

Report a vulnerability privately through the repository's security advisories,
not in a public issue. Never commit keys, tokens or records. The suite runs
`scripts/secret_scan.py` on every pull request.

## License

[PolyForm Noncommercial 1.0.0](LICENSE). People, schools, nonprofits and
governments may use, change and share the software for noncommercial purposes.
A company needs a separate licence from the copyright holder to use it in its
business, or to sell something built on it. That is deliberate: the point of the
project is a model a company owns outright, and the software that builds it is
not given away for someone else to resell.

[LICENSING.md](LICENSING.md) sets out what needs a commercial licence, why the
models you build are yours, and which base models this install will load.
Components and their licences are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
