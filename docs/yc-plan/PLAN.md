# Internal-AI → "Own Your Intelligence": Audit and Claude Code Build Plan

Audit date: 2026-10-29. Repo: `taksha17/Internal-AI` at `main` (207 commits). Target: YC Winter 2027 RFS, "Own Your Intelligence" (with "AI for the Air-Gapped Economy" and "Vertical Foundation Models" as adjacent fits).

Put this file at `docs/yc-plan/PLAN.md`. Every phase below ends with a prompt block you paste into a Claude Code session. Run phases in order, one PR per numbered change, as `CONTRIBUTING.md` already requires.

---

## Part 1. What the audit found

### 1.1 What I ran

| Check | Result |
|---|---|
| `pytest` (CPU torch 2.14.1) | **432 passed** in 75s. `STATUS.md` still says 88, so it is stale |
| `ruff check` / `ruff format --check` | Clean (215 files formatted) |
| `internal_ai.cli demo` (Phase 0 loop) | 13/13 tasks accepted, audit chain verified (279 events) |
| `serve` + console | Boots; `/console/signin`, `/health`, `/docs` all 200 |
| `scripts/stress.py --records 1200` | **Crashes at HEAD**: `PermissionError: a domain expert has not reviewed the language sample`. The script predates the language-review release gate, so the README's "Verified at scale" numbers cannot be reproduced from `main` |

Not verified (could not be run here): the alpha track on real Qwen weights (no access to model hosting, no GPU), the console click-through beyond a smoke test, and real concurrency behaviour beyond what the README reports.

### 1.2 What is strong (keep and sell)

This is the part of the repo that is genuinely differentiated for regulated buyers:

- **Governance as code.** Dual-control gates (requester cannot approve), append-only hash-chained audit log enforced by triggers, immutable dataset versions, lineage, per-document ACLs carried into retrieval, budget caps and a kill switch that needs a second person to release.
- **Data plane.** Ingestion, exact and near-dup removal, PII and secret scrubbing, language tagging, toxicity policy, prompt-injection handling, leakage-safe splits, release gates.
- **Evaluation discipline.** Frozen eval sets that only a person can approve, independent evaluation agent, promotion gates, model cards, contamination and memorisation probes, retrieval-permission red-team tests.
- **Offline by default.** `INTERNAL_AI_OFFLINE` defaults on, outbound calls raise, external LLM adapter redacts prompts and audits cost. This matches the air-gapped RFS directly.
- **Engineering hygiene.** 432 tests, CI, secret scan, 26 written change records, a tester guide, an acceptance run record.

### 1.3 Gaps against the YC thesis

YC's "Own Your Intelligence" says companies want to own models built on open-weight bases, improved on their own data, run where they choose, and upgraded on their own schedule. The hard part, in their words, is turning workflows into evals, expertise into training data, and production failures into improvements, then carrying all of it forward with each new base model.

| YC need | Repo today | Gap |
|---|---|---|
| Start from strong open-weight models | Main track forbids pretrained weights. "Alpha" track loads one hash-pinned Qwen3-4B, adapts only the query projection of the first 2 layers on 4 sequences at length 64, layer-streamed for a 4 GB GPU | Alpha is a proof of concept, not a fine-tuning system. The product's centre of gravity (from scratch, no base model) is the opposite of the thesis |
| Own a model that is competitive | From-scratch `TinyTransformer` (128-dim, 4 layers, absolute positions, encoder-style embedder; causal flag exists, no trained language model). Own README: 25k-290k tokens cannot beat BM25 | A from-scratch model at this data scale cannot be the headline. It is a valid niche (small vertical models, embeddings) but not the core |
| Workflows → evals | Frozen eval sets, expert-written questions, promotion gates | No capture of real workflows or traces as eval cases; no regression suite that grows from usage |
| Expertise → training data | `supervision.py`, preference records (closed to training until release) | No annotation workflow for experts, no SFT/DPO training jobs on adapters |
| Production failures → improvements | `feedback.py`: a person files failed Ask answers by hand, exported as a next dataset version | Manual only; no trace store, no triage queue, no automatic candidate eval cases |
| Carry forward with each new base model | One hardcoded Qwen3-4B pin with Qwen-specific layer code | **No rebase pipeline.** This is the single most valuable missing capability |
| Run where they choose | Local backend works; Slurm and Kubernetes are "integration points, not finished"; no Dockerfile or Helm chart | No deployable artifact an enterprise can install |
| Serving | `/v1/search`, `/v1/embed`, `/v1/answer` (needs an external local runner via env var); no streaming; no OpenAI-compatible API | Existing apps cannot switch to it with a base-URL change |
| Enterprise readiness | Local accounts, one SQLite file, subprocess-level agent sandbox | No SSO, no Postgres, no container isolation, no SIEM export |

### 1.4 Hygiene issues to fix first

1. `scripts/stress.py` fails at HEAD (see 1.1).
2. `STATUS.md` is stale (88 vs 432 tests, "Genuinely open items" overlap with later work).
3. The three concurrency gaps the README itself documents are **still present in code**: budget `check` and `record_spend` are not atomic (no reservation or lock in `budget.py`), `/v1/search` re-embeds every visible document per query (no cache or index in `serving/service.py`), and gate decisions have no uniqueness constraint (`db.py` has no unique key on decisions).
4. **Licensing.** The repo is PolyForm Noncommercial, which is a source-available licence that forbids commercial use. That is workable for a startup only if the owners offer a separate commercial licence, and there is **no CLA** in `CONTRIBUTING.md`, so outside contributions could block relicensing. Also the alpha fallback pin (`Qwen/Qwen2.5-3B`) is labelled `qwen-research` in `pin.py`, which is a research licence; verify before any commercial use.
5. The default external model in `llm.py` is an older Claude model string; make it configurable and current.
6. README's sign-in example creates an account named "Ada Admin" under the id `ppl_priya_raghunathan`. Cosmetic, but it will confuse a first evaluator.

### 1.5 Strategic recommendation

Reposition from "an agent platform that trains LLMs from scratch" to **"the control plane that lets a regulated company own, improve, and re-base its own models."**

- Make the **open-weight adaptation plane** the primary track, with licence allowlisting and hash-pinned provenance.
- Keep **from-scratch** as a first-class but secondary track: a "zero third-party provenance" mode for customers who need it, and the home for small vertical models (embeddings, classifiers, time-series), which is the "Vertical Foundation Models" RFS.
- Make the **loop** the product: workflow → eval → data → adapter → gate → serve → trace → failure → eval/data → rebase onto the next base model.
- Lead the pitch with what already exists and is hard to copy: governance, offline operation, auditability. Frame it as: "renting intelligence is a compliance problem; this is how a regulated company owns it."

**Human decision required before Phase 1.** `AGENTS.md` non-negotiable 1 says no pretrained weights outside the alpha package, and `PRODUCT.md` repeats "no pretrained weights" as a binding rule. This plan changes that into a *provenance mode* setting (`scratch_only` vs `open_weight_allowed`), keeping the strict guarantee as a mode. The repo owner must approve rewording those files. Claude Code must not rewrite non-negotiables on its own.

---

## Part 2. Rules for the Claude Code session

Paste these once at the start, or keep them in `CLAUDE.md` pointing at `AGENTS.md`.

- Read `AGENTS.md`, `PRODUCT.md`, `CONTRIBUTING.md`, `docs/architecture.md`, `docs/prd-traceability.md` before editing anything.
- One change per branch (`change/<slug>`), one PR, one `docs/changes/NN-<slug>.md` record. Never push to `main`.
- Write the failing test first. A phase is not done until `pytest`, `ruff check .` and `ruff format --check .` are clean.
- Never weaken a non-negotiable. Where a task needs one reworded, stop and ask the owner.
- Keep the console rules: no CDN, no inline script, no inline style, `SameSite=Strict` cookies.
- Spend must go through `BudgetManager`. Audit writes go through `AuditLog.record`. Agents never decide gates.
- No third-party model output in any training set. Any model-generated text carries provenance and is excluded from the from-scratch track.
- Environment notes: Python 3.12; CPU torch is needed for tests. If the PyTorch CPU index is unreachable, use `conda-forge pytorch-cpu` in a separate environment. The default PyPI torch wheel pulls several GB of CUDA libraries and can fill a small disk.
- Keep each PR small enough to review in one sitting. Update `STATUS.md` at the end of every phase.

### Master kickoff prompt

```text
You are working in the Internal-AI repository. Read AGENTS.md, PRODUCT.md,
CONTRIBUTING.md, docs/architecture.md and docs/yc-plan/PLAN.md first.

Goal: turn this repo into a product that answers Y Combinator's "Own Your
Intelligence" request: a regulated company owns models built on open-weight
bases, improves them on its own data, runs them where it chooses, and re-bases
onto each new base model on its own schedule, with the existing governance
(dual-control gates, hash-chained audit, lineage, ACLs, offline default).

Work through PLAN.md phase by phase. For each numbered change: branch
change/<slug>, failing test first, implement, run pytest + ruff check +
ruff format --check, add docs/changes/NN-<slug>.md, open a PR. Do not touch the
non-negotiables in AGENTS.md; if a task seems to require it, stop and ask me.
Start with Phase 0. After each phase, update STATUS.md and summarise what
changed, what was tested, and what is still open.
```

---

## Part 3. Phases

Timeline assumes one engineer plus Claude Code. Phases 0-2 plus the demo in Part 4 make a credible YC-application build in about 4-6 weeks.

### Phase 0: Make the repo tell the truth (3-5 days)

**Why:** a first evaluator who clones the repo and runs the stress script should not hit a crash, and documented concurrency bugs should not still be open.

| # | Change | Acceptance |
|---|---|---|
| 0.1 | Fix `scripts/stress.py` for the language-review gate | `stress.py --records 1200 --steps 80` completes; README numbers regenerated from a fresh run on stated hardware |
| 0.2 | Atomic budget: reserve-then-settle inside one SQLite transaction (`BEGIN IMMEDIATE`) | Test: 24 threads against a $0.60 cap record at most $0.60 plus one in-flight call; no cap breach |
| 0.3 | Unique gate decision per (gate, approver) | Test: 8 simultaneous decisions by one person leave exactly one row; distinct approvers still count |
| 0.4 | Retrieval cache or index per dataset version | Test: second query against a 1,000-doc dataset is under 100 ms; cache invalidates on a new version and respects ACL groups |
| 0.5 | Refresh `STATUS.md`, README numbers, sign-in example; add Dockerfile (non-root, read-only root fs) | `docker build` and `docker run ... demo` work; STATUS test count matches CI |
| 0.6 | Licensing pack: add `CLA.md` + DCO/CLA check in CI; draft `COMMERCIAL_LICENSE.md` term sheet (human to finalise); flag fallback pin licence | CI blocks a PR without sign-off; `pin.py` lists licence class per base |

```text
Phase 0 of docs/yc-plan/PLAN.md. Do changes 0.1 to 0.6 in order, one PR each.
For 0.2, 0.3 and 0.4 write the concurrency or latency test first and show it
failing on main before you fix it. Do not edit LICENSE or AGENTS.md. For 0.6,
draft files only; tell me what a lawyer must review. Finish by updating
STATUS.md with the real test count and the real stress-run output.
```

### Phase 1: Open-weight adaptation plane (2-3 weeks)

**Why:** this is the product YC is describing. Replace the Qwen-specific streaming proof of concept with a general, licence-aware, hash-pinned base-model system.

Design:

- `ModelSource` registry: repo id, revision, file hashes, licence (SPDX), licence class (`permissive`, `restricted`, `research_only`), parameter count, architecture, tokenizer hash.
- Allowlist policy: default allows only `Apache-2.0` and `MIT`. Anything else needs a `base_model_exception` gate (legal + program lead).
- **Provenance mode** setting: `scratch_only` (today's behaviour, loaders disabled) or `open_weight_allowed`. Recorded as a `policy_decision` gate, shown on every model card and in the audit log.
- Import is offline-first: a base arrives as a signed bundle on disk (air-gap friendly), verified against pinned hashes before use. No runtime downloads.
- Adaptation backend uses standard libraries (Hugging Face Transformers + PEFT) for LoRA / QLoRA, with the layer-streamed Qwen code kept as a `tiny_gpu` fallback for 4 GB cards.
- Training jobs run through the existing scheduler, budget, sandbox, checkpoint, divergence-guard and reproducibility machinery; each adapter records base id + hash, dataset hash, config, seed, commit.
- Model card gains: base model, licence, adapter size, training data lineage, eval results vs the **un-adapted base** and vs the customer's current rented model.

| # | Change | Acceptance |
|---|---|---|
| 1.1 | `ModelSource` registry, licence classes, allowlist, provenance-mode policy and gate | Tests: restricted-licence base refused without exception gate; `scratch_only` mode makes every adapter command refuse |
| 1.2 | Offline base-model import (bundle format, hash verify, manifest) | Test: tampered shard rejected; import works with network disabled |
| 1.3 | `adapt/` package with LoRA/QLoRA SFT backend (PEFT), CPU smoke config + GPU config | Smoke test trains 20 steps on a tiny pinned test model in CI; loss decreases; adapter reproducible from seed |
| 1.4 | Training data builder for SFT: tickets, docs, expert Q&A → instruction pairs with provenance, PII-safe, no third-party model output | Test: record with `source=third_party_model` is refused |
| 1.5 | Adapter evaluation: base vs adapter vs baseline on the frozen set; promotion gate requires beating the **base** | Test: an adapter that does not beat its base is refused promotion with the reason printed |
| 1.6 | Adapter registry and model cards (base hash, licence, lineage) | Card shows base, adapter, dataset, eval deltas; audit event on every load |
| 1.7 | Keep `training/alpha` working behind the new interface or retire it with a migration note | Existing alpha tests pass or are replaced with equal coverage |

```text
Phase 1 of docs/yc-plan/PLAN.md. First, STOP and confirm with me that the
owner has approved changing AGENTS.md non-negotiable 1 and PRODUCT.md into a
provenance-mode setting; do not reword them yourself. Then implement 1.1 to
1.7. Use a tiny open model that is licensed Apache-2.0 or MIT as the CI test
fixture, stored by hash, not downloaded at test time. Never add a code path
that downloads weights at runtime. Every adapter must record base model id,
revision, file hashes and licence class on its model card and in the audit log.
The from-scratch track must keep zero code paths that read pretrained weights.
```

### Phase 2: Workflows → evals, failures → improvements (2 weeks)

**Why:** YC names this as the hard, valuable part. The repo has the frozen-eval and feedback pieces; they need to become a loop.

- **Trace store:** every serving request (question, retrieved docs, answer, citations, latency, user, model version) stored in a PII-scrubbed, ACL-tagged trace table with retention rules.
- **Triage queue:** failed citation checks, guardrail blocks, thumbs-down, low-confidence retrievals and expert flags enter a queue in the console.
- **One-click promotion** of a triaged trace into (a) a candidate eval case, (b) a candidate SFT example, or (c) a preference pair. Each needs an expert's approval and lands in a *new* dataset or eval version (never an edit).
- **Eval growth:** frozen sets are versioned; a `regression` set accumulates approved failures and runs on every candidate model.
- **Workflow capture:** a console wizard where an expert describes a workflow (inputs, expected outputs, rubric) and gets a draft eval set to approve.
- **Judges:** rubric-based scoring with a local open-weight judge model when provenance mode allows, always sampled for human check with agreement tracked.

| # | Change | Acceptance |
|---|---|---|
| 2.1 | Trace store with scrubbing, ACL tags, retention | Test: a trace never exposes text from a document the caller cannot read |
| 2.2 | Triage queue (API + console) | Test: failed-citation request appears in queue within one request |
| 2.3 | Promote trace → eval case / SFT example / preference pair behind expert approval | Test: promotion creates a new dataset version with lineage to the trace |
| 2.4 | Regression set runs on every candidate and blocks promotion on regression | Test: injected regression blocks promotion with a named failing case |
| 2.5 | Workflow-to-eval wizard | Click-through test produces a draft eval set with rubric |
| 2.6 | Judge harness with human-agreement tracking | Test: judge agreement below threshold flags the report as unreliable |

```text
Phase 2 of docs/yc-plan/PLAN.md. Build the trace store first and prove with a
test that ACLs and PII scrubbing apply to stored traces. Everything that turns
a trace into training or eval material must (1) require a human expert's
approval, (2) create a new immutable version, (3) record lineage back to the
trace id. Never edit a released dataset or a frozen eval set. Console changes
must keep the no-CDN, no-inline-script, no-inline-style rules and add tests in
the style of tests/test_console_*.py.
```

### Phase 3: Rebase onto the next base model (1-2 weeks)

**Why:** "carry all of it forward with each new base model" is the moat. Nobody else in the compliance niche does this with an audit trail.

Design: a **recipe** is a versioned bundle of (datasets, adapter config, eval sets, gates). A `rebase` job takes a recipe plus a new `ModelSource` and:

1. Checks licence class and provenance mode.
2. Re-runs adaptation on the same released datasets.
3. Runs the frozen eval sets and the regression set.
4. Produces a comparison report: old base+adapter vs new base+adapter vs un-adapted new base vs rented API baseline, with cost and latency.
5. Opens a promotion gate only if the new candidate meets the project brief's metrics.
6. Keeps the previous model serving until the gate is approved; rollback stays one command.

| # | Change | Acceptance |
|---|---|---|
| 3.1 | Recipe object (versioned, hashed) | Test: recipe re-run on the same base reproduces the adapter within tolerance |
| 3.2 | `rebase` job + comparison report | Test: report shows all four columns and refuses to propose promotion if any metric from the brief fails |
| 3.3 | Console page: "New base model available" with the report and a gate | Click-through test ends in an approved promotion and a verified audit chain |
| 3.4 | Agent task: scheduled "scan registry for approved new bases and propose rebases" (agent proposes, human decides) | Test: agent cannot start a rebase on a base outside the allowlist |
| 3.5 | Model lineage graph (base → adapter → dataset versions → evals → traces that caused changes) | Graph renders in console and exports as JSON for auditors |

```text
Phase 3 of docs/yc-plan/PLAN.md. Implement recipes and rebase. The whole point
is reproducibility and a comparison a buyer's CTO can read: old model, new
model, un-adapted new base, and the rented-API baseline on the customer's own
frozen evals. A rebase never auto-promotes. The previous model keeps serving
until a human gate is approved, and rollback is tested.
```

### Phase 4: Serving and deployment that an enterprise can install (2 weeks)

| # | Change | Acceptance |
|---|---|---|
| 4.1 | OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` with streaming, wrapping existing auth, ACL, rate limits, guardrails and audit | Test: the official OpenAI client works by changing only `base_url`; every call is audited |
| 4.2 | Pluggable inference backend (llama.cpp / vLLM / Transformers) selected per model; LoRA adapters hot-swapped per tenant | Test: two adapters on one base serve concurrently with isolation |
| 4.3 | Replace the `INTERNAL_AI_LOCAL_RUNNER` env hack with a managed runner process and health checks | Test: runner crash is detected, alerted, and requests fail closed |
| 4.4 | Container image, Helm chart, and an **air-gap bundle**: pinned wheels, model bundle, signed manifest, install script, SBOM | A fresh machine with no internet reaches `/health` and a served answer using only the bundle |
| 4.5 | Quantisation option (INT8/INT4) with eval-gated quality drop | Test: quantised model must stay within the brief's tolerance or is refused |
| 4.6 | SLO dashboard (p50/p95, tokens/s, cost per 1k requests, error rate) | Dashboard populated from the stress run |

```text
Phase 4 of docs/yc-plan/PLAN.md. The success test is: an existing app using
the OpenAI Python client switches to this platform by changing base_url and
api_key, gets streaming answers with citations, and every call appears in the
audit log. Serving must keep failing closed: an unpromoted model never serves,
a flagged answer can be withheld by policy, and ACLs apply to retrieval.
Build the air-gap bundle last and test it in a container with networking off.
```

### Phase 5: Scale and real compute (2-3 weeks, parallelisable)

| # | Change | Acceptance |
|---|---|---|
| 5.1 | Finish the Slurm backend and add a Kubernetes Job backend behind `JobBackend`, with cost caps and kill switch honoured | Integration test against a kind cluster and a Slurm-in-Docker fixture |
| 5.2 | Distributed training for adapters (`torchrun` + FSDP) | Multi-process CPU test proves parity with single process on a tiny model |
| 5.3 | Experiment tracker interface with MLflow (and optional W&B) export | Runs appear in MLflow with config, metrics, dataset hash |
| 5.4 | Modernise the from-scratch model: RoPE, RMSNorm, SwiGLU, grouped-query attention, true causal LM head, sharded streaming data loader, bf16 | Overfit test and spec checks pass; loss matches a reference implementation on a tiny batch |
| 5.5 | Vertical model track: a small time-series or tabular foundation model recipe (matches the "Vertical Foundation Models" RFS) | One end-to-end example on a public synthetic dataset, evaluated against a classical baseline |
| 5.6 | Container sandbox for agent tool execution (no network by default, read-only mounts) with a security-agent review | Escape attempts in the red-team tests fail |

```text
Phase 5 of docs/yc-plan/PLAN.md. Backends must sit behind the existing
JobBackend interface so budget caps, quotas, kill switch and audit keep
working unchanged. Do not add a second budget path. Prove each backend with an
integration test that runs in CI or is skipped with a clear reason.
```

### Phase 6: Enterprise readiness (2-3 weeks)

| # | Change | Acceptance |
|---|---|---|
| 6.1 | SSO (OIDC, then SAML) with group → role mapping; keep local accounts as break-glass only | Test: OIDC login yields the right roles; requester-cannot-approve still holds |
| 6.2 | Postgres option alongside SQLite, with migrations | The full test suite passes on both |
| 6.3 | Encryption at rest for datasets, adapters, and traces; key management interface | Test: files on disk are unreadable without the key |
| 6.4 | Audit export (JSONL/syslog/SIEM webhook) plus external anchoring of the audit hash chain | Verifier detects tampering after export |
| 6.5 | Multi-tenant isolation (workspaces) | Test: tenant A cannot see tenant B's datasets, models, traces or audit |
| 6.6 | OpenTelemetry tracing and metrics | Dashboard shows agent task latency and failures |
| 6.7 | Backup/restore and disaster-recovery runbook, tested | Restore drill verifies audit chain and model hashes |
| 6.8 | Threat model, security whitepaper, SOC 2 control mapping (documentation only; human review) | Documents reviewed by a person |

```text
Phase 6 of docs/yc-plan/PLAN.md. Governance invariants must survive every
change here: requester cannot decide their own gate, audit is append-only and
hash-chained, frozen sets are write-protected, spend goes through
BudgetManager. Add a test per invariant that runs against both SQLite and
Postgres and against SSO-authenticated actors.
```

### Phase 7: Product, proof and go-to-market (ongoing, start in week 3)

| # | Change | Acceptance |
|---|---|---|
| 7.1 | **Rent-vs-own benchmark**: run the customer's frozen eval set against a rented API and against their adapted open-weight model; show quality, latency, and cost per 1k requests side by side | One-page report generated from a real run |
| 7.2 | Onboarding wizard (brief → source approval → first release → first eval → first adapter) with a 30-minute path on sample data | Timed click-through in CI |
| 7.3 | ROI dashboard: projected annual rental spend vs owned cost, using measured token volumes | Inputs from traces; exports to PDF |
| 7.4 | Demo dataset and scripted demo for a regulated vertical (pick one: bank, hospital, defence contractor) | `make demo` reproduces the story end to end |
| 7.5 | Public docs site, quickstart, security page, pricing/licensing page | Reviewed by the owners |
| 7.6 | Pilot kit: data-readiness checklist (adapt `tests/scratch_ai_acceptance_tests.md`), pilot SOW template, success criteria template | Used on a first real pilot |

```text
Phase 7 of docs/yc-plan/PLAN.md. Build 7.1 first; it is the demo. It must use
only the customer's own frozen eval set, show the three-way comparison (rented
API, un-adapted open model, adapted open model), and never send raw records to
an external API: for the rented-API column use only redacted or synthetic
prompts that a human has approved, and mark that column clearly.
```

---

## Part 4. The minimum slice for a YC application (4-6 weeks)

If time is short, do exactly this, in this order:

1. Phase 0 in full (credibility).
2. Phase 1 changes 1.1 to 1.6 (open-weight adapters with provenance).
3. Phase 2 changes 2.1 to 2.4 (traces → triage → eval/SFT → regression).
4. Phase 3 changes 3.1 to 3.3 (rebase with comparison report).
5. Phase 4 change 4.1 (OpenAI-compatible endpoint).
6. Phase 7 changes 7.1 and 7.4 (rent-vs-own benchmark and the demo).

The story this slice supports: *"A regulated company can own a model: it improves from their own workflows and failures, runs fully inside their walls, proves its quality against the API they rent today, and moves to each new open model with a human-approved, fully audited upgrade."*

## Part 5. Decisions only a person can make

1. Approve reframing non-negotiable 1 as a provenance mode (Phase 1 blocker).
2. Commercial licence terms and CLA wording, and whether the core stays source-available or goes open-core (needs a lawyer).
3. First vertical and design-partner target (bank, hospital, defence, manufacturing).
4. Which open-weight bases are allowlisted (licence review per model, including the current Qwen fallback pin).
5. Whether agents may use an external LLM for code and config work in the product build itself (policy P2), and which model string is the default.
6. Hardware plan for the demo (single GPU workstation vs rented private cluster) and the budget for it.
7. Whether to keep the from-scratch track as a product line or reduce it to a documented niche.

## Part 6. Definition of done for "product-like"

- A new customer can install from a signed bundle in an air-gapped network and reach a served, cited answer in under a day.
- An existing OpenAI-client app works against the platform by changing `base_url`.
- A domain expert can go from a workflow description to a frozen eval set, and from a failed answer to a new approved eval case, without a developer.
- A rebase onto a new base model produces a four-way comparison and a gated promotion with one-command rollback.
- Every action is attributable, every artifact has lineage, and the audit chain verifies after export.
- The README's numbers come from a script that runs green on `main`.
