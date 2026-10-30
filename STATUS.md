# STATUS — hand-off & verification

Updated 2026-10-24. The operator console is in place: a person signs in and
runs the build path in the browser, and governance writes take their actor
from that session or from a personal token. The 2026-10-23 rebrand still
stands (`internal_ai`, distribution `internal-ai`, CLI `internal-ai` /
`python -m internal_ai.cli`).
The authoritative PRD is `../PRD Agent-Driven In-House LLM Build Platform.md`
(v1.0).

## Verification

- Tests: `.venv/bin/python -m pytest` → **442 passed**, exit 0, at the last
  full Linux run (torch 2.14.1+cpu). Phase 0 changes 0.2 to 0.4 add 12 tests,
  so the suite collects 454. The re-run for those changes was made on the 16 GB
  Windows test machine, since the Linux workstation used for the edit cannot
  hold a full run: the 12 new tests pass there and `ruff` is clean, while the
  18 failures that machine reports fail identically on a pristine `HEAD`
  checkout there (missing external binaries and PDF tooling in that
  environment), so none of them comes from this change.
- Lint and format: `.venv/bin/ruff check .` → clean; `.venv/bin/ruff format --check .`
  → clean.
- Phase 0 change 0.1 (`docs/yc-plan/PLAN.md`), 2026-10-30: `scripts/stress.py`
  now satisfies the language-review release gate (a domain expert files the
  review before release) and the demo model is resized to clear the
  tokens-per-parameter floor, so the script completes (exit 0). The README
  "Verified at scale" numbers are regenerated from a real run on a 16 GB AMD
  Ryzen 5 5600H Windows PC. Record: `docs/changes/29-stress-language-gate.md`.
- Phase 0 changes 0.2 to 0.4, 2026-10-30: budgets hold under concurrency
  (`reserve`/`settle`/`release` inside one `START IMMEDIATE` transaction, so 24
  threads against a $0.60 cap grant 12 and record $0.60 instead of granting 16
  and recording $1.60); one person leaves one decision row per gate (a unique
  index on `gate_decisions(gate_id, approver)`, with the duplicate reported as
  a `GateError`); and serving embeds a released dataset once per version and
  model instead of once per query, with the caller's ACLs applied to the
  ranking. Record: `docs/changes/30-phase-0-concurrency.md`.
- Tester-facing gaps closed on 2026-10-23: the serving API now reports the
  version the router actually serves as `active`, and the agentic search path
  enforces document ACLs like the serving API does. The agentic path had no
  tests before, so it now has coverage for citations, per-group visibility and
  the audited denial. `docs/tester-guide.md` is the test sheet.
- Second tester sweep on 2026-10-23, running the governance section of the
  guide by hand, closed three more gaps: a plan submitted twice into the same
  root was an unhandled `sqlite3.IntegrityError` and is now a clean refusal
  (CLI exit 1, API 409); the kill switch stopped work without saying why and
  now blocks the tasks it holds, names the reason on the board, in the run
  summary and in the audit log, and releases them back into the queue when the
  switch is lifted; `budget status` now reports the kill switch state. The
  guide's governance section was executed verbatim afterwards: 20 checks, all
  passing.
- Phase 0 exit criteria: `.venv/bin/python -m internal_ai.cli --root var_demo
  demo` → plan submitted (13 tasks), approved by a human, **13/13 accepted**,
  gates cleared, **audit chain verified (279 events)**.
- Full model path re-verified via the CLI on a synthetic corpus in `var_e2e`:
  sample → build (dedup + PII scrub) → split (85/11/11) → tokenizer from
  scratch → train (run `run_ca1d9a163c30`, 60 steps, CPU) → eval set frozen
  behind an approved `eval_set_freeze` gate → eval **passed** (model beats the
  keyword-search baseline; contamination and PII checks clean) → register →
  promote behind an approved `model_promotion` gate (`mdl_59341e45cf70`) →
  serving API: authenticated `/v1/search` and `/v1/embed` return 200 with the
  promoted model, 401 for invalid keys.
- The classifier path beats the majority-class baseline but misses the
  accuracy floor, so the evaluation agent refuses to promote it. That refusal
  is the intended behaviour.
- Browser console, 2026-10-24: sign-in with local accounts, an approvals
  inbox, a kill switch that a second person must release, and the data →
  dataset → tokenizer → eval freeze → train → evaluate → promote → ask path.
  A click-through on the synthetic corpus with separate approvers ends in a
  verified audit chain. Governance writes on the JSON API refuse a shared
  key that tries to name the actor; `scripts/probe.py` checks that. The
  command line keeps asserted names only while a root has no accounts.

## Real-data pilot (2026-10-22, second pass)

The first two pilot sources were selected by inventorying this machine with
the platform's own scanner, which auto-excluded personal, regulated and
pretrained-weight directories (it also caught that folders named "Docs"
held personal identity documents — excluded under P5):

- Source A: this repository's own engineering documentation.
- Source B: the Taksha_AI_Application_Builder codebase (extension,
  dashboard, scripts, site; its `resume` and `data` folders excluded).

Both sources cleared `new_data_source` gates (data_owner + compliance) and
were built into dataset `pilot-corpus` (98 records after dedup, 15 PII
redactions, 25k tokens, leakage-safe 72/14/12 split). A tokenizer was trained
from scratch (3143 vocab) and a from-scratch embedding model was trained
under the trainer's divergence and collapse guardrails.

Three honest findings, all from the platform working as designed:

1. The paraphrase pair/eval builder only fits corpora containing the
   synthetic ticket vocabulary, so a `title` pair mode was added
   (`train run --pair-mode title`) and `--learning-rate` was exposed;
   the default LR diverges on title pairs and the trainer stops itself.
2. On the human-authored eval (8 natural-language queries over held-out
   records, frozen behind a gate), BM25 keyword search beats the from-scratch
   model (MRR 0.18 vs 0.06). The evaluation agent refuses promotion.
   **This is the correct answer, not a failure:** 25k tokens cannot train a
   from-scratch model past keyword search. The PRD predicted this in section 3.
3. The path to a promotable model is data volume and compute (PRD section
   10), not more training loops on the same 98 records.

The human roster placeholder for `ml_lead` was replaced with a real person
(Taksha) via the new `people replace` command; a test caught and fixed a bug
where seeding overwrote real people with placeholder identities.

The rename is now complete down to the public contracts: the control-plane and
serving APIs authenticate with `X-Internal-Key` (the old `X-Forge-Key` is
rejected with 401), the platform object is `Platform`, and the runtime
database is `var/internal_ai.db`. The existing databases were migrated with
SQLite's backup API so no audit event was lost, and the split seed keeps its
original value because the released dataset versions were split with it.

Agentic retrieval over the real corpus works today and does not wait on a
promotable embedding model: the app-builder code was versioned as a tree
dataset (`data build-tree`, 40 files, lineage + ACLs) and the `ask` searcher
answers natural questions with cited file:line evidence (e.g. it locates the
autopilot's multi-page form logic and the shortlist apply script's
docstring), and honestly reports "no supporting evidence" instead of guessing
when the query does not match.

## What is implemented

| PRD | Implementation |
| --- | --- |
| Orchestrator, FR-1 to FR-4 | `engine.py`, `planner.py` |
| Role agents and sub-agents, sections 6.1 and 7 | `roles.py`, `subagents.py`, `agent.py` |
| Gates, section 14 | `gates.py` |
| Budgets and kill switches, section 15 | `budget.py` |
| Alerting, section 15 | `alerts.py` |
| Audit, FR-20 and FR-21 | `audit.py` |
| Data plane, FR-5 to FR-9 | `data/pipeline.py`, `data/transforms.py`, `data/splits.py` |
| Retention and erasure, section 11 | `data/governance.py` |
| Tokenizer from scratch, FR-9 | `data/tokenizer.py` |
| Training from scratch, FR-10 to FR-13 | `training/model.py`, `training/trainer.py` |
| Classifier models, G2 | `training/classifier.py`, `data/labels.py` |
| Compute plane, C4 | `compute/scheduler.py` |
| Evaluation and promotion, FR-14 to FR-16 | `evals/` |
| Serving, FR-17 to FR-19 | `serving/` |
| Operator console | `console/`, `services/` |

See `docs/prd-traceability.md` for the requirement-by-requirement map with the
test that covers each one.

## Genuinely open items

1. **P1 to P5 were decided on 2026-10-22** and recorded through
   `policy_decision` gates (dual control: program_lead + legal) in the default
   `var/` home. Decided values: P1 `internal_data_only`; P2
   `allowed_for_code_and_config_no_raw_data`; P3 `acceptable`; P4 `no` —
   the open-model exception was briefly allowed and then reverted to the PRD
   default the same day because no legal counsel is engaged to review it;
   revisit with counsel before ever exercising the exception; P5
   `exclude_pii_customer_confidential_regulated`. Rerun
   `.venv/bin/python -m internal_ai.cli policy list` for the live state.
2. **The pilot corpus is real but far too small.** The first real dataset
   (98 records, 25k tokens) demonstrates the pipeline end to end, but no
   from-scratch model can beat keyword search at that scale — see the
   real-data pilot above. The blockers for a promotable model are data
   volume and the compute decision in section 10.
3. **Sandbox is subprocess-level.** Environment scrubbing, resource limits and
   a jailed working directory are in place; container isolation with a network
   policy is still open.
4. **Cluster backend.** The scheduler implements quotas, priorities and
   per-job cost caps and runs jobs locally. The Slurm and Kubernetes backends
   are integration points, not finished implementations.
5. **Experiment tracker.** The SQLite store is the tracker. The interface is
   narrow enough for MLflow or W&B to sit behind it, but that has not been
   done.
6. **Human evaluation panel.** The PRD requires domain experts to rate a sample
   of outputs at each release. The platform records the results when they are
   supplied; it cannot substitute for the panel.
7. **GA and scale, section 18.** SLOs, adoption metrics and the operational
   runbook for models beyond two months.

## Recommended next actions

A. ~~Decide P1 to P5~~ — done 2026-10-22, see open items above.
B. Approve the first two pilot sources and run the pipeline on them for real.
C. Pick the compute option in section 10 and implement its backend against
   `compute/scheduler.py`.
D. Harden the sandbox to container isolation with a network policy, with the
   security agent reviewing the change.
