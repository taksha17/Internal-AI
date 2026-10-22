# STATUS — hand-off & verification

Updated 2026-10-22 after the rebrand to `internal_ai` (package `internal_ai`,
distribution `internal-ai`, CLI `internal-ai` / `python -m internal_ai.cli`).
The authoritative PRD is `../PRD Agent-Driven In-House LLM Build Platform.md`
(v1.0).

## Verification

- Tests: `.venv/bin/python -m pytest` → **65 passed**, exit 0 (torch 2.14.1+cpu).
- Lint and format: `.venv/bin/ruff check .` → clean; `.venv/bin/ruff format --check .`
  → clean.
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

See `docs/prd-traceability.md` for the requirement-by-requirement map with the
test that covers each one.

## Genuinely open items

1. **P1 to P5 need human sign-off.** The platform runs on the PRD defaults
   until a human decides. Each change goes through a `policy_decision` gate.
2. **No GPU and no approved corpus in this environment.** Training is CPU
   scale on synthetic data, so model quality is a pipeline demonstration, not
   a usable model. This is the real blocker for Phase 2b and Phase 4. It needs
   data owners, the compute decision in section 10 and a budget.
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

A. Decide P1 to P5 with legal and the program lead. Everything downstream
   inherits from those five answers.
B. Approve the first two pilot sources and run the pipeline on them for real.
C. Pick the compute option in section 10 and implement its backend against
   `compute/scheduler.py`.
D. Harden the sandbox to container isolation with a network policy, with the
   security agent reviewing the change.
