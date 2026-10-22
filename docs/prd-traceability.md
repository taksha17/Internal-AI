# PRD traceability

Where each PRD requirement is implemented and verified.

## Functional requirements

| ID | Requirement | Implementation | Test |
| --- | --- | --- | --- |
| FR-1 | Decompose goals into tasks with owners, dependencies, acceptance criteria, cost estimates | `planner.py`, `engine.submit_plan` | `test_orchestrator.py::test_plan_respects_dependencies` |
| FR-2 | Defined task states | `domain.TaskState`, `store.ALLOWED_TRANSITIONS` | `test_control_plane.py::test_task_state_transitions_are_enforced` |
| FR-3 | No task accepted without a separate review | `agent.review` raises when reviewer equals builder | `test_orchestrator.py::test_every_task_gets_a_separate_reviewer` |
| FR-4 | Detect and halt loops, escalate to a human | `engine._check_loops`, `engine.escalate` | `test_orchestrator.py::test_repeated_failure_escalates_to_a_human` |
| FR-5 | Versioned, immutable datasets traceable to sources, transformations and commits | `data/pipeline.py`, `store.add_dataset` | `test_data_plane.py::test_dataset_is_versioned_with_lineage` |
| FR-6 | Exact and near-duplicate removal, language and quality filtering, PII and secret scrubbing, per-step statistics | `data/transforms.py` | `test_data_plane.py::test_pipeline_scrubs_pii_and_drops_duplicates` |
| FR-7 | Tag every record with source and access-control metadata | `data/records.py`, `dataset_records.acl` | `test_evals_training_serving.py::test_retrieval_enforces_document_permissions` |
| FR-8 | Prevent train and eval leakage | `data/splits.py`, `evals/harness._safety_checks` | `test_data_plane.py::test_splits_are_deterministic_and_leak_free` |
| FR-9 | Tokenizer trained on the cleaned corpus with vocabulary coverage report | `data/tokenizer.py` | `test_data_plane.py::test_tokenizer_is_trained_from_scratch_and_covers_jargon` |
| FR-10 | Reproducible from commit hash and config with fixed seeds | `trainer.train` writes `reproducibility.json` | `test_evals_training_serving.py::test_training_run_completes_and_checkpoints` |
| FR-11 | Checkpoint at intervals, resume after failure | `trainer._checkpoint`, `trainer.train` resume path | `test_evals_training_serving.py::test_training_run_completes_and_checkpoints` |
| FR-12 | Stream metrics, alert on divergence, loss spikes, throughput collapse | `trainer` metric stream and detectors | `test_evals_training_serving.py::test_training_respects_cost_cap_and_records_reproducibility` |
| FR-13 | Pre-registered cost estimate and hard cap enforced by the scheduler | `training/cost.py`, `trainer` cap checks | `test_evals_training_serving.py::test_cost_estimates_follow_prd_formula` |
| FR-14 | Eval sets frozen after approval, changes need a new version | `evals/harness.freeze_eval_set` | `test_evals_training_serving.py::test_frozen_eval_set_cannot_be_written` |
| FR-15 | Promotable only if it beats the baseline and passes safety and leakage checks | `evals/harness.evaluate_retrieval`, `workflows.promote_model` | `test_evals_training_serving.py::test_promotion_requires_passing_eval_and_gate` |
| FR-16 | Model card with data sources, config, eval results, limitations, intended use | `workflows.register_model`, `workflows.model_card_markdown` | exercised by `internal_ai model card` |
| FR-17 | Authenticated API, per-user rate limits, privacy-respecting request logging | `serving/service.py`, `serving/app.py` | `test_evals_training_serving.py::test_rate_limiter_blocks_after_limit` |
| FR-18 | Retrieval enforces document-level permissions | `serving/service.RetrievalService.search` | `test_evals_training_serving.py::test_retrieval_enforces_document_permissions` |
| FR-19 | Rollback to the previous model version within minutes | `serving/service.ModelRouter.rollback` | `test_evals_training_serving.py::test_model_router_rolls_back` |
| FR-20 | All side-effecting actions logged immutably with identity, inputs, outcome | `audit.py`, `audit_events` triggers | `test_control_plane.py::test_audit_chain_verifies_and_rejects_tampering` |
| FR-21 | Gate service records who approved what, when, with what evidence | `gates.py`, `gate_decisions` | `test_control_plane.py::test_gate_requires_all_named_approvers` |
| FR-22 | Sampled PII audit before dataset release | `data/governance.py:pii_spot_check`, `data/pipeline.py:run_pii_audit` | `test_governance_compute.py::test_release_is_blocked_when_pii_survives` |
| FR-23 | Retention and erasure of a withdrawn source | `data/governance.py:RetentionManager` | `test_governance_compute.py::test_withdrawn_source_is_excluded_from_new_versions` |
| FR-24 | Task-scoped sub-agents, three levels only | `subagents.py`, `agent.py:_run_subagents` | `test_governance_compute.py::test_subagents_do_not_spawn_further_agents` |
| FR-25 | Scheduler with quotas, priorities and per-job cost caps (C4) | `compute/scheduler.py` | `test_governance_compute.py::test_job_scheduler_rejects_over_cap_jobs` |
| FR-26 | Alerts on spend pressure, repeated failures and unusual access volume | `alerts.py` | `test_governance_compute.py::test_alerts_detect_anomalous_data_access` |

## Section 7 role agents

All ten roles are defined in `roles.py` with mission, sub-agents, inputs,
outputs, capabilities, gates and KPIs. The starter set from Appendix B
(orchestrator, data engineer, ML engineer, evaluation) is enabled by default.

## Section 14 gates

| Gate | Required approvers | Implementation |
| --- | --- | --- |
| New data source access | data owner, compliance | `gates.HUMAN_APPROVER_ROLES` |
| Dataset release | ML lead, compliance | same |
| Eval set freeze | ML lead | same |
| Training run above threshold | ML lead, program lead | same |
| Model promotion | ML lead, domain expert | same |
| Production deployment | engineering manager, security | same |
| Budget increase | program lead, finance | same |
| Policy decision | program lead, legal | same |

## Section 3 policy decisions

P1 to P5 are registered in `policy.py` with the PRD defaults. Each is changed
only through `PolicyRegistry.propose`, which opens a `policy_decision` gate,
and takes effect only after that gate is approved.

## Non-functional requirements

| Category | Status |
| --- | --- |
| Security | Sandboxed subprocess with scrubbed environment and resource limits. Container-level isolation is a known gap |
| Privacy | PII scrubbing validated by per-step statistics; a sampled human audit before release is a release gate, not yet automated |
| Reliability | SQLite store with WAL and thread-local connections; training runs are resumable |
| Reproducibility | Commit, config, dataset hash, tokenizer hash, seed and environment recorded per run |
| Cost control | Five budget scopes with a kill switch at global, agent, job, project and weekly level |
| Auditability | Append-only hash-chained log with database-level protection |
| Performance | Agent task latency and token throughput are recorded per run |
| Maintainability | Agents, prompts, tools and permissions live in version-controlled code |

## Section 18 success metrics

| Metric | How the platform measures it |
| --- | --- |
| Delivery | Promoted models in the model registry with promotion gates recorded |
| Quality | Eval reports comparing against the baseline on frozen sets |
| Adoption | Serving request log, grouped by user |
| Efficiency | Human hours are tracked outside the platform; the agent task and cost history is the input |
| Compute | Spend by agent, job and week against the caps |
| Governance | Lineage coverage is 100 percent by construction; audit chain verification; gate records |
