# PRD: Agent-Driven In-House LLM Build Platform

Draft v1.0 | October 21, 2026 | Status: For review | Owner: TBD (Program Lead)

## 1. Overview

### 1.1 Problem statement

The company relies on third-party AI providers for internal development and holds many years of proprietary data. Leadership wants an in-house, locally controlled AI capability. Policy prohibits using any existing base model (open-weight or otherwise) as the foundation, so models must be trained from scratch. The target window for first value is 1-2 months, and the human team is small.

### 1.2 Product vision

A multi-agent engineering organization, a hierarchy of role agents and task-level sub-agents, supervised by a small group of experienced humans, that designs, builds, trains, evaluates, and deploys from-scratch models on company data. Agents accelerate engineering work. Humans own decisions, approvals, and quality.

### 1.3 What this product is and is not

- It IS an internal platform (orchestration, agent roles, shared artifacts, gates, budgets, evals) that produces models.
- It IS NOT the model itself, and it is not a replacement for ML expertise. It is a force multiplier for 2-4 experienced humans.
- It IS NOT a path to a frontier-grade general assistant within 2 months. See section 4.

## 2. Goals and Non-Goals

### 2.1 Goals

- G1: Deliver the first production-grade from-scratch model (embedding/retrieval) within 8 weeks.
- G2: Deliver a set of small task-specific models (classifiers, extractors) within 8 weeks.
- G3: Establish a reusable data pipeline, tokenizer, and evaluation suite that any later, larger model reuses.
- G4: Reduce human engineering effort on pipelines, training code, configs, tests, and documentation by at least 60% versus a fully human baseline.
- G5: Keep all training data inside company boundaries, with full lineage and auditability.
- G6: Keep compute spend within approved budgets, with zero uncontrolled spend events.

### 2.2 Non-goals (v1)

- A general-purpose conversational assistant competitive with frontier models.
- Training on outputs of third-party models (distillation). This is prohibited.
- Fully autonomous operation with no human approval gates.
- Procuring hardware (handled outside this product; see section 10).

## 3. Assumptions and Policy Decisions Required

The following need explicit decisions before Week 1 ends. Each materially changes scope.

| ID | Question | Why it matters | Default if undecided |
| --- | --- | --- | --- |
| P1 | Are public or licensed datasets permitted alongside internal data? | Internal data alone is likely only billions of tokens. Without outside data, models stay narrow. | Internal data only; narrow models |
| P2 | May agents run on third-party LLM APIs as development tooling? | Agents need an LLM. If banned, agents need a local model, which conflicts with the no-base-model rule. | Allowed for code/config work; no raw data sent |
| P3 | Is implementing a published architecture (e.g., a Llama-style transformer) with random initialization acceptable? | Architecture is published research, not weights. Legal must confirm. | Yes, pending legal confirmation |
| P4 | Is an audited fully open model (published data and code) ever an acceptable exception? | Would shorten the path to a capable assistant to weeks. | No |
| P5 | What data classes may be used for training (PII, customer, regulated)? | Defines the scrubbing and exclusion scope. | Exclude PII, customer-confidential, regulated |

## 4. Scope and Phasing

| Phase | Window | Deliverables | Exit criteria |
| --- | --- | --- | --- |
| 0: Foundation | Weeks 1-2 | Orchestrator, shared repo and tracker, data inventory, access approvals, eval harness skeleton, sandbox environments | Agents run a no-op end-to-end loop; data access approved for pilot sources |
| 1: Data and tokenizer | Weeks 2-4 | Cleaned, deduplicated, PII-scrubbed corpus v1; tokenizer v1; eval sets v1 | Data quality report passes review; eval sets frozen |
| 2: First models | Weeks 4-7 | Embedding model, 2-3 classifiers/extractors, optional 100M-1B domain LM | Models beat agreed baselines on frozen evals |
| 3: Serving and pilot | Weeks 7-8 | Inference API, retrieval service, pilot rollout to a small user group | Pilot users active; latency and quality SLOs met |
| 4: Scale-up (post v1) | Months 3-12 | Larger domain LM (1-7B), instruction tuning, preference tuning | Separate PRD and budget |

## 5. Users and Stakeholders

| Persona | Needs |
| --- | --- |
| Program lead (human) | Visibility into progress, budget, risk; final say on scope |
| Supervising ML lead (human) | Technical oversight; approves training runs, model promotion, eval changes |
| Senior data/infra engineers (human) | Review agent output; handle access and cluster issues |
| Security, legal, compliance | Lineage, license records, access audits, sign-off on data use |
| End users (developers, analysts, support staff) | Useful internal search, classification, extraction, and drafting tools |
| Executive sponsor | Clear cost, timeline, and capability statements |

## 6. System Architecture

### 6.1 Principles

1. Three levels only: orchestrator, role agents, sub-agents. Deeper hierarchies lose information at each handoff.
2. Artifacts over chat. Agents communicate through versioned files, tickets, and tracker entries, not long free-form messages.
3. Evals are the referee. An independent evaluation agent with write-protected held-out sets decides whether results count.
4. Least privilege. Each agent holds only the permissions its role needs.
5. Humans approve irreversible or expensive actions.
6. Every builder has a separate reviewer.

### 6.2 Components

- **Orchestrator service:** decomposes goals into tasks, assigns agents, tracks dependencies, enforces gates and budgets.
- **Agent runtime:** hosts role agents and sub-agents with tool access, sandboxing, and per-agent identity. Framework-agnostic (LangGraph, Claude Agent SDK, AutoGen, or similar).
- **Shared source of truth:** one monorepo (code, configs, docs), one experiment tracker (MLflow or W&B), one task board, one data catalog with lineage.
- **Data plane:** storage, processing jobs, and versioned datasets, all inside company boundaries. Raw data never leaves.
- **Compute plane:** job scheduler over whatever compute section 10 selects (Slurm or Kubernetes with a queue), with budget enforcement.
- **Gate service:** approval requests, signoff records, and an immutable audit log.
- **Observability:** logs, metrics, traces for agents and jobs; cost dashboards.

### 6.3 Data handling rule for agents

If agents use an external LLM API (decision P2), they may see schemas, aggregate statistics, redacted samples approved by data owners, code, and logs without data content. They may not see raw records. Data processing executes as code inside the company environment, and agents inspect only the code's summarized output.

## 7. Agent Specifications

Each role agent has: mission, sub-agents, inputs, outputs, tools, permissions, approval gates, and KPIs.

### 7.1 Program Orchestrator

- **Mission:** turn the PRD into a dependency-ordered plan, keep it current, and keep all agents within scope, schedule, and budget.
- **Sub-agents:** Planner/decomposer; Dependency and schedule tracker; Budget monitor; Status and report writer.
- **Outputs:** task graph, weekly status report, risk register updates, gate requests.
- **Permissions:** task board write; read access to trackers and cost data; no data or cluster access.
- **KPIs:** plan accuracy (planned vs. actual), blocked-task age, reporting latency.

### 7.2 Data Engineer Agent (largest workload)

- **Mission:** produce clean, deduplicated, PII-scrubbed, versioned training corpora with full lineage.
- **Sub-agents:** Source ingestion; Format normalization and extraction; Deduplication (exact and near-duplicate); PII/secret scrubber; Quality and language filter; Tokenizer trainer; Dataset versioning and lineage recorder; Data-split manager (train/validation/test, leakage guard).
- **Inputs:** approved source list, data owner rules, schema docs.
- **Outputs:** versioned datasets, tokenizer, data quality reports, lineage graph.
- **Permissions:** read on approved sources; write to dataset store; no external network egress.
- **Gates:** new source added; changes to scrubbing rules; dataset release.
- **KPIs:** PII leakage rate on audit samples (target: 0 critical findings), duplicate rate post-processing, throughput, lineage coverage (100%).

### 7.3 Data Scientist Agent

- **Mission:** understand the data and define how success is measured.
- **Sub-agents:** Corpus profiler; Eval-set builder; Baseline builder; Error analyst; Ablation analyst.
- **Outputs:** corpus profile, draft eval sets, baselines, error analyses.
- **Gates:** eval-set freeze requires human approval; after freeze, no agent may modify it.
- **KPIs:** eval coverage of target use cases, baseline quality, analysis turnaround.

### 7.4 Researcher Agent

- **Mission:** propose architectures, sizes, and training recipes grounded in published research and scaling-law estimates.
- **Sub-agents:** Literature scanner; Architecture and hyperparameter proposer; Scaling-law estimator; Experiment designer.
- **Outputs:** design memos with cited sources, experiment plans, compute estimates.
- **Gates:** any experiment with estimated cost above the approval threshold.
- **KPIs:** proposal hit rate (share of proposals that improve over baseline), estimate accuracy.

### 7.5 ML Engineer Agent

- **Mission:** implement and run training correctly.
- **Sub-agents:** Model implementation; Training-config writer; Loss and instability debugger; Checkpoint and resume manager; Reproducibility checker.
- **Outputs:** training code, configs, run logs, checkpoints, run summaries.
- **Gates:** launch of any run above the cost threshold; final-run launch.
- **KPIs:** run success rate, wasted GPU-hours, reproducibility pass rate.

### 7.6 Infra / MLOps Engineer Agent

- **Mission:** make compute available, efficient, observable, and capped.
- **Sub-agents:** Environment and container builder; Job scheduler; GPU utilization monitor; Cost tracker; Storage and checkpoint lifecycle manager; Alerting.
- **Outputs:** reproducible environments, job submissions, utilization and cost reports.
- **Permissions:** submit jobs within budget caps only; no deletion of datasets or checkpoints; no budget changes.
- **KPIs:** GPU utilization (target above 40% model FLOPs utilization on large jobs), idle-GPU cost, job queue wait.

### 7.7 AI Engineer Agent

- **Mission:** turn trained models into useful, served capabilities.
- **Sub-agents:** Task-data builder (from internal resolved tickets, docs, Q&A); Fine-tuning runner; Embedding and retrieval builder; Inference optimizer (batching, quantization); Prompt and template author.
- **Outputs:** task-tuned models, retrieval index, inference configurations.
- **Constraint:** no synthetic training data generated by third-party models.
- **KPIs:** task accuracy vs. baseline, p95 latency, cost per 1k requests.

### 7.8 Software Engineer Agent

- **Mission:** build the services and tooling around the models.
- **Sub-agents:** API/service builder; Test writer; CI/CD maintainer; Code reviewer; Documentation writer.
- **Outputs:** inference API, retrieval service, CI pipelines, runbooks.
- **KPIs:** test coverage, change failure rate, review turnaround.

### 7.9 Evaluation Agent (independent)

- **Mission:** be the single source of truth on model quality.
- **Sub-agents:** Benchmark runner; Regression detector; Contamination/leakage checker; Report generator.
- **Permissions:** read-only on models; exclusive write on eval results; held-out sets are write-protected from all other agents.
- **KPIs:** eval reproducibility, regression detection rate, time-to-report.

### 7.10 Security and Compliance Agent

- **Mission:** maintain the audit trail and catch policy violations early.
- **Sub-agents:** Access auditor; License and provenance checker; Red-team tester; Governance documentation writer.
- **Outputs:** audit logs, license register, model cards, risk assessments.
- **KPIs:** audit completeness, time-to-flag policy violations.

## 8. Functional Requirements

### 8.1 Orchestration

- FR-1: The orchestrator shall decompose goals into tasks with owners, dependencies, acceptance criteria, and cost estimates.
- FR-2: Tasks shall move through defined states: proposed, approved, in progress, in review, accepted, rejected.
- FR-3: No task shall be marked accepted without a passing review by a different agent or a human.
- FR-4: The orchestrator shall detect and halt loops (same task retried more than N times, or no state change in T minutes) and escalate to a human.

### 8.2 Data pipeline

- FR-5: Every dataset shall be versioned, immutable once released, and traceable to source systems, transformations, and code commits.
- FR-6: The pipeline shall perform exact and near-duplicate removal, language and quality filtering, and PII and secret scrubbing, with per-step statistics.
- FR-7: The pipeline shall tag every record with source and access-control metadata so models and retrieval indexes can respect permissions.
- FR-8: The pipeline shall prevent train/test leakage between training corpora and frozen eval sets.
- FR-9: A tokenizer shall be trained on the cleaned corpus, with a report on vocabulary coverage of internal jargon and code.

### 8.3 Training

- FR-10: Training code shall be reproducible from a commit hash and config, with fixed seeds and logged environment.
- FR-11: Runs shall checkpoint at defined intervals and resume automatically after node failure.
- FR-12: Runs shall stream metrics (loss, gradient norm, throughput, learning rate) to the tracker, with automatic alerts on divergence, loss spikes, or throughput collapse.
- FR-13: Each run shall have a pre-registered cost estimate and a hard cost cap enforced by the scheduler.

### 8.4 Evaluation

- FR-14: Eval sets shall be frozen after human approval, with changes requiring a new version and approval.
- FR-15: A model shall be promotable only if it beats the agreed baseline on frozen evals and passes safety and leakage checks.
- FR-16: Every promoted model shall have a model card with data sources, training config, eval results, known limitations, and intended use.

### 8.5 Serving

- FR-17: Models shall be served behind an authenticated API with per-user rate limits and request logging that respects privacy policy.
- FR-18: Retrieval shall enforce document-level permissions inherited from source systems.
- FR-19: Rollback to the previous model version shall be possible within minutes.

### 8.6 Governance

- FR-20: All agent actions with side effects (jobs, writes, deployments) shall be logged with agent identity, inputs, and outcome in an immutable audit log.
- FR-21: The gate service shall record who approved what, when, and with what evidence.

## 9. Non-Functional Requirements

| Category | Requirement |
| --- | --- |
| Security | Raw data stays inside company boundaries. Agents are sandboxed with per-role credentials. No agent holds production deploy rights without a gate. Secrets managed by a vault, never in prompts. |
| Privacy | PII scrubbing validated by sampled human audit before each dataset release. |
| Reliability | Orchestrator and tracker available 99% during business hours. Training jobs resumable. |
| Reproducibility | Any reported result reproducible from commit, config, dataset version, and seed. |
| Cost control | Per-agent, per-job, and per-week spend caps with automatic kill switches. |
| Auditability | 100% of side-effecting actions logged and retained per policy. |
| Performance | Agent task latency tracked; training MFU and inference latency SLOs defined per model. |
| Maintainability | Agents, prompts, tools, and permissions version-controlled and reviewable like code. |

## 10. Compute Requirements and Options

This section is intentionally structured so the compute workstream can plug in findings. It defines what the product needs, not which option to pick.

### 10.1 Sizing rules of thumb

Training FLOPs are approximately 6 x parameters x tokens. At roughly 400 TFLOPs sustained per H100-class GPU (about 40% utilization of bf16 peak):

| Model | Tokens | Approx. FLOPs | Ideal GPU-hours | Planning figure (3-5x for failures, ablations) |
| --- | --- | --- | --- | --- |
| 100M | 10B | 6e18 | \~4 | \~20 |
| 1B | 50B | 3e20 | \~200 | \~1,000 |
| 3B | 200B | 3.6e21 | \~2,500 | \~10,000 |
| 7B | 1T | 4.2e22 | \~29,000 | \~100,000+ |

Phase 1-3 needs roughly 2,000-5,000 GPU-hours in total, including embedding and classifier training, ablations, and fine-tuning. This is small. Phase 4 is where compute becomes a major line item.

### 10.2 Capability requirements

- C1: Multi-GPU, multi-node training with fast interconnect (NVLink intra-node; InfiniBand or equivalent inter-node) for runs beyond 8 GPUs.
- C2: Storage with high-throughput reads for token shards, and durable checkpoint storage.
- C3: Network isolation so training data never leaves company boundaries.
- C4: Scheduler support for quotas, priorities, and per-job cost caps.
- C5: Inference capacity separate from training capacity.

### 10.3 Options matrix (to be completed by the compute workstream)

| Option | Time to availability | Cost profile | Data-control fit | Notes / open items |
| --- | --- | --- | --- | --- |
| Existing internal cluster | TBD | Sunk plus opportunity cost | Strongest | Check free capacity and queue policy |
| Reserved cloud GPUs in a private tenancy (VPC, dedicated) | Days to weeks | Medium, committed | Strong with private networking and customer-managed keys | Confirm data residency and egress approvals |
| On-demand cloud GPUs | Hours to days | Highest per hour, flexible | Moderate; needs the same controls | Capacity not guaranteed; good for burst |
| GPU cloud / neocloud provider | Days to weeks | Often lowest per hour | Varies; needs vendor security review | Review contract, isolation, certifications |
| New on-prem purchase | Months | High capital cost | Strongest | Fails the 2-month window; consider for Phase 4 |
| Hybrid (internal for data prep, cloud for training bursts) | Weeks | Balanced | Depends on data movement rules | Requires tokenized or scrubbed data approval to move |

### 10.4 Decision criteria

Weighted scoring: time to first GPU, total cost for Phase 1-3 and projected Phase 4, data-control and compliance fit, interconnect quality, support and reliability, exit flexibility (no lock-in).

## 11. Data Requirements and Governance

- **Inventory first:** catalog sources, owners, volume, formats, sensitivity class, and legal status in Week 1.
- **Token budget:** estimate usable tokens per source after cleaning. Expect heavy reduction from deduplication and filtering.
- **Access:** data owners approve each source through the gate service. Approval records are stored with the dataset lineage.
- **Exclusions:** regulated data, customer-confidential content, credentials, and any class excluded by decision P5.
- **Permissions metadata:** retained so retrieval can honor source-system access rights. Training corpora for shared models exclude content not widely accessible.
- **Retention and deletion:** support removal of a source or record from future dataset versions upon request.

## 12. Evaluation Framework

| Model | Primary metrics | Baseline to beat |
| --- | --- | --- |
| Embedding / retrieval | Recall@k, MRR, nDCG on internal query-document pairs | Keyword search (BM25) and current search tool |
| Classifiers / extractors | Precision, recall, F1 per class; calibration | Rules-based system or current manual process |
| Domain LM | Perplexity on held-out internal text; task-specific completion accuracy; format validity | n-gram or smaller model |
| Serving | p50/p95 latency, throughput, error rate, cost per 1k requests | Defined SLOs |

Additional checks: contamination between train and eval, PII regurgitation probes, permission-leak tests on retrieval, and regression suites on every candidate model.

Human evaluation: a panel of domain experts rates a sample of outputs each release. Agents never act as the only judge for model promotion.

## 13. Security, Safety, and Compliance

- Agents run in sandboxes with scoped, short-lived credentials and no default internet access.
- Prompt injection defense: content from documents, tickets, or web pages is treated as data, never as instructions. Tool calls triggered by ingested content require confirmation.
- Dual control: an agent that proposes an action cannot be the one that approves it.
- Irreversible actions (deletions, deployments, budget changes, access grants) are human-only or human-approved.
- Red-team testing before pilot: memorization and PII extraction, permission bypass, and misuse scenarios.
- Records: model cards, license register, data lineage, and approval logs available for audit.

## 14. Human-in-the-Loop Gates

| Gate | Approver | Evidence required |
| --- | --- | --- |
| New data source access | Data owner plus compliance | Source description, sensitivity class, purpose |
| Dataset release | ML lead plus compliance | Quality report, PII audit sample results, lineage |
| Eval-set freeze or change | ML lead | Coverage analysis, leakage check |
| Training run above cost threshold | ML lead plus program lead | Design memo, cost estimate, success criteria |
| Model promotion | ML lead plus domain expert | Eval results vs. baseline, model card, red-team results |
| Production deployment | Engineering manager plus security | Load tests, rollback plan, access controls |
| Budget increase | Program lead plus finance | Spend report, justification |

## 15. Observability and Cost Management

- Dashboards: agent task throughput and failure rates, GPU utilization, queue times, cost by agent, project, and run, data pipeline stats, eval trends.
- Alerts: loss divergence, throughput collapse, spend approaching cap, repeated task failures, anomalous agent behavior such as unusual data access volume.
- Weekly review: program lead reviews cost versus plan and forecast to completion.
- Kill switches: per-agent, per-job, and global, executable by any designated human.

## 16. Staffing and Responsibilities

| Human role | Count | Responsibility |
| --- | --- | --- |
| Program lead | 1 | Scope, schedule, budget, stakeholder management |
| Senior ML lead | 1 (required) | Technical judgment, run health review, model promotion |
| Data/infra engineer | 1-2 | Access, cluster, pipeline review, incident response |
| Security/compliance partner | Part-time | Policy interpretation, audits, sign-offs |
| Domain experts | Part-time | Eval design, human evaluation |

Risk note: without an experienced ML lead who can recognize a quietly broken training run, agents can produce a model that looks successful and is not. This role is not optional.

## 17. Timeline (8 Weeks)

| Week | Milestones |
| --- | --- |
| 1 | Policy decisions P1-P5 resolved; data inventory; compute option selected; repo, tracker, and sandboxes live; orchestrator and four starter agents (orchestrator, data engineer, ML engineer, evaluation) running a no-op loop |
| 2 | First source approvals; ingestion and scrubbing pipelines running on pilot sources; eval harness working with placeholder sets |
| 3 | Corpus v1 candidate; tokenizer v1; baselines built; eval sets drafted |
| 4 | Dataset release gate; eval-set freeze; training code validated on a tiny run |
| 5 | Embedding model and classifier training runs; ablations; infra and researcher agents added |
| 6 | Task fine-tuning; optional small domain LM run; model candidates evaluated |
| 7 | Model promotion gate; inference API and retrieval service; red-team testing |
| 8 | Pilot rollout; metrics review; Phase 4 proposal with updated cost and capability estimates |

## 18. Success Metrics

- Delivery: embedding model and at least two task models promoted by Week 8.
- Quality: each promoted model beats its baseline by an agreed margin on frozen evals.
- Adoption: at least X pilot users active weekly (target set with sponsor).
- Efficiency: at least 60% reduction in human hours versus a fully human estimate for pipeline and engineering tasks.
- Compute: at least 80% of budget spent on productive runs; no cap breaches.
- Governance: 100% lineage coverage; zero critical PII findings in release audits; all gates documented.

## 19. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Data access and legal approvals delay start | High | High | Start approvals in Week 1; begin with the lowest-risk sources |
| GPU capacity unavailable | Medium | High | Decide compute option in Week 1; keep a burst fallback |
| Internal data too small or narrow for desired capability | High | Medium | Scope to narrow models; revisit P1 and P4 |
| Agents produce plausible but wrong results (silent failures) | High | High | Independent eval agent; human run-health review; reproducibility checks |
| Runaway agent loops or spend | Medium | High | Caps, loop detection, kill switches |
| Sensitive data exposure to external LLM APIs | Medium | High | Data handling rule (6.3); egress controls; review logs |
| Prompt injection via ingested documents | Medium | Medium | Treat content as data; confirmation for triggered tool calls |
| Overpromising a general assistant | High | Medium | Communicate scope clearly in section 4; show the policy tradeoff to leadership |
| Key-person dependency on the ML lead | Medium | High | Documentation by agents; second senior reviewer where possible |

## 20. Open Questions

1. Decisions P1-P5 (section 3).
2. Which compute option will be selected, and what is the approved budget for Phases 1-3?
3. What are the top three use cases by business value, and who owns each?
4. Which data sources are first, and who are the data owners?
5. Which agent framework and hosting environment will be used, and does it need a security review?
6. What are the cost thresholds that trigger approval gates?
7. Who is the named ML lead?

## Appendix A: Handoff Contracts (Phase 1)

| From | To | Artifact | Acceptance check |
| --- | --- | --- | --- |
| Orchestrator | Data engineer | Approved source list and task spec | Sources approved in gate service |
| Data engineer | Data scientist | Dataset candidate v1 plus quality report | Lineage 100%; dedup and PII stats attached |
| Data scientist | Evaluation agent | Draft eval sets plus baselines | Leakage check passed |
| Evaluation agent | ML lead (human) | Frozen eval sets for approval | Human signoff recorded |
| Researcher | ML engineer | Design memo with config and cost estimate | Gate passed if above threshold |
| ML engineer | Infra agent | Training job spec | Cost cap set; environment reproducible |
| ML engineer | Evaluation agent | Checkpoints and run summary | Run logs complete |
| Evaluation agent | Program lead / ML lead | Eval report vs. baseline | Promotion gate decision recorded |
| AI engineer / SWE | Security agent | Serving build and config | Red-team and access review passed |

## Appendix B: Starter Agent Set

Begin with four agents (orchestrator, data engineer, ML engineer, evaluation) on the embedding model, and add infra, researcher, AI engineer, SWE, data scientist, and security agents as the workload justifies. Each addition requires a defined mission, tool list, permission set, KPI, and reviewer pairing before activation.
