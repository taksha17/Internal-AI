# From-Scratch AI Build: Acceptance Test Checklist

Use this file to test the product (the agent-driven LLM build platform and the model it produces) against every requirement for building an AI from scratch.

## How to use

1. Copy this file per test run (e.g., `run_2026-11-04.md`).
2. For each test, mark the result: `[x]` pass, `[ ]` not yet passed / fail. Add evidence (link, log, report path) in the **Evidence** line.
3. Severity: **P0** = blocker (must pass before any training or release), **P1** = required for v1, **P2** = should have.
4. A section passes only when all its P0 and P1 tests pass.
5. Final sign-off is in the Scorecard at the bottom.

**Run metadata**

- Product / version:
- Model(s) under test:
- Test date:
- Tester(s):
- Environment (cluster / cloud / on-prem):

---

## 1. Foundations

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| F-01 | P0 | Target defined | Written statement of what the model must do, languages, context length, and quality bar vs. current alternative |
| F-02 | P0 | Success metrics defined | Numeric thresholds exist per use case (e.g., Recall@10 >= X, F1 >= Y) |
| F-03 | P0 | Constraints documented | Budget, timeline, data restrictions, deployment location, latency and cost targets all written down |
| F-04 | P1 | Policy decisions recorded | Decisions on external data, agent use of third-party LLMs, architecture reuse, and data classes are documented and signed |
| F-05 | P0 | Team coverage | Named owners exist for ML lead, data engineering, infra, evaluation, and security/legal |
| F-06 | P1 | Scope honesty | Documentation states what the model will NOT do (e.g., not a frontier general assistant) |

- [ ] F-01  - [ ] F-02  - [ ] F-03  - [ ] F-04  - [ ] F-05  - [ ] F-06

Evidence:

---

## 2. Data

### 2.1 Sources and legal clearance

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| D-01 | P0 | Source inventory | Every data source listed with owner, volume, format, sensitivity class, legal status |
| D-02 | P0 | Approval records | Each source used in training has a recorded approval (who, when, evidence) |
| D-03 | P0 | Exclusion enforcement | Regulated, customer-confidential, and excluded classes are provably absent (sampled audit finds zero) |
| D-04 | P1 | License register | Every external dataset has a license entry and has been checked for compatibility with intended use |

### 2.2 Volume and mixture

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| D-05 | P0 | Token budget computed | Usable token count after cleaning is measured and compared with the target (>= 20 tokens/parameter minimum; 100+ for small models where intended) |
| D-06 | P1 | Data mixture documented | Ratios of each source category recorded and justified by ablation results |
| D-07 | P2 | Mixture ablations | Small-scale experiments exist comparing at least two mixtures |

### 2.3 Pipeline stages

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| D-08 | P1 | Extraction and normalization | PDFs, HTML, code, logs convert to clean text; sampled output reviewed with error rate under agreed threshold |
| D-09 | P1 | Language identification | Language tags assigned; mislabel rate measured on a sample |
| D-10 | P1 | Quality filtering | Filter applied with before/after statistics and sampled review |
| D-11 | P0 | Exact deduplication | Zero exact duplicates remain (verified by hash check) |
| D-12 | P1 | Near-duplicate removal | MinHash (or equivalent) run; residual near-duplicate rate on sample under agreed threshold |
| D-13 | P0 | PII scrubbing | Human audit of random sample finds zero critical PII leaks; automated detectors run with logged recall estimate |
| D-14 | P0 | Secret scrubbing | Scan for API keys, passwords, tokens finds none in the final corpus |
| D-15 | P1 | Toxicity / safety filtering | Filter applied and measured; policy for borderline content documented |
| D-16 | P0 | Decontamination | No overlap between training data and frozen eval sets (n-gram or embedding-based check, report attached) |

### 2.4 Storage, format, and lineage

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| D-17 | P1 | Tokenized, sharded format | Data stored as shards streamable at the throughput training needs (no GPU starvation) |
| D-18 | P0 | Versioning | Each dataset version is immutable and identifiable by ID/hash |
| D-19 | P0 | Lineage | Any record or shard traces back to source, transformations, and code commit (100% coverage on a sampled check) |
| D-20 | P1 | Access-control metadata | Records carry source and permission tags usable by retrieval |
| D-21 | P1 | Deletion support | A source or record can be removed from future dataset versions; process tested once |

- [ ] D-01  - [ ] D-02  - [ ] D-03  - [ ] D-04  - [ ] D-05  - [ ] D-06  - [ ] D-07
- [ ] D-08  - [ ] D-09  - [ ] D-10  - [ ] D-11  - [ ] D-12  - [ ] D-13  - [ ] D-14
- [ ] D-15  - [ ] D-16  - [ ] D-17  - [ ] D-18  - [ ] D-19  - [ ] D-20  - [ ] D-21

Evidence:

---

## 3. Tokenizer

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| T-01 | P0 | Trained on own data | Tokenizer trained on the cleaned corpus (BPE/unigram), with training config recorded |
| T-02 | P1 | Vocabulary size justified | Chosen size (commonly 32k-128k) has a documented rationale |
| T-03 | P1 | Special tokens | Padding, BOS/EOS, separators, and any control tokens defined and tested |
| T-04 | P1 | Coverage report | Fertility (tokens per word) measured for internal jargon, code, numbers, and non-English text |
| T-05 | P1 | Round-trip test | Encode then decode returns original text on a diverse sample (including whitespace and unicode) |
| T-06 | P0 | Frozen before pretraining | Tokenizer version locked and hashed; recorded in the training config |

- [ ] T-01  - [ ] T-02  - [ ] T-03  - [ ] T-04  - [ ] T-05  - [ ] T-06

Evidence:

---

## 4. Model Design

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| M-01 | P0 | Architecture specified | Written spec: layers, width, heads, attention variant, positional encoding, normalization, activation, context length |
| M-02 | P1 | Sizing justified | Parameter and token counts backed by a scaling-law or compute-budget estimate |
| M-03 | P1 | Hyperparameters documented | Learning-rate schedule, batch size, warmup, weight decay, optimizer, precision recorded |
| M-04 | P1 | Implementation matches spec | Parameter count and layer shapes of the built model match the spec exactly |
| M-05 | P1 | Small-scale ablations | At least one set of 10M-100M experiments informs architecture, learning rate, and data mix |
| M-06 | P0 | Overfit test | Model can overfit a tiny dataset to near-zero loss (confirms implementation correctness) |
| M-07 | P1 | Initial loss sanity | Starting loss is close to ln(vocab_size) |
| M-08 | P2 | Numerical checks | No NaN/Inf in forward or backward pass on a test batch in the target precision |

- [ ] M-01  - [ ] M-02  - [ ] M-03  - [ ] M-04  - [ ] M-05  - [ ] M-06  - [ ] M-07  - [ ] M-08

Evidence:

---

## 5. Compute and Infrastructure

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| C-01 | P0 | Compute secured | Reserved or owned GPU capacity confirmed for the planned run, with dates |
| C-02 | P0 | Budget sizing | Estimate (about 6 x params x tokens FLOPs) with 3-5x overhead is documented and approved |
| C-03 | P1 | Interconnect verified | Multi-node bandwidth test (e.g., NCCL all-reduce) meets expected throughput |
| C-04 | P1 | Storage throughput | Data loader sustains required tokens/sec without stalling GPUs |
| C-05 | P1 | Checkpoint storage | Durable storage sized for N checkpoints; write and restore time measured |
| C-06 | P1 | Scheduler | Slurm/Kubernetes queue works; quotas and priorities configured |
| C-07 | P1 | Node health checks | Failed or degraded GPUs/nodes are detected and excluded automatically |
| C-08 | P0 | Network isolation | Training environment cannot send training data outside company boundaries (egress test) |
| C-09 | P2 | On-prem extras | Power, cooling, spare hardware plan documented (if on-prem) |
| C-10 | P1 | Separate inference capacity | Serving does not compete with training for the same GPUs |

- [ ] C-01  - [ ] C-02  - [ ] C-03  - [ ] C-04  - [ ] C-05
- [ ] C-06  - [ ] C-07  - [ ] C-08  - [ ] C-09  - [ ] C-10

Evidence:

---

## 6. Training Software Stack

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| S-01 | P1 | Framework in place | Training stack (e.g., PyTorch + Megatron/NeMo/torchtitan/DeepSpeed) runs a multi-GPU job end to end |
| S-02 | P1 | Parallelism | Required parallelism (data, FSDP/ZeRO, tensor, pipeline) works at the planned scale |
| S-03 | P1 | Efficiency | Mixed precision, fused kernels, and FlashAttention enabled; model FLOPs utilization measured (target at or above 40% on large jobs) |
| S-04 | P0 | Checkpoint and resume | Kill a run mid-training; it resumes from the last checkpoint with matching loss trajectory |
| S-05 | P0 | Data-loader state | Resumed run does not repeat or skip data (verified by logged shard/offset) |
| S-06 | P1 | Automatic failure recovery | Simulated node failure triggers restart without manual steps |
| S-07 | P0 | Reproducibility | Same commit, config, dataset version, and seed give the same early-loss curve within tolerance |
| S-08 | P1 | Experiment tracking | Loss, gradient norm, throughput, LR, GPU utilization stream to W&B/MLflow |
| S-09 | P1 | Alerts | Alerts fire on divergence, loss spike, throughput collapse (tested with injected fault) |
| S-10 | P1 | Cost cap | Job is killed automatically when its pre-set cost cap is reached |

- [ ] S-01  - [ ] S-02  - [ ] S-03  - [ ] S-04  - [ ] S-05
- [ ] S-06  - [ ] S-07  - [ ] S-08  - [ ] S-09  - [ ] S-10

Evidence:

---

## 7. Pretraining Run

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| R-01 | P0 | Tiny smoke test | A tiny config runs full pipeline (data to checkpoint to eval) successfully |
| R-02 | P1 | Short target-scale run | Throughput and stability confirmed before the full run |
| R-03 | P1 | Instability playbook | Written steps for loss spikes/NaNs (rollback, LR change, skip batches); drill performed |
| R-04 | P0 | Loss curve health | Loss decreases smoothly; no unresolved spikes; validation loss tracks training loss |
| R-05 | P1 | In-training evals | Periodic evals run during training and logged |
| R-06 | P1 | Budget tracking | Actual GPU-hours vs. plan reported; overrun triggers review |
| R-07 | P1 | Final checkpoint integrity | Checkpoint loads cleanly, hashes recorded, and reproduces logged eval numbers |
| R-08 | P0 | Human run-health review | Experienced ML lead signed off that the run is genuinely healthy (not just plausible-looking) |

- [ ] R-01  - [ ] R-02  - [ ] R-03  - [ ] R-04  - [ ] R-05  - [ ] R-06  - [ ] R-07  - [ ] R-08

Evidence:

---

## 8. Post-Training

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| P-01 | P1 | SFT data built | Instruction-response set exists, with source breakdown, size, and quality review |
| P-02 | P0 | No prohibited data | SFT/preference data contains no outputs from third-party models (provenance check) |
| P-03 | P1 | SFT run | Fine-tuned model beats the base on instruction-following evals |
| P-04 | P2 | Preference tuning | DPO/RLHF (or equivalent) run with documented preference data and measured gain |
| P-05 | P1 | Safety behavior | Refusal, privacy, and formatting behaviors tested on a dedicated suite |
| P-06 | P2 | Tool use / structured output | JSON validity rate and tool-call accuracy meet target (if required) |
| P-07 | P2 | Long-context | Quality at the target context length verified (needle-in-haystack or task-based) |
| P-08 | P1 | No capability regression | Post-training does not drop base evals beyond agreed tolerance |

- [ ] P-01  - [ ] P-02  - [ ] P-03  - [ ] P-04  - [ ] P-05  - [ ] P-06  - [ ] P-07  - [ ] P-08

Evidence:

---

## 9. Evaluation

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| E-01 | P0 | Internal eval sets | Frozen, versioned sets built from real internal tasks, covering every target use case |
| E-02 | P0 | Write protection | Agents/engineers cannot modify frozen eval sets (permission test) |
| E-03 | P0 | Baselines | Baseline results (BM25, rules, current tool) exist for every eval |
| E-04 | P0 | Beats baseline | Candidate model beats each baseline by the agreed margin |
| E-05 | P1 | Public benchmarks | Run for sanity (e.g., MMLU, HellaSwag, code benchmarks) with results recorded |
| E-06 | P0 | Contamination check | Train/eval overlap report attached and clean |
| E-07 | P1 | Human evaluation | Domain-expert panel rates a sample; inter-rater agreement recorded |
| E-08 | P1 | Red-team: memorization/PII | Extraction probes find no leakage of PII or secrets |
| E-09 | P0 | Red-team: permissions | Retrieval permission-leak tests show users cannot see documents they lack access to |
| E-10 | P1 | Regression suite | Suite runs automatically on every candidate model and flags drops |
| E-11 | P1 | Reproducible evals | Re-running an eval gives the same score (within tolerance) |
| E-12 | P1 | Independent judge | Evaluation is run by an agent/team separate from the builders |

- [ ] E-01  - [ ] E-02  - [ ] E-03  - [ ] E-04  - [ ] E-05  - [ ] E-06
- [ ] E-07  - [ ] E-08  - [ ] E-09  - [ ] E-10  - [ ] E-11  - [ ] E-12

Evidence:

---

## 10. Serving and Inference

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| V-01 | P1 | Serving engine | vLLM/TensorRT-LLM/SGLang (or equivalent) serves the model behind an API |
| V-02 | P1 | Quantization | Quantized version evaluated; quality loss within tolerance |
| V-03 | P1 | Latency SLO | p50 and p95 latency meet targets under expected load |
| V-04 | P1 | Throughput and cost | Tokens/sec and cost per 1k requests measured |
| V-05 | P1 | Load test | System holds under peak load with acceptable error rate |
| V-06 | P0 | Authentication | Unauthenticated requests are rejected |
| V-07 | P1 | Rate limiting | Per-user limits enforced |
| V-08 | P1 | Retrieval (RAG) | Embedding model plus vector index returns relevant results; Recall@k meets target |
| V-09 | P1 | Guardrails | Input/output checks block defined bad cases (tested) |
| V-10 | P1 | Logging and privacy | Request logs exist and comply with privacy policy |
| V-11 | P0 | Rollback | Revert to the previous model version in minutes (drill performed) |
| V-12 | P1 | Autoscaling / capacity | Capacity plan exists and scaling behavior is tested |

- [ ] V-01  - [ ] V-02  - [ ] V-03  - [ ] V-04  - [ ] V-05  - [ ] V-06
- [ ] V-07  - [ ] V-08  - [ ] V-09  - [ ] V-10  - [ ] V-11  - [ ] V-12

Evidence:

---

## 11. Governance, Security, and Legal

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| G-01 | P0 | Model card | Documents data sources, training setup, evals, intended use, and known limits |
| G-02 | P0 | Audit log | All side-effecting actions (jobs, data writes, deployments) logged with identity and outcome; log is immutable |
| G-03 | P0 | Approval gates | Gates exist for data access, dataset release, eval freeze, expensive runs, promotion, deployment, and budget changes; each shows a recorded approver |
| G-04 | P0 | Least privilege | Each agent/role holds only needed permissions (permission matrix tested) |
| G-05 | P1 | Sandboxing | Agents run isolated with scoped, short-lived credentials and no default internet access |
| G-06 | P1 | Secrets management | No secrets in prompts, code, or logs (scan) |
| G-07 | P1 | Prompt-injection defense | Injected instructions in ingested documents do not trigger tool actions (test with planted payload) |
| G-08 | P1 | Dual control | The proposer of an action cannot also approve it |
| G-09 | P1 | Kill switches | Per-agent, per-job, and global kill switches work (tested) |
| G-10 | P1 | Budget caps | Per-agent, per-job, and weekly spend caps enforced automatically |
| G-11 | P1 | Loop detection | Repeated failing task halts and escalates to a human |
| G-12 | P1 | Data deletion process | Request-to-removal procedure documented and exercised |
| G-13 | P1 | Security review | Training and serving environments reviewed and signed |
| G-14 | P2 | Ongoing monitoring | Drift, misuse, and quality regression monitoring in place |
| G-15 | P0 | External data handling | If external LLM APIs are used by agents, no raw data is sent (egress/log audit) |

- [ ] G-01  - [ ] G-02  - [ ] G-03  - [ ] G-04  - [ ] G-05
- [ ] G-06  - [ ] G-07  - [ ] G-08  - [ ] G-09  - [ ] G-10
- [ ] G-11  - [ ] G-12  - [ ] G-13  - [ ] G-14  - [ ] G-15

Evidence:

---

## 12. Agent Platform (the product itself)

| ID | Sev | Test | Pass criteria |
|---|---|---|---|
| A-01 | P0 | Orchestrator plan | Orchestrator turns a goal into a task graph with owners, dependencies, acceptance criteria, and cost estimates |
| A-02 | P1 | Task states | Tasks move through defined states; accepted state requires a review by a different agent or human |
| A-03 | P1 | Role agents present | Orchestrator, data engineer, ML engineer, and evaluation agents work end to end (starter set); others added as needed |
| A-04 | P1 | Shared source of truth | Repo, tracker, task board, and data catalog are used by all agents; agents exchange artifacts, not long chat |
| A-05 | P1 | Handoff contracts | Each handoff (e.g., data engineer to evaluation) has a defined artifact and acceptance check, verified in a run |
| A-06 | P1 | Reviewer pairing | Every builder agent has a separate reviewer; test shows a reviewer rejecting a bad output |
| A-07 | P1 | Human gate enforcement | Agent attempts to skip a gate are blocked (negative test) |
| A-08 | P1 | Reproducible agent config | Agent prompts, tools, and permissions are version-controlled |
| A-09 | P1 | Observability | Dashboards show agent throughput, failures, cost by agent/job, and GPU utilization |
| A-10 | P2 | End-to-end dry run | Pipeline from raw data to a small trained and evaluated model completes with only gate approvals from humans |

- [ ] A-01  - [ ] A-02  - [ ] A-03  - [ ] A-04  - [ ] A-05
- [ ] A-06  - [ ] A-07  - [ ] A-08  - [ ] A-09  - [ ] A-10

Evidence:

---

## 13. Order-of-Operations Verification

Confirm each step happened, in order, with artifacts.

- [ ] 1. Goal, constraints, and success metrics defined
- [ ] 2. Legal and data access approvals obtained
- [ ] 3. Compute secured; cluster and tracking set up
- [ ] 4. Data pipeline built; tokenizer trained
- [ ] 5. Eval sets and baselines built
- [ ] 6. Small-scale ablations run
- [ ] 7. Smoke test passed; main pretraining run launched
- [ ] 8. Checkpoints evaluated continuously
- [ ] 9. SFT then preference tuning completed
- [ ] 10. Red-team and safety behavior finalized
- [ ] 11. Optimized, deployed, monitored
- [ ] 12. Feedback loop from real usage established

---

## Scorecard

| Section | P0 passed / total | P1 passed / total | P2 passed / total | Section result |
|---|---|---|---|---|
| 1. Foundations | | | | |
| 2. Data | | | | |
| 3. Tokenizer | | | | |
| 4. Model Design | | | | |
| 5. Compute and Infrastructure | | | | |
| 6. Training Software Stack | | | | |
| 7. Pretraining Run | | | | |
| 8. Post-Training | | | | |
| 9. Evaluation | | | | |
| 10. Serving and Inference | | | | |
| 11. Governance, Security, Legal | | | | |
| 12. Agent Platform | | | | |

**Release rule:** the product satisfies the requirements only if every P0 test passes and at least 90% of P1 tests pass, with any failed P1 tests carrying a written remediation plan and owner.

**Open issues / waivers**

| Test ID | Reason not passed | Owner | Due date | Approved by |
|---|---|---|---|---|
| | | | | |

**Sign-off**

- ML lead:
- Program lead:
- Security/compliance:
- Date:
