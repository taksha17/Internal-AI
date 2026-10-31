# Own Your Intelligence: what the repo has, what it needs, and the order to do it

Assessment date: 2026-10-30. Repo `main` at `9dd8211` (454 tests). Companion to
`PLAN.md`; this file supersedes its Part 1 where the two disagree, because this
one was written after running the code.

## 1. What I measured, not assumed

| Check | Result |
|---|---|
| `python -m internal_ai.cli --root var_check demo` | exit 0, 13/13 tasks accepted, gates cleared, **audit chain verified (279 events)**, spend $21 of $500 |
| `pytest` (Windows test machine) | **454 passed, 0 failed**, 104 s |
| `ruff check .` / `ruff format --check .` | clean |
| Source size | 93 Python files, 25,820 lines under `src/internal_ai/`; 72 test files; 31 change records |
| torch | 2.14.1+cpu, `cuda.is_available()` False |
| `internal_ai.training.alpha` | imports; three commands (`fetch`, `smoke`, `train`) |

The control plane genuinely works. Gates, budgets, the hash-chained audit log,
lineage, ACL-aware retrieval and the console are real and tested. Nothing below
disputes that.

## 2. The RFS, sentence by sentence

Jon Xu's ask, split into its four claims, against what the code does today:

| The RFS says | The repo today | Verdict |
|---|---|---|
| "Open-weight models are now competitive" | Main track forbids pretrained weights; only `training/alpha/` may load one | **Inverts the thesis.** The product's centre of gravity is the opposite of what YC describes |
| "Improve them on their own data" | `alpha/train.py` adapts rank-8 `q_proj` on layers 0-1, 4 sequences, `max_seq_len` 64, hand-rolled autograd, no PEFT | **Proof of concept, not adaptation.** Cannot be sold as fine-tuning |
| "Run them where they choose" | Offline by default, `INTERNAL_AI_OFFLINE` gate in `config.py`, `fetch.py` refuses downloads; but no Dockerfile, no Helm, no air-gap bundle | Good instinct, no artifact |
| "Decide for themselves when they change" | Rollback is a pointer change; `ModelRouter.set_active` audits it | **Real, and rare.** Keep selling this |
| "Turn workflows into evals" | Frozen eval sets, gates, `brief.py` thresholds, harness with contamination/PII checks | **Half there.** Evals are hand-authored; no capture of real workflows or traces |
| "Expertise into training data" | `supervision.py`, `preferences.py`, `build_pair_dataset` from internal text only | Partial; no SFT builder, no expert annotation workflow |
| "Production failures into improvements" | `feedback.py` files failed citations **by hand**, exports an unreleased dataset | **Manual loop, no trace store.** `log_request` stores `query_hash`, never the question |
| "Carry all of it forward with each new base model" | One hardcoded Qwen3-4B pin, Qwen-specific layer code | **Absent.** The single most valuable missing capability |

## 3. Ten findings from reading the code

Ordered by how much each one hurts the pitch.

1. **Licences are decorative.** `PinnedBase.license` is a string nothing reads,
   and `available_base()` hands back the research-licensed 3B pin the moment
   the Apache-2.0 4B pin fails to verify. `tests/test_alpha_track.py` covers
   registration, the gate-before-serve path and evaluation on a frozen set, but
   nothing covered a base's licence or the fallback until change 32.
2. **Adapter training is ungoverned.** `register_alpha` in `alpha/promote.py` is
   properly governed (`require(...)`, `Action.REGISTER_MODEL`, audit, promotion
   gate) — but `train_prefix` is a bare function, and
   `python -m internal_ai.training.alpha train` calls it directly. No run row,
   no `reproducibility.json`, no seed/commit lineage, no budget check, no
   scheduler. The artifact is audited after the fact; the act that produced it
   is invisible.
3. **A research-licensed model is a silent fallback.** `pin.py` sets
   `PRIMARY` Qwen3-4B as Apache-2.0 and `FALLBACK` Qwen2.5-3B with
   `license="qwen-research"`; `available_base()` catches the primary's failure
   and returns the fallback without a word. For a product whose pitch is
   "you know exactly what you own", silently serving a research-only licence
   is the worst possible bug class.
4. **No licence classes, no allowlist.** `PinnedBase` carries a `license`
   string and nothing interprets it. There is no `permissive`/`restricted`/
   `research_only` classification, no exception gate, no policy setting.
5. **No Hugging Face or PEFT dependency.** `grep` for `transformers`, `peft`,
   `vllm`, `bitsandbytes` finds only `PLAN.md`. The `alpha` extra is
   `torch + safetensors`. Qwen layers are reimplemented by hand in `qwen.py`,
   correct for one architecture and one architecture only.
6. **Serving is not OpenAI-compatible.** Six routes: `/health`, `/v1/models`,
   `/v1/search`, `/v1/embed`, `/v1/models/{name}/rollback`, `/v1/answer`. No
   `/v1/chat/completions`, no SSE streaming. Note the direction: `llm.py` is a
   *client* of a private runner at `INTERNAL_AI_LOCAL_RUNNER`, so today the
   platform borrows someone else's inference server rather than being one.
7. **No trace store by design.** `log_request` and `RetrievalService.search`
   record hashes, counts and latency, never text. That is the right privacy
   default and it is also why "production failures → improvements" cannot
   work: the failure is unrecoverable after the request ends.
8. **The feedback loop is one person's clipboard.** `file_feedback` requires
   `data_owner` and takes `question`, `expected_file`, `expected_line`;
   `export_feedback` builds an unreleased dataset named `ask-feedback`. Good
   bones, but it depends on somebody noticing a bad answer and typing the right
   line number.
9. **Evals compare only against keyword search.** `evaluate_embedding_model`
   reports `metrics` and `baseline_metrics` against BM25. For an adapter the
   only comparison that matters is **the un-adapted base**, and there is no
   `baseline_kind` for it.
10. **The from-scratch track cannot be the headline.** The README's own scale
    run reached recall@10 0.000 and was refused promotion; the pilot found
    25k tokens loses to BM25 (MRR 0.18 vs 0.06). Those refusals are the
    harness working, but they are also the ceiling: 25k-290k tokens does not
    train a language model that beats a keyword index.

## 4. The plan, in dependency order

Each step is one branch, one change record, failing test first, `pytest` +
`ruff` green. Days assume one engineer with an agent.

### Step 1 — Licence-aware base registry (3 days)

Extend `pin.py` into `training/bases/` with `ModelSource`: repo id, revision,
shard hashes, SPDX licence, `license_class`, parameter count, architecture,
tokenizer hash. Default allowlist `Apache-2.0`, `MIT`; anything else needs a
`base_model_exception` gate (legal + program lead). Replace the silent fallback:
`available_base()` must raise and say which class blocked it.

Acceptance: a `research_only` base is refused without the exception gate; a
`scratch_only` install refuses every adapter command; the refusal names the
licence class. New capabilities go in `roles.py` and get a denial test per
`AGENTS.md`.

### Step 2 — Provenance mode (2 days)

The `policy` setting `scratch_only` (default) | `open_weight_allowed`, decided
through a `policy_decision` gate, printed on every model card and written to
the audit log on every load. **Blocks on the owner-approved rewording of
`AGENTS.md` non-negotiable 1 and `PRODUCT.md`** — approved in conversation
2026-10-30, not yet written into either file.

Acceptance: `scratch_only` makes every command in `training/alpha/` and
`training/adapt/` refuse; `open_weight_allowed` requires the gate; the
from-scratch track keeps zero paths that read pretrained parameters, and a test
asserts it by walking the import graph of `training/model.py`.

### Step 3 — Put adaptation under the control plane (4 days)

Before adding capability, close finding 2: an adapter run becomes a job.
`adapt/train.py` runs through `compute/scheduler.py`, records
`reproducibility.json` (commit, config, dataset hash, base hash, seed),
`BudgetManager.reserve`/`settle`, and an audit event. Reuse the trainer's
divergence guard.

Acceptance: killing the process mid-run leaves a job row that says so; the
adapter card carries commit + seed + base revision + dataset hash; spend shows
under the job scope.

### Step 4 — Real adapter backend (1 week)

`adapt/` with a `backend` interface: `peft` LoRA/QLoRA on Hugging Face
`transformers` behind an optional extra (`[adapt]`), keeping `qwen.py` as the
`tiny_gpu` fallback for 4 GB cards. CI fixture is a tiny Apache-2.0 model
stored by hash on disk — never downloaded at test time.

Acceptance: a CPU smoke run of 20 steps on the tiny pinned model decreases loss
and reproduces byte-identically from the seed; a second architecture (not Qwen)
trains through the same interface; `scratch_only` refuses both.

### Step 5 — SFT data builder from internal material (4 days)

Tickets, documents and expert Q&A → instruction pairs with provenance, PII-safe
by the existing scrubber, and a hard refusal for any record whose `source` is a
third-party model. Grow `data/supervision.py`; do not add a second path.

Acceptance: a `source=third_party_model` record is refused with a named reason;
a released dataset produces a pair set whose every row carries `record_key`
lineage.

### Step 6 — Evaluate an adapter against its base (3 days)

`harness.evaluate_retrieval` gains `baseline_kind="base_model"`. Promotion of an
adapter requires beating the un-adapted base on the same frozen set, and the
report shows adapter / base / keyword baseline side by side.

Acceptance: an adapter that does not beat its base is refused with the three
numbers printed; an adapter that does is promotable only through the existing
gate.

### Step 7 — Trace store and triage (1 week)

A `traces` table holding the question, the answer, citations, retrieved
document ids, latency, model and user, **PII-scrubbed and ACL-tagged at write
time**, with a retention rule. Triage queue in the console for failed
citations, guardrail blocks, thumbs-down and low-confidence retrieval. Keep the
privacy stance: a trace never stores text from a document the caller could not
read.

Acceptance: a trace from a restricted group is not readable by a principal
without that group (test it, do not assume it); scrubbing runs before insert; a
failed-citation request lands in the queue within one request.

### Step 8 — Promote a trace into a new version (5 days)

One-click from triage to (a) candidate eval case, (b) SFT example, (c)
preference pair — each behind an expert approval, each creating a **new**
dataset or eval-set version with lineage back to the trace id. Reuse
`feedback.py`'s export pattern; never edit a released dataset or a frozen set.

Acceptance: promotion creates version *n+1* with `lineage.trace_ids`;
attempting to mutate a frozen set raises; a regression set grows by one case and
runs on the next candidate.

### Step 9 — Rebase and the rent-vs-own report (1.5 weeks)

A `Recipe` = (dataset versions + hashes, adapter config, eval set ids, brief
thresholds). `rebase` runs adaptation on the new base, then evals, then emits
four columns: old base+adapter, new base+adapter, un-adapted new base, rented
API baseline. Promotion gate opens only if the brief's metrics pass; the old
model keeps serving until it is approved; rollback is one command.

Acceptance: the same recipe on the same base reproduces the adapter within
tolerance; the report refuses to propose promotion when any brief metric fails;
a completed rebase leaves a verified audit chain and a one-command rollback that
is tested.

### Step 10 — OpenAI-compatible serving (3 days)

`/v1/chat/completions` (SSE streaming) and `/v1/embeddings` in
`serving/app.py`, reusing `Principal`, `RateLimiter`, `log_request` and the ACL
filter. The official `openai` client must work by changing `base_url` only.

Acceptance: a client request streams tokens, is audited, respects the rate
limit and the model router, and fails closed when the model is out of service.

## 5. What I would cut, and why

- **Do not make the from-scratch language model the product.** Keep it as the
  `scratch_only` mode and the home of vertical small models later (that is the
  *other* RFS). Leading with it invites the comparison the README already lost
  at 25k tokens.
- **Do not build a chat UI.** The valuable surface is the console's evidence:
  gate, report, comparison, lineage.
- **Defer** Postgres, SSO, Helm, MLflow, distributed training and container
  sandbox until a design partner asks. They are Phase 5-6 work and none of them
  is in the RFS paragraph.
- **Retire or gate `FALLBACK`.** A research-licensed base reachable without a
  gate contradicts the entire pitch.

## 6. Decisions still needed from the owner

1. Reword `AGENTS.md` non-negotiable 1 and `PRODUCT.md` into provenance mode
   (approved verbally 2026-10-30; the files still say the old thing).
2. Licence and CLA position (PolyForm Noncommercial plus a commercial term
   sheet; no CLA today, so outside contributions cannot be relicensed).
3. First vertical and design partner (the demo in step 9 needs one).
4. Which bases are allowlisted, and what happens to the Qwen research pin.
5. External-LLM policy P2 and the default model string in `llm.py`.
6. Hardware: this workstation is CPU-only, so steps 4 and 9 need either the
   16 GB Windows machine for smoke runs or a rented GPU for a real adapter.

## 7. Commands used for this assessment

```bash
.venv/bin/python -m internal_ai.cli --root var_check demo
.venv/bin/python -m pytest                      # 454 passed on the Windows PC
.venv/bin/ruff check . && .venv/bin/ruff format --check .
PYTHONPATH=src .venv/bin/python -m internal_ai.training.alpha --help
.venv/bin/python -m pytest -q -k "serving or eval or preference or supervision or scheduler"
```

## 8. Second pass: corrections

Four things changed my sizing after reading the schema, the answer route and the
supervision modules. Two gaps are smaller than I wrote and one is larger.

- **The schema has no trace table.** All 29 tables: `schema_meta`, `tasks`,
  `task_events`, `gates`, `gate_decisions`, `audit_events`, `budget_limits`,
  `spend_records`, `spend_reservations`, `runs`, `datasets`, `dataset_records`,
  `models`, `eval_sets`, `eval_reports`, `policy_decisions`, `kv`, `evidence`,
  `feedback`, `preference`, `project_brief`, `alerts`, `people`,
  `source_tombstones`, `accounts`, `sessions`, `api_tokens`, `console_jobs`,
  `data_sources`. Nothing stores a question. Step 7 is genuinely new work.
- **`/v1/answer` is a cite-checker, not an answer endpoint.** It requires the
  caller to send `question` *and* `hits`, needs `INTERNAL_AI_LOCAL_RUNNER` set,
  and returns 503 otherwise. Retrieval and generation both happen outside the
  platform; what it adds is citation scoring, a memorisation probe, guardrails
  and latency enforcement. So "serve a cited answer" is not self-contained
  today, and step 10 has to include closing that, not only adding routes.
- **Step 5 is smaller than I said.** `data/supervision.py` already ingests
  instruction/response pairs and refuses `authored_by=model` and any alpha
  source; `preferences.py` already stores preference pairs, refuses a
  model-authored preference, and exports an unreleased dataset; both are wired
  to CLI commands. What is missing is a *consumer*: a trainer that turns those
  versions into adapter updates, plus the approval that attaches a version to a
  run. That is 2-3 days of wiring, not a 4-day builder.
- **`llm.DEFAULT_MODEL` is current**, `claude-sonnet-4-5-20250929`, so
  `PLAN.md` item 1.4(5) ("an older Claude model string") is stale and needs no
  work.

- **The alpha track is not untested.** `tests/test_alpha_track.py` covers
  registration, the capability check, the forged-base refusal, gate-before-serve,
  evaluation on a frozen set and the streamed LoRA step. My earlier "zero
  tests" line was wrong, produced by a search that returned nothing; the real
  gap was narrower — base licences and the fallback path, closed in change 32.

Two existing pieces should be reused rather than rebuilt:

- `brief.ProjectBrief.metric_verdicts` already reads measured numbers against
  the brief's thresholds and directions. The rebase report in step 9 should
  call it, not add a second threshold parser.
- `compute/scheduler.py` already offers `JobSpec`/`JobState`, per-job cost caps,
  quotas, priority and a pluggable backend. Step 3 is wiring adapter training
  into it, not building a scheduler.

Revised sizes: step 3 drops to 3 days, step 5 to 3 days, step 10 rises to 4
because it must also make the answer path self-contained.
