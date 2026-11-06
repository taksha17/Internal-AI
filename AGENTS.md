# AGENTS.md

Rules for agents working in this repository. Humans contributing a branch and a
pull request read `CONTRIBUTING.md`; where the two differ, this file wins.

What you write here changes how other agents behave, so the bar is higher than
for ordinary application code.

## What this is

The control plane a company runs on its own machines. It builds, evaluates,
registers and serves a model whose data never left the boundary, and it records
who decided what.

Two tracks:

- **From scratch.** A model trained from random initialisation on approved
  internal data only. This is the default and it has no code path that reads
  pretrained parameters.
- **Alpha.** One pinned open base that is already on disk, adapted into an
  adapter the company owns. Off unless a person decides otherwise.

Offline is the default: no outbound call unless a lab sets
`INTERNAL_AI_OFFLINE=0`. The licence is PolyForm Noncommercial 1.0.0 in
`LICENSE`; never restore Apache or any permissive grant.

Two installs exist on disk and both are real evidence, not demos:

- `var_alpha/` — a public RFC corpus taken through the governed data path and
  adapted on a rented T4.
- `var_nba/` — a 73,428-row results CSV turned into a line-addressed corpus, an
  adapter, a frozen question set and a scored run. It is the worked example of
  every rule below, including the ones that refused.

## Why the alpha track exists

The from-scratch track refuses to start without `TOKENS_PER_PARAMETER_FLOOR`
(20, in `training/trainer.py`) tokens per parameter. The NBA corpus is 865,165
tokens: 0.0865 tokens per parameter for the smallest model the roster names. The
floor refused all three sizes, and it was right to — a 10M model shown 865k
tokens has not seen enough text to be a model of anything.

The answer to that refusal is more data or a smaller claim, never a lower floor.
An install that wants a working assistant from a small corpus takes the alpha
track, decides it through a gate, and says on the model card that the base is
not ours.

## Non-negotiables

1. **A base model is loaded only in provenance mode, and the default mode loads
   none.** `scratch_only` is the default and keeps the old guarantee exactly:
   the from-scratch track has no code path that reads pretrained parameters, and
   none may be added. `open_weight_allowed` is a per-install setting decided
   through a `policy_decision` gate on `P6` and recorded in the audit log; a
   pending proposal does not count. In that mode the adapter track may load a
   base that is on the allowlist, hash-pinned, and recorded on every model card
   by repo id, revision and licence. Only `internal_ai/training/alpha/` may load
   those weights. A base whose licence class is not `permissive` needs its own
   `base_model_exception` gate. Alpha output is never training data for the
   from-scratch track. Weight files found in a data source are refused.
2. **Never train on third-party model output.** Distillation is a non-goal.
   Task data comes from internal documents, tickets and resolved questions. A
   model that wrote the questions would put third-party output in the training
   set, so training questions are built from a passage's own content words.
3. **Never send raw records to an external API or a rented machine.** Schemas,
   aggregate statistics, redacted samples, code and logs may leave; records may
   not. See PRD 6.3. A rented GPU receives the released training text, the
   frozen questions **by id**, and a hashed copy of the training step — never a
   reference answer, never a raw record.
4. **Never write to a frozen eval set.** Frozen sets are write-protected from
   every agent. A change is a new version behind a new `eval_set_freeze` gate.
5. **Never decide a gate as an agent.** Gates are human-only, the requester
   cannot approve their own, and a gate passes only when the approvers' roles
   cover every required role.
6. **Never write to `audit_events`.** It is append-only, enforced by triggers
   and chained by hash. Use `AuditLog.record`.
7. **Never weaken a refusal to make a result look better.** The floor, the
   citation scorer, the gates and the guardrails are the product. If a run is
   bad, the report says it is bad; a suppressed check is a lie with a green
   tick.
8. **Never commit a credential, a record or a client dataset.** `.secrets/`,
   `Datasets/`, `var*/` and `docs/changes/` stay out of git. Do not print a key,
   a password or a record, not even to prove it works.

## Where the guardrails live

Change the rule in its own file, never around it.

| Rule | Home | A change needs |
| --- | --- | --- |
| Who may do what | `internal_ai/roles.py` (agents), `services/permissions.py` (people) | the role that needs it and nothing else |
| Every agent action | `internal_ai/agent.py`, `internal_ai/tools.py`, `agentic.py` | capability check plus an audit event |
| Task states | `internal_ai/store.py:ALLOWED_TRANSITIONS` | the map entry is the specification |
| Spend | `internal_ai/budget.py` | every new spend path calls `record_spend` or `check`; bypassing it is a bug |
| Datasets | `internal_ai/data/` | released versions are immutable; a change is a new version with a new hash and lineage entry |
| Promotion | `internal_ai/workflows.py:promote_model` | a passing eval report against a frozen set and an approved `model_promotion` gate |
| Gates | `internal_ai/gates.py`, `GateType` in `domain.py` | a new type plus the roles that may decide it |
| Base models | `internal_ai/training/alpha/pin.py` | a licence class and an allowlist entry, never an unchecked pin |
| Provenance mode | `internal_ai/training/provenance.py`, policy item `P6` | the mode reads only a decided policy; `scratch_only` refuses every load |
| Alpha registration | `internal_ai/training/alpha/promote.py` | a released dataset, a pinned base, a licence class that allows it |
| Training pairs | `internal_ai/training/alpha/pairs.py` | one pair format, shared by training and serving |
| Citations | `internal_ai/training/alpha/citations.py` | the scorer the frozen questions were checked with |
| Traces | `internal_ai/serving/trace.py` | scrub before insert, inherit the asker's groups, 30-day retention |
| Compute advice | `internal_ai/compute/advise.py` | the floor is read from the trainer, never restated |
| Supervision sets | `internal_ai/data/supervision.py` | estimated token counts are not a licence to release |

## The alpha track, end to end

This is the order the platform enforces. Any step may refuse, and a refusal is
the answer.

1. **Pin.** `training/alpha/pin.py` names the bases this install may load, with
   a revision, a licence and a sha256 per shard. `verify_snapshot` checks size
   and hash. A pin is never added without its licence class.
2. **Decide.** `policy_decision` on `P6`, decided by `program_lead` and `legal`.
   `provenance.require_open_weights` then permits the load; nothing else does.
3. **Ingest and release.** The corpus goes through the ingest pipeline, and the
   `dataset_release` gate releases it. `scripts/nba_build.py` shows the order,
   resumably, with the brief and compute advice in front of it.
4. **Freeze the questions.** `eval_set_freeze` over items drawn from the
   held-out split. Each item is checked against the same scorer the model will
   be judged by, or it measures nothing.
5. **Stage for the GPU.** `scripts/alpha_y15_kaggle.py` writes the training
   text, the questions by id, the corpus index and the trainer modules into one
   private Kaggle dataset. Raw prose is not staged; reference answers are not
   staged. The staged copy of the trainer is hashed, because an adapter only
   counts if it came from the step this repository audits.
6. **Run.** `scripts/alpha_y15_kernel.py` trains and generates on the rented
   box. It returns what the model said, and — when the install wants to register
   and serve the adapter — the adapter bytes through a private dataset. It never
   sees a reference answer.
7. **Register and score at home.** `register_alpha` needs a released dataset
   and a pinned base; scoring happens here, against the frozen set, with the
   repo's own citation scorer.

Two packaging facts cost a run each and are now part of the ritual: a Kaggle
dataset uploads **files, not directories**, so the package travels as
`internal_ai.zip`; and Kaggle unzips it where it likes, so the loader globs for
`**/internal_ai/training/alpha/pairs.py` and derives the root from the match.
Never hardcode the unpacked path.

## A corpus of checkable lines

Retrieval answers are only honest if the citation names something a person can
open. The NBA corpus is the reference implementation:

- One game per line, id `NBA.csv:<line>`, wording fixed by a template.
- A template is deterministic, cannot introduce a fact the row does not carry,
  and costs no model. Never let a rewrite step invent detail.
- A training passage is a line plus the continuations needed to finish its
  sentence (`pairs.py:passages`). A line alone teaches "Network Working Group",
  because that is a header. Page furniture is dropped wherever it appears.
- The citation stays the line the passage starts on, because that is what a
  reader jumps to.

## Selection, not recall

Three T4 runs taught this, in order:

1. **Raw prose.** The model learned the register and answered in it. Citations:
   almost none.
2. **Keywords in, cited sentence out.** The shape of an answer, not the choice
   of one: it cited runs of consecutive line numbers from whichever file it had
   seen most.
3. **Five BM25 candidates in the prompt, copy one.** The job became choosing,
   which a 4B model can do. `CANDIDATE_LIMIT = 5` is measured, not decorative.

`prompt_for` and `prompt_for_candidates` live in `pairs.py` and are imported by
everything that asks a question — the kernel, the serving script, the console
path. A prompt retyped in a script would drift, and the drift would look like
the model getting worse. Change the prompt in `pairs.py` or not at all.

`scripts/nba_serve.py` loads the model only on the machine with the GPU and
answers over HTTP with the passages it was shown, so an answer can be checked
against its own evidence. `scripts/nba_ask.py` is the client.

## What a score has to mean

Report three numbers on the same frozen set, or the score is not evidence:

| Line | Why it is there |
| --- | --- |
| the adapter | what was built |
| its base, unadapted | what the base already did |
| keyword search picking its first hit | what a `grep` already does |

On the NBA set the run reads adapter 29/32, base 32/32, keyword 32/32. That set
is saturated: it cannot show learned skill, and the only claim it supports is
"the adapter matches its base on an easy set". Say that plainly. A saturated set
is a reason to build a harder one, not a reason to quietly report the best
column.

The model card names the base by repo id, revision and licence, carries
`trained_from_scratch: false`, and states that alpha output is not training data
for the from-scratch track. `promote.py` builds it; do not hand-write one.

## Console

`/console` is a third client of the same services as the CLI and the JSON API:
server-rendered Jinja pages, htmx vendored under
`src/internal_ai/console/static/`, no CDN, no inline script, no inline style.
The build and its safety checks are documented in `docs/phase-2-console.md`.

Sign-in is a safety surface. The session cookie and the form token are both
`SameSite=Strict`, `HttpOnly`, and `Secure` when TLS is on. A missing `Origin`
or `Origin: null` is accepted, because some browsers send `null` for a
same-site form and the cookie is what proves the post. `Sec-Fetch-Site:
cross-site` and a named foreign origin are refused. Treat a regression here as
a safety bug. The failure text is identical for an unknown account and a wrong
password.

The kill switch sits on every signed-in page and needs a typed reason.
Engaging is immediate; releasing needs a different person.

Brand lives in `PRODUCT.md`. Two invariants the tests hold: colours appear only
in the token block at the top of `console.css`, so a rule carrying its own hex
is a bug; and body text stays at or above 4.5:1 (`tests/test_console_palette.py`).

## Traces and improvement

A trace keeps the question, the answer and the citations, scrubbed before it is
stored and tagged with the asker's groups. It exists so a failed request can
become an eval case, a supervision example or a preference pair
(`serving/trace.py`, `services/triage.py`, `tests/test_console_traces.py`).
Never store a raw
question to make triage easier; the scrubbing is the feature.

## Rented hardware

Free Kaggle T4 only, and only through the staged path above.

- Credentials live in `.secrets/kaggle.json`, read through
  `KAGGLE_CONFIG_DIR="$PWD/.secrets"`; the package is driven as
  `.venv/bin/python -m kaggle` because there is no CLI on `PATH`. Never print
  the token, never copy it into a kernel.
- A kernel has no inbound HTTP. Answers come back as a downloaded file; the
  judging happens at home, with the repo's scorer.
- Poll status in a loop and respect the tool's timeout; a kernel that is still
  `RUNNING` is not a failure.
- Do not drive a browser to work around a missing CLI.
- Record the row cap a run used in the staged recipe (`train_records`). A run
  that finishes and is scored is evidence; a larger one that never returns is
  not.

## Style

- Python 3.12. Dataclasses for domain objects, `StrEnum` for closed sets, type
  hints on public functions.
- No comments unless the reasoning is not obvious; docstrings explain why, not
  what.
- Determinism where it is visible: seeded randomness, sorted iteration, content
  hashes over the fields that define an artifact.

## Checks

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python scripts/secret_scan.py .
```

A commit leaves all four green; CI runs the same list with the `alpha` extra
installed, so alpha tests do not skip there. A generated kernel or script that
fails `ruff` is not finished: wrap the long lines.

Every new capability, gate type or state needs a test showing it is **denied**
to the roles that must not hold it, not only a test showing it works. A
restricted state also needs a test that the state machine refuses the illegal
move.

## Git

- One change is one branch. The change record is a working note at
  `docs/changes/NN-<slug>.md`, kept out of git like the plans and the PRD; the
  pull request body carries what a reviewer needs.
- Agents do not push to `main`. Only the owner (`taksha17`) merges there, so an
  agent commits on its branch, pushes that branch, and opens or updates one
  pull request.
- Agent commits use the agent identity, so machine work stays distinguishable
  from the owner's:

  ```bash
  git -c user.name='Cline Agent' -c user.email='cline.agent@internal-ai.local' commit -m '...'
  ```

  Leave the repository's default identity as the owner's.
- Commit messages say why the change exists, not what the diff contains.
- A change that makes a shipped document wrong updates that document in the same
  pull request. This file included.

## Review

- A builder never reviews its own work; `roles.py:reviewer_for` names the
  reviewer for each agent role.
- `gates.py`, `budget.py`, `audit.py`, `policy.py` and `roles.py` define the
  guardrails themselves and need a second reviewer. So does anything that
  changes what a model is allowed to load, or what a score is allowed to claim.

## Where to look first

```text
src/internal_ai/            control plane, data, training, evals, serving, console
src/internal_ai/training/alpha/   the only code that loads pretrained weights
scripts/nba_*.py            the worked install: corpus, build, policy, release, score, serve, ask
scripts/alpha_y15_*.py      the RFC install: data, questions, Kaggle staging and kernel
scripts/stress.py           the scale run the README quotes
tests/                      the suite, including the refusal tests
docs/                       architecture, runbook, tester guide, console notes
```
