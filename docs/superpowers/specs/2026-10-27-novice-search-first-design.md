# Novice search-first mode — design

Intent: add a second door into Internal-AI for people who want to build AI but don't know where to start. They paste a folder path and get working search over their own data. The expert control plane stays unchanged.

Why: today the path is source approval, build, split, tokenizer, train-from-scratch, frozen eval, promote, ask. That is correct for governance but asks a novice to understand pair modes, vocab size and MRR before they get value. STATUS.md shows the honest failure: 98 real records / 25k tokens loses to BM25 and refuses promotion. A novice reads that as "it doesn't work" rather than "keyword search is the right answer at this size."

## Goals

- A signed-in person pastes a local folder path and gets searchable Ask with cited file:line answers without configuring anything.
- Defaults are safe and cheap: internal-only roots, PII scrub on, small capped run, BM25 always shown next to the model.
- Every run keeps lineage, budget checks, kill switch, ACL filtering and audit; refusal to promote is presented as the safety check working.

## Non-goals

- No new model family, no classifier / extractor auto-build, no multi-use recommendation.
- No change to promotion rules, no auto-approval of gates, no background tuning loops.
- No public-internet exposure, no SSO, no new compute backend.

## Anti-goals

- Failing to look like a failure when keyword search is the right answer would make this a failure even if it technically works.
- Guessing a use-case the data cannot support, or hiding cost and data-size limits, would make this a failure.
- Adding a parallel training stack outside `workflows.py` would make this a failure.

## Constraints

- AGENTS.md non-negotiables stay binding: random initialisation, no pretrained weights, no distillation, no raw records to external APIs, frozen eval sets immutable, gates human-only with requester-can-never-approve, append-only audit.
- Console stays server-rendered Jinja + vendored htmx, no CDN, no inline script or style, SameSite=Strict, palette test stays green.
- Sources stay inside configured data roots; weight files and secret-bearing folders stay refused.

## Approaches considered

- A: Search-first default (picked). Scan, build, always land on retrieval + Ask. Cheapest, never guesses wrong, but won't serve someone who needed triage.
- B: Scan proposes two or three one-click plans from data shape. Higher hit rate for the cost of one extra choice. Dropped for now to keep the promise of just-paste-a-path.
- C: Tiny probes for each use then recommend from eval numbers. Most magical, heaviest on compute and budget plumbing. Dropped as too much for a first cut.

They picked A because it keeps the just-paste-a-path promise while avoiding building the wrong tool silently; B stays a natural follow-up.

## What they decided

- Input is a folder path only. No task picker, no hyperparameter fields.
- Output is always search: released tree or corpus dataset, Ask page working, BM25 baseline visible, model result only promoted behind the existing gate.
- Small-data honesty is a feature: when the model loses, the page says keyword search wins at this size and what volume would change that.

## What is left to the builder

- Exact Data-page flow and copy, default caps and thresholds, where the scan summary and Ask entry live in the console shell.
- Reuse of `data/inventory.py`, `data/pipeline.py`, `workflows.py` build/split/tokenizer/train/evaluate and `services/jobs.py`; no new training logic.
- Tests: denial coverage for roots, weights, secrets, gates, ACLs and budget, plus the e2e novice path on the synthetic corpus.

## Builder check (self-read)

- Answered already: services to reuse, governance that cannot change, console stack and safety rules.
- My calls: keep it to search only for v1; surface BM25 win as success; put thresholds and copy in the plan for review.
- Theirs: none open that changes what gets built. Copy tone and exact default caps can be settled in review.
