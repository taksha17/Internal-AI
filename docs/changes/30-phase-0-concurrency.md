## What failed

`docs/yc-plan/PLAN.md` Phase 0 lists three defects that the README documented
but the code still had:

- **0.2 Budgets were a soft cap under concurrency.** `BudgetManager.check` and
  `record_spend` were separate statements, so a burst of callers all passed the
  check before any of them wrote spend. A probe of 24 threads against a $0.60
  agent cap allowed 16 and recorded $1.60.
- **0.3 A signoff record was not unique.** Eight simultaneous decisions by one
  person left duplicate `gate_decisions` rows (up to seven for one approver on
  one gate). No gate was decided by one person wearing two roles, because
  distinct approvers were still counted, but the record said a person had signed
  off more times than they had.
- **0.4 Serving re-embedded the corpus per query.** `RetrievalService.search`
  embedded every visible document on every request, so a corpus of any size made
  `/v1/search` unusable.

## Change

- `spend_reservations` is a new table holding a place for money that is about to
  be spent. `BudgetManager.reserve` checks every scoped cap and writes the hold
  inside one `START IMMEDIATE` transaction, so two callers cannot both take the
  last of a budget. `settle` turns a hold into the usual scope rows (global,
  project, agent, weekly and job) and deletes it; `release` gives the room back;
  a hold that is never settled expires after 15 minutes. `check` and the cap
  comparison count settled spend plus live holds, so a granted hold is visible
  to the next caller. `record_spend` is unchanged for callers that already know
  the cost, and remains the one path for writing spend.
- `gate_decisions` gets a unique index on `(gate_id, approver)`. `init_db`
  removes pre-existing duplicate rows (keeping the earliest, which is the
  decision that counted) before it creates the index, so an existing database
  upgrades instead of failing. `GateService.decide` records the decision and
  updates the gate in one transaction and reports a duplicate as
  `GateError("<person> has already decided gate <id>")`, which the console and
  the CLI already render as a refusal.
- `RetrievalService` caches the document matrix per dataset, keyed by the
  released `content_hash`, the record count and the embedder's identity
  (class, `name`, `max_seq_len`). Every record is embedded, not only the visible
  ones, and the caller's ACLs are applied to the ranking on each request, so the
  cache cannot widen what a principal reads. A new dataset version or a
  different model misses the key and re-embeds. The serving app names its
  embedder after the checkpoint file so two models cannot share vectors.

## Proof

- `tests/test_budget_reservations.py`: 24 threads against a $0.60 global cap
  grant exactly 12 holds and settle $0.60; a released hold returns its room; a
  hold settles once; an expired hold stops counting.
- `tests/test_gate_decision_uniqueness.py`: eight simultaneous decisions by one
  person leave exactly one row and one success; the table refuses a second row;
  two-role gates still need two people; the duplicate is a `GateError`.
- `tests/test_retrieval_cache.py`: 1,000 records are embedded once across two
  queries; a warm query answers in under 100 ms; a contractor sees only
  permitted documents after an admin warmed the cache; a changed content hash
  re-embeds.
- The suite was run on the 16 GB Windows test machine (`D:\Internal-AI`),
  because the Linux workstation used for the edit cannot hold a full run. The
  12 new tests pass there, and `ruff check .` and `ruff format --check .` are
  clean (219 files). That machine reports 18 failures in release, PDF and
  scheduler tests; every one of them fails identically on a pristine `HEAD`
  checkout there, so they come from that environment, not from this change.
  The Linux runner was green at 442 passed before this change and collects 454
  with it.

## Out of scope

- Spending paths that already know their cost still call `record_spend` after
  `check`, as before. Converting the LLM adapter and the trainer to
  reserve-then-settle would need each caller to hold a reservation across an
  await, which is a larger change than closing the cap breach.
- The cache is per serving process, not a persisted index. A restarted process
  re-embeds once; SQLite-backed vectors are a different change.
- Phase 1 and later in `PLAN.md` are untouched. They need an owner decision on
  non-negotiable 1, model hardware, and licence review.
