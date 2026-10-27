# Model-type suggester — design

Intent: after a folder is inspected, the console should say what is worth building
from it — search over it, decide from it, or generate from it — in that order when
more than one fits, with a one-line reason each. The person accepts a suggestion or
picks another. This is the guidance a novice is missing today: nothing in the product
currently tells them that a 98-record folder should be a search index rather than a
generative model.

Why now: the guided-search path ships a single door (paste a folder, get search).
`guided_register` hardcodes the purpose as "guided search over this folder", and the
six guided steps assume a search dataset throughout. STATUS.md records the honest
outcome at real size — 98 records / 25k tokens loses to BM25 on retrieval and refuses
promotion — which a novice reads as failure rather than as the right answer at that
size. A recommendation up front turns that into guidance instead of a dead end.

## Goals

- After Inspect, the page ranks the model types this data can support, best first,
  each with a plain reason and what it would need if it is not ready.
- The ranking is computed from what the scan already measured. No new model, no new
  state machine, no background work, no extra compute.
- The same recommendation is shown on the guided progress card once a dataset exists,
  re-read from dataset stats so it stays honest as the data changes.
- Accepting a suggestion follows the existing path unchanged: same `register_source`
  checks, same gate, same approval by someone other than the requester, same audit.
- Small-data honesty stays a feature: "keyword search wins at this size" is a
  recommendation, not an error.

## Non-goals

- No auto-build of a classifier or a generative model in this cut. The suggestion
  says what a decide or generate run would need; it does not start one.
- No new capability, permission, gate type or task state unless the chosen type is
  persisted (see the open question).
- No hosted or pretrained anything. AGENTS.md non-negotiables are untouched.
- No change to promotion rules, budget caps, ACL filtering or the kill switch.

## Anti-goals

- Recommending a model the data plainly cannot support, without saying why, would
  make this a failure even if it renders correctly.
- Saying "generate" because a folder is large, when its text is duplicated boilerplate,
  would be worse than saying nothing.
- Adding a parallel training path outside `workflows.py` would be a failure.

## Signals available today

The scanner already measures enough to rank without new instrumentation:

- `services/build.py:SourceInspection` — `kind` (tree or file), `files`, `text_files`,
  `weight_files`, `secret_hits`, `sensitivity`, `problems`. This is what the page holds
  right after Inspect.
- `formats` by suffix, counted during the walk the scan already does. `.csv` / `.tsv` /
  `.jsonl` present means rows exist; mostly `.md` / `.txt` means documents.
- Dataset stats once built — `record_count`, `token_count`, and the per-step output
  in `lineage["steps"]` / `stats["steps"]`, which show how much survived dedup and the
  quality filter.
- `data/tree.py:TreeIndex.stats()` — files, bytes, lines, formats for a tree source.

## Ranking rules

Deterministic and testable, in the shape the repo already uses (pure function over
measured inputs, no clock, no randomness):

- **Search** — ready whenever there are usable text files at all. Ranked first below a
  few hundred usable records, where STATUS.md shows retrieval beats a from-scratch model.
- **Decide** — ready only on evidence of the answer column: a tabular file whose first
  row carries a column named for a label (`label`, `intent`, `category`, `status`,
  `priority`, and the rest of the hint set in `suggest.py`). Read from the first row
  only, from content the scan already holds, so it costs no extra reading. A table
  without a label column stays visible but ranks *below* search: it can be searched
  today, and it could be a classifier once someone adds the answer column.
- **Generate** — ready only on volume and clean long-form text that survived dedup.
  Not ready below that bar, with the reason naming the bar.

Each entry carries `kind`, `rank`, `ready`, `why`, and `needs` (what would make an
unready type ready). Unready types stay visible so the person can see the path, but
they are not the recommendation.

A single-file source (`kind == "file"`) is judged by its own shape like any other: a
labelled `.csv` favours Decide, a document favours Search.

## Where the recommendation is computed

Two places, from two different sources, and the difference matters:

- Straight after Inspect, from the in-memory `SourceInspection` the worker just
  finished. That result is held in the `inspections` dict in `console/app.py`, which is
  per-process and short-lived.
- On the guided progress card, which runs in a later request and may hit a restarted
  server. That one must recompute from `data_sources.scan`, the persisted JSON copy of
  the same inspection written by `register_source`, plus the dataset stats for
  `record_count` and `token_count`. It must not depend on the in-memory run surviving.

This is why the rules take a plain summary rather than an inspection object.

## Approaches considered

- **A: Rule-based suggester (picked).** Deterministic rules over the scan and dataset
  stats. Predictable, testable, no new capability, fits the determinism rule and the
  audit story. Cannot write prose as fluently as an agent.
- **B: Agent-written recommendation.** Reads the inspection summary and writes a free
  plan suggestion. Better on edge cases, but nondeterministic, harder to test, and it
  needs a new capability and permission so it cannot overreach. Left as a later swap
  behind the same interface if the rules read too rigidly.

They picked A for the first cut.

## What is left to the builder

- Where the rules live: a pure module taking the inspection summary (and, when present,
  dataset stats) and returning the ranked list. No I/O, so it is testable directly.
- Exact thresholds for "few hundred records" and the generate bar, chosen to match the
  numbers already quoted in the guided copy rather than invented fresh.
- How a tabular header is sampled without reading whole files, reusing the existing
  size caps in `inventory.py`.
- Where the card sits in `sources.html` and `_inspection.html`, and how it reads on the
  guided progress card. Existing classes only; the palette test stays green.
- Reading the summary back out of `data_sources.scan` for the guided card, without
  widening that stored JSON's shape in a way that breaks older rows.
- Tests: ranking rules against synthetic shapes (prose, labelled CSV, large clean
  corpus), plus a case asserting a prose folder never recommends generate, and the
  existing guided e2e path still passing unchanged.

## Builder check (self-read)

- Answered already by the repo: reuses `SourceInspection`, dataset stats, `guided_register`
  and the existing approval path; no new governance.
- My calls: advisory-first for v1; rules in a pure module; keep unready types visible.
- Theirs: one — see below.

## Decided

Advisory-first, approved. The suggester recommends and explains; only search is
actionable in this cut. Picking Decide or Generate is read, not run — the card says what
each would need. Wiring those builds stays a separate, later request: Decide needs a
labelled source and a console route onto the existing `workflows.train_classifier_model`,
and Generate needs the volume bar met.
