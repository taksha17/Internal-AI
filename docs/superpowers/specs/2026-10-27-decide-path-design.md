# Decide path — design

Intent: make `decide` a path a person can actually take, and make it learn the
labels in their data rather than the labels this repository happens to ship.

Why: the suggester now ranks a classifier first when a source carries a label
column, and there is nothing to click. Worse, the classifier that does exist
labels its training rows with `data/labels.py:BILLING_LABEL_SET` — keyword rules
tuned to the synthetic ticket corpus (billing / fulfilment / onboarding /
general). Exposing that unchanged would tell someone "your rows carry a label
column" and then train on keywords that ignore it. That is a misleading feature,
not a missing one.

## Goals

- A row-oriented source keeps the answer column from its own data, and the
  classifier is trained on those labels.
- The console gains a classifier panel on the Models page: pick a released
  dataset whose tokenizer and encoder exist, train as a job, and evaluate on a
  held-out split against the majority-class baseline. No new gate type, no new
  capability.
- The panel is honest about where the path ends, because a classifier is **not
  promotable** here and the code already says so. `workflows.latest_frozen_pass`:
  "A classifier run records a report too ... That report has no frozen eval set,
  so it must not satisfy promotion." The panel states it rather than implying a
  promotion that would be refused.
- Labelling prefers the data's own label and falls back to the shipped rules
  when there is none, so the CLI path and its tests keep working unchanged.
- The evaluation keeps telling the truth: a classifier that only ties the
  baseline is refused, exactly as STATUS.md records today.
- The suggester and the build agree: `decide` is only promised when the source
  shape can actually deliver row labels.

## Non-goals

- No new model family. A hashed-feature linear classifier needs no encoder and
  would likely beat the shipped rules on small data, but it is a new training
  path and a separate decision. Left out deliberately, noted below.
- No row-level labels from a folder of tables: a tree source is one record per
  file (`data/tree.py`), so a folder of CSVs cannot carry a label per row in
  this cut.
- No auto-promotion, no new state machine, no change to promotion rules, budget
  caps, ACL filtering or the kill switch.
- No pretrained weights and no third-party labels. Labels come from internal
  rows a person supplied.

## Anti-goals

- Reporting a classifier as ready when it only reproduces the shipped keyword
  rules would make this a failure even if it trains and renders.
- Silently choosing which column is the text and which is the answer, with no
  way to see or correct it, would be a failure for the person who knows their
  data.

## What the code does today

- Ingest: `data/records.py:ingest_source` reads `.jsonl` / `.ndjson` / `.csv`
  into one record per row. The text is `row[source.text_field or "text"]`, so a
  file with no column literally named `text` yields **no records at all** — a
  real `tickets.csv` (columns `ticket,description,label`) is currently ingested
  as empty. `Source.text_field` exists but nothing in the console sets it.
  `meta` carries only `format` and `extra_keys`.
- Tree sources go through `data/tree.py` and become one record per file.
- Labelling: `data/labels.py:build_labeled_dataset(records)` applies
  `BILLING_LABEL_SET` by keyword.
- Training: `workflows.train_classifier_model(platform, dataset_id,
  encoder_path, tokenizer, ...)` labels by rule, then trains a linear head over
  frozen from-scratch encoder embeddings (`training/classifier.py`), auditing
  `classifier.trained` and recording spend through the budget manager.
- Evaluation: `workflows.evaluate_classifier_model` compares against the
  majority class, floor `min_accuracy=0.6`, and reports rule agreement as a
  reference figure.
- Console: tokenizer training is a dataset step (`console/app.py`), embedding
  models train on the Models page, and the classifier has no route at all.
- `Record.fingerprint()` hashes the text only; a pipeline dataset's hash is
  `content_hash({records, config})`.

## Approaches considered

- **A: expose the classifier as-is.** Small, but trains on shipped rules and
  makes the recommendation dishonest. Dropped.
- **B: labels from the data (picked).** The answer column travels with the
  record and drives labelling, with the rules as fallback.
- **C: a console labelling/confirmation step.** Useful later for correcting a
  mapping; premature before labels are read from the data at all.
- **A hashed-feature linear classifier trained from scratch** would fit small
  data far better than a transformer head and needs no encoder, but it is a new
  model family with its own evaluation story. Not in this cut.

## Decisions

- The label travels in `Record.meta["label"]`. The `Record` schema does not
  change, so nothing that reads records breaks.
- `build_labeled_dataset` prefers `meta["label"]` and falls back to the rule set
  per record, so unlabelled corpora behave exactly as today.
- The dataset hash gains the labels **only when some records have one**, so
  every existing unlabelled build keeps its exact hash and released versions
  stay comparable.
- Lineage records the label column name and the label distribution, so an
  approver can see what the classifier was taught and how balanced it is.
- The console path needs the text column and the answer column named. Defaults
  are derived (the longest text column, the label-hinted column) and shown, and
  the person can correct them. The expert route keeps `Source.text_field`.
- The panel states the chain it requires — released dataset, then tokenizer,
  then encoder, then the head — because a classifier in this codebase is a head
  over a from-scratch encoder, not a shortcut. Hiding that would mislead.
- `data/suggest.py` narrows `decide` to a source whose records can be rows (a
  single table file with a label column). A folder of tables reports what it
  would need instead. This keeps the recommendation and the build in step.

## Later, not now

A promotion story for a task classifier needs its own design: what a frozen
set means for classification, which evidence the gate should demand, and what
"promoted" means for a head over an encoder. The existing retrieval promotion
path is deliberately closed to it, and reopening that is a governance change.

## What is left to the builder

- Where the text and label columns are chosen (scan time or build time), and
  where they are reported so a person can see them.
- Which label values are acceptable (a small closed set versus free text), and
  what happens to rows with no label or a rare label.
- The panel's copy, its disabled states, and how it links to the tokenizer and
  encoder steps that must come first.
- Tests: labels from a CSV and a JSONL reach the classifier; a row with no label
  falls back to the rules; a dataset built without labels keeps its old hash; a
  CSV with no `text` column is no longer ingested as empty and says what it
  needs; a classifier that ties the baseline is still refused; the new panel is
  denied to roles without `train_model`.

## Builder check (self-read)

- Answered already: the two-column naming problem, the tree restriction, and the
  encoder requirement come from the code above.
- My calls: labels in `meta`; conditional dataset hash; narrowing the suggester;
  no new model family.
- Theirs: one, below.

## Open question for the owner

A folder holding many tables stays one record per file in this cut, so `decide`
will not light up for it. Accept that for now — the single export file is the
common case for someone with a labelled dataset — or extend tree ingestion to
read rows out of tables inside a folder as part of this work? The recommended
answer is to accept it now: the row-level tree ingest is a data-plane change
that deserves its own design.
