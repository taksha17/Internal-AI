# What next — design

Intent: when a run does not pass, the platform should say what would change
that, using the numbers the run already produced.

Why: an honest refusal is not a next step. STATUS.md records the pilot losing
to BM25 (MRR 0.18 vs 0.06) and refusing promotion, and section 10 of the PRD
says the path forward is volume and compute, not more training loops — but none
of that reaches the person looking at a failed report. A novice reads "failed"
and stops, or trains again with different settings and gets the same answer.
The suggester already gives advice before a build ("about 500 usable records");
nothing gives the matching advice after a run.

## Goals

- A failed evaluation states the reason in the report's own numbers and what
  would change the outcome.
- The advice is deterministic: the same report and the same dataset always give
  the same sentences.
- The before advice and the after advice agree, because both read the same
  floor from `data/suggest.py` rather than two different constants.
- Nothing new can be triggered: this only explains. No build, no gate, no
  promotion, no spend.

## Non-goals

- No agent-written explanation: an evaluation agent reading the report and
  composing prose would need its own capability and would stop being testable.
- No automatic remedy. The platform does not go and fetch more data, retrain on
  its own, or promote anything.
- No change to evaluation thresholds, promotion rules, gates or budgets.
- No change to the ask page's existing small-data copy, which is already honest
  and speaks about the dataset rather than about a run.

## Anti-goals

- Encouraging another training run when the honest answer is "this corpus is too
  small" would turn a correct refusal into a loop.
- Inventing a cause the numbers do not support — a guess dressed as a finding —
  would be worse than silence.

## Signals available today

- `domain.EvalReport` — `metrics`, `baseline_metrics`, `checks`, `passed`,
  `notes`. This is what the Models page already renders.
- The dataset behind the model — `record_count`, `token_count`, and the pipeline
  steps in `lineage["steps"]`, which name what dedup and the quality filter
  removed.
- The classifier report from `workflows.evaluate_classifier_model` —
  `accuracy`, `baseline_accuracy`, `min_accuracy`, `per_class`,
  `label_distribution`, `passed`, `notes`.
- `data/suggest.py:GENERATE_RECORDS` — the volume floor the suggester already
  uses, and the reason it prints for a small corpus.

## Approaches considered

- **Rules over the numbers (picked).** Small, deterministic, testable, and it
  reuses the floor the suggester already states.
- **An agent-written explanation.** Better prose on awkward cases, but
  nondeterministic, needs a new capability to be triggered, and cannot be
  asserted in a test.
- **Extend the suggester card to cover this.** Dropped: that card answers "what
  should I build from this data", and a just-failed run needs "what would change
  this result". Two questions, two answers.

## Decisions

- The rules live in one pure module that takes a report and the numbers around
  it, and returns short sentences. No I/O, so it is tested directly.
- Advice is shown for runs that did not pass. A passing report needs no advice;
  the promotion step is already on the card.
- Retrieval advice turns on volume: below the floor it says keyword search wins
  at this size and names the floor; at or above it, it says the gap is likelier
  to be duplicate or boilerplate text or training budget, and points at the
  dataset's own dedup and quality steps.
- Classifier advice turns on the labels: no better than the majority class, or
  under the floor, plus the support of the least-represented label, which is
  what a person can actually act on.
- Every sentence names the number it rests on. Where the numbers do not support
  a cause, the advice says what is measurable and stops.

## What is left to the builder

- Exact wording and ordering of the sentences, and how many are shown.
- Where they sit on the Models page against the existing report block and the
  classifier runs table.
- Whether a passing report gets one line confirming what it cleared, or nothing.
- Tests: a small-corpus retrieval failure names the floor; a large-corpus one
  does not; a classifier no better than the majority class says so; the weakest
  label's support is reported; a passing report produces no advice; and the same
  inputs give the same output.

## Builder check (self-read)

- Answered already: the signals exist on the reports; the floor already exists in
  `suggest.py`; the surfaces already render the reports.
- My calls: rules not prose; advice only on failure; no new capability; leave the
  ask page alone.
- Theirs: none. Nothing here changes what gets built beyond wording.
