# 41 — An adapter must beat the base it sits on

Step 6 of `docs/yc-plan/RFS-READINESS.md`.

## What was wrong

`evaluate_alpha` scored an adapter's answers against a frozen set and said
`passed: True` when every citation stayed on its file. The un-adapted base was
never run, and `baseline_metrics` was an empty dict on every report. So the one
claim the RFS makes — "improve them on their own data" — had no measurement
behind it: an adapter that changed nothing, or made answers worse, produced the
same green report as one that helped, and `serve_alpha` would promote it.

## What changed

`evaluate_alpha` now takes the base's answers for the same frozen set as a
required argument, scores both with the same function, and records the
comparison three ways:

- `metrics["citation_pass_rate"]` — the adapter
- `baseline_metrics["citation_pass_rate"]` — the un-adapted base
- `checks["beats_base"]` — whether the adapter kept strictly more citations,
  with both rates, the base's own mismatches, and a sentence saying which of
  four situations occurred.

`passed` is still `all(checks)`, so a report that loses to, matches, or merely
duplicates its base now fails, and `promote_model`/`serve_alpha` follow it:
`latest_frozen_pass` stops finding a passing report. `serve_alpha` also checks
`beats_base` directly, so a passing report written by some other path before
this rule existed cannot serve an adapter.

The four situations are named rather than collapsed into "not better":

| Base | Adapter | Note |
| --- | --- | --- |
| more | fewer | "loses to its base is not an improvement" |
| same count | same count | "a set that cannot tell the two apart cannot promote" |
| identical answers | identical answers | "the adapter was not loaded" |
| fewer | more | passes |

Equal counts are refused on purpose. If the adapter cannot beat the model it
was layered onto, it carries cost and a second artifact for nothing; and when
both sit at 1.0 the useful signal is that the eval set is saturated, not that
the adapter is good.

## Why the comparison is not `baseline_kind="base_model"` on the harness

The plan proposed adding a retrieval baseline to `EvalHarness`. An adapter
changes what a runner generates, not what retrieval returns — the BM25 and
embedding baselines in `evaluate_embedding_model` measure a different model
entirely. Putting the base comparison in the alpha report is where the answers
actually differ.

The platform still holds no generator: both answer sets come from the on-prem
runner. What this change guarantees is that the comparison cannot be skipped —
not that the runner is honest about which weights produced which answers. That
limit is unchanged from before and is visible in the report, which carries the
base's mismatches as well as the adapter's.

## Tests

`tests/test_alpha_vs_base.py` — losing, matching, identical, missing and
wrong-length base answers, a pre-existing passing report that never asked the
base, and a winning adapter promoted with both rates on its card. Four existing
calls in `tests/test_alpha_track.py` were updated to supply the base's answers.
