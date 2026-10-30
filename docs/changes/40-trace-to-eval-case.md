# 40: A failed answer becomes the next regression case

## Why

Change 38 stored the failure and change 39 made it visible, but the person who
read it could only act by retyping the question and the line number into an eval
set by hand — which is the clipboard loop `feedback.py` already had, with better
screens. The point of keeping a trace is that the platform can carry it forward
itself.

## What changed

- `serving/trace.py:get_trace` reads one trace and applies the same ACL rule the
  list does. Another group's trace is refused as a permission error rather than
  hidden as a 404, because the id exists and pretending otherwise would make the
  store look emptier without making it safer.
- `services/triage.py:eval_case_from_trace` checks `CREATE_EVAL_SET`, reads the
  trace, and appends a case to the next version of a regression set. The newest
  version is **copied forward**, so a frozen set is read and never written —
  the one thing the eval rules forbid — and the copy is verifiable: its
  `content_hash` is unchanged by the new version existing.
- The item keeps `query`, the citation lines as `hits`, `kind="citation"`,
  `failure` and `trace_ids` for lineage. `hits` is the shape
  `evaluate_alpha` already scores answers against, so the case is measurable by
  an existing evaluator rather than by one that does not exist yet.
- Refusals: a trace that passed is not a regression, a trace with no citation
  line has nothing to assert, a trace already in the set is not added twice, and
  another group's trace is out of reach.
- The console's citation section gains one button per row, and the redirect
  lands on Evals with a line saying the new set must be frozen before a model is
  measured against it.

## Verification

`tests/test_trace_regressions.py`:

- a failed trace creates version 1 with its query, hits, lineage and audit event;
- a second trace creates version 2 carrying the first case, and version 1 is
  untouched;
- a frozen version 1 is copied forward with an identical `content_hash`, and the
  new version is not frozen;
- the same trace cannot be added twice;
- a clean trace, a trace with no citations, another group's trace and a role
  without `CREATE_EVAL_SET` are each refused, with the reason named.

`ruff check` and `ruff format --check` stay clean; the console markup and safety
tests still pass with the new form.

## Not done

- The case is not frozen automatically. Freezing needs the `eval_set_freeze`
  gate and a different person, and automating that would be the platform
  deciding a gate.
- Nothing yet writes the *answer* a model gave into the case; the case asserts
  which lines the answer should have cited. Recording the failed answer as the
  thing to beat is a later step, once there is a runner that replays it.
