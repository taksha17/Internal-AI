# 42 — A person's correction becomes a supervision example

Step 8 of `docs/yc-plan/RFS-READINESS.md`, second of its three promotions.

## What was wrong

Triage could turn a failed answer into a regression case, which stops the same
mistake shipping again. It could not turn one into *training* material, which is
the other half of "expertise into training data": `ingest_supervision` took rows
from the CLI, `file_feedback` took a file and a line number, and neither knew a
trace existed. A person who noticed a wrong answer had nowhere to put the right
one.

## What changed

`sft_example_from_trace` writes one row — the trace's question, the person's
answer — as version *n+1* of a supervision dataset named `trace-sft`, carrying
every earlier row forward so the set grows rather than being replaced. The
dataset lands unreleased; releasing it is still the gate it always was.

The trace supplies the question only. Training on the reply the model got wrong
would teach the mistake, so the stored answer is always the person's, and four
refusals keep it honest:

- an answer that is empty, or identical to the failed one, is not a correction;
- the correction must cite a line the question was actually answered from, which
  is `score_answer` against the trace's own citations, so a person cannot file an
  uncited answer that the next evaluation would then score as a failure;
- a trace that already has an example in the newest version cannot be added
  twice;
- only a failed trace qualifies. A trace whose citations passed may still be a
  bad answer, but nothing in the record says so, and inventing that signal is
  worse than leaving it out.

`ingest_supervision` gained `name`, `version` and `lineage`, and its refusals
moved into `require_supervision_row` so every carried row is checked again: a
dataset cannot gain a model-authored row just by being copied forward.

Permission is `build_dataset`, not `create_eval_set`. A regression case changes
what gets measured; a supervision example changes what a model learns, and that
is a data owner's or an ml lead's decision. A `domain_expert` may write the
correction in prose but not file it — tested.

## Console

`/console/traces/{id}/sft` is a form rather than a one-click button, because
there is nothing to click: the answer has to be written. The page shows the
question, the model's failed reply and the cited lines, and the form refuses the
same four things the service does. The triage row links to it beside the
regression button.

## Tests

`tests/test_trace_sft.py` — first version, carried-forward second version, the
four refusals, another group's trace, a missing capability, and an unknown trace
that must not leave a dataset behind.
