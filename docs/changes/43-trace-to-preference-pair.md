# 43 — A person's answer, preferred over the one that failed

Step 8 of `docs/yc-plan/RFS-READINESS.md`, third of its three promotions, plus
the permission change it forced.

## What was wrong

Two of the three promotions existed: a failed answer could become a regression
case, and a person could write a correction as a supervision example. Neither
recorded a *judgement*: this answer is better than that one. `file_preference`
could store a choice, but only the CLI could reach it, so the pair had to be
typed from memory rather than taken from the request that exposed the mistake.

## The collision, and the decision

`file_preference` requires `domain_expert`. `view_traces` was held by
`ml_lead`, `program_lead` and `compliance` only. The people allowed to express
a preference could not see the trace that contains one, and the people who
could see it were not allowed to file. Either the feature is unusable or one of
the two roles is wrong.

`view_traces` now includes `domain_expert`. It is the role that needs it: triage
is the queue of answers this person is here to correct, and traces are scrubbed
at insert and tagged with the asker's groups, so the new holder reads only
traces from groups they already belong to. The denial tests hold the edge in
both directions — a `data_owner` is still refused, and an expert in another
group still cannot read the trace (existing `tests/test_trace_store.py` covers
the group edge, `tests/test_trace_preference.py` covers both).

`domain_expert` was not added to `build_dataset`. A regression case changes what
gets measured and a supervision example changes what a model learns; a
preference changes neither, so the two do not move together, and the console
form offers both from the same written answer so the difference is visible.

## What changed

`preference_from_trace` files one `preference` row through the existing
`file_preference`, with `answer_a` the person's answer, `answer_b` the model's
failed one and `preferred="a"`. Reusing that function keeps every check that
already existed — the `domain_expert` role and the refusal of a model-authored
pair — instead of restating them.

The four refusals `sft_example_from_trace` makes moved into
`_corrected_answer`, which both promotions call, so a correction that is empty,
identical to the failed answer, uncited, or offered for a trace that passed is
refused the same way in both places.

The trace id is written to the audit log as `preference.from_trace`, and
`export_preferences` now reads those events to put `lineage.trace_ids` on the
exported set. That is deliberate: the `preference` table has no column for it,
and adding one to an append-only table for one link is worse than reading the
link from the log that already records it.

## Console

`/console/traces/{id}/sft` becomes the page for both promotions: one textarea,
two submit buttons, the second carrying a `formaction` to
`/console/traces/{id}/preference`. No script, no second page with the same
question on it, and the copy says which of the two is training data and which is
an opinion.

## Tests

`tests/test_trace_preference.py` — the row's shape, the trace id reaching the
export lineage, the expert now reading a trace in their group, a role without
`view_traces` still refused, another group refused, a lead who may read but may
not file, and the three content refusals shared with the supervision path.
