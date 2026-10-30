# 39: A page for the failed answers

## Why

Change 38 started storing scrubbed, group-tagged traces, but nothing read them.
A store with no reader is a table, not a loop: the whole point is that somebody
looks at a failed answer and turns it into an eval case or a fix, and the only
way to see one was to run SQL.

## What changed

- `Action.VIEW_TRACES` names who may read production questions: `ml_lead`,
  `program_lead` and `compliance`. Reading what a colleague asked and what the
  model answered is a capability, not a side effect of being signed in.
- `services/triage.py:triage_view` checks that capability and then asks the
  store for the queue, so the two rules stay independent: a person with the role
  still sees only traces their groups may read, because the filter is the
  trace's own ACL.
- `/console/traces` renders three sections — citation moved, guardrail withheld,
  no supporting evidence — each with the question, model, asker, time and the
  redaction counts. A refused read is a 403 page rather than a stack trace, like
  every other console refusal.
- The rail gains "Failed answers" between Inbox and Data, and the page is listed
  in `NAV` so the breadcrumb and the rail agree.

## Verification

`tests/test_console_traces.py`:

- a failed citation is listed with its question;
- an email in the question is shown redacted, with the count, and the raw address
  never reaches the page;
- a `data_owner` gets 403 and sees nothing;
- an `ml_lead` from another group gets 200 and an empty queue, because the role
  is not a key to another group's traces;
- a guardrail block and an empty retrieval land in their own sections;
- the rail links to the page.

`tests/test_console_markup.py` and `tests/test_console_safety.py` still pass: the
template carries no inline script, no inline style and no literal colour.
`ruff check` and `ruff format --check` stay clean.

## Not done

- No action on a trace yet: turning one into an eval case or an SFT example is
  the next step and must create a new version with lineage, never edit a released
  dataset or a frozen set.
- No filters or pagination. The queue is capped at 25 per bucket, which is what a
  person works in one sitting; a filter belongs here once a real deployment
  produces more.
