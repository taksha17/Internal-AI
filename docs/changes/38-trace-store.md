# 38: A trace store that keeps the failure and not the secret

## Why

`log_request` records the endpoint, the status, the latency and a SHA-256 prefix
of the question. That is the right default for a private deployment, and it is
also why one clause of the product cannot work: "production failures into
improvements" needs the failure, and after the request ends there is nothing
left but a hash. `feedback.py` can file a bad citation, but it needs a person to
notice the bad answer and type the right line number, so the loop depends on
somebody's clipboard rather than on the platform.

The fix is not to start logging raw text. It is to store the smallest thing that
makes a failure actionable, scrubbed before it is written and tagged with the
groups that may read it.

## What changed

- A `traces` table: question, answer, citations (path and line, no excerpts),
  retrieved document ids, model, user, the user's groups, latency, status,
  failure reason, redaction counts, and an expiry.
- `serving/trace.py` scrubs with the same detectors the corpus scrubber uses,
  **before** the insert, and records how much it removed so a reader can see
  that a redaction happened rather than silently reading a mangled question. A
  13-to-19 digit run counts as a card number whether or not it passes Luhn:
  over-redacting an order number costs a note, under-redacting a card number
  costs an incident.
- A trace inherits the `acl` of the principal that caused it, so the store is
  shared but a trace from `engineering` is unreadable to `finance` unless they
  share a group. `platform-admins` keeps its superuser read, as it does for
  documents.
- The audit event keeps the privacy property: it carries the trace id and a
  question hash, never the text.
- `triage()` returns the queue bucketed by why the request failed —
  `citation_mismatch`, `guardrail_block`, `no_evidence` — and an unrecognised
  reason is stored as no failure at all rather than inventing a bucket.
- `purge_expired()` deletes past the 30-day retention.
- `/v1/answer` writes a trace on both outcomes: the withheld answer (422,
  guardrail) and the returned one, which is tagged `citation_mismatch` when the
  citation check failed.

## Verification

`tests/test_trace_store.py`:

- an email and a card number in the question and the answer are redacted in the
  row that was actually written, with counts;
- the audit event contains a hash and not the question;
- the trace is visible to its own group and to `platform-admins`, and invisible
  to another group;
- a citation failure and a guardrail block land in their own buckets, and a
  clean request lands in neither;
- an unknown failure name is stored as no failure;
- retention removes everything past expiry;
- `scrub` reports what it removed.

`ruff check .` and `ruff format --check .` stay clean.

## Not changed, and not yet done

- `/v1/search` and `/v1/embed` still audit without tracing: neither can produce a
  citation failure, and a search trace with no answer would add volume without
  an action.
- There is no console page yet. The read API is `list_traces` and `triage`, so
  the queue can be built without touching storage, but a page needs an action in
  `services/permissions.py` and that is a governance change with its own denial
  tests. It should be its own commit.
- A trace is not yet promotable into an eval case or an SFT example. That is the
  next step, and it must create a new version rather than edit a released
  dataset or a frozen set.
