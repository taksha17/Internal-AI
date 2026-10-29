## What is left

The router falls back to the newest version carrying a name even when that
version is only a candidate, and `set_active` accepts any model id. The answer
endpoint has no ceiling on how many runs at once, no latency target, and no
check on the text it returns.

## Change

- `ModelRouter.resolve` serves only a promoted model, and never a candidate.
- `ModelRouter.set_active` refuses a model that is not promoted, with the same
  refusal it already gives a model that failed its newest frozen report.
- The answer endpoint holds a process-wide semaphore,
  `INTERNAL_AI_MAX_CONCURRENT` (default 5), around the runner call.
- A served model whose config sets `latency_target_ms` answers `408` with the
  measured latency and the target when the wait exceeds it.
- The returned text is scanned for password, API-key and confidentiality
  wording. Violations are returned as `guardrail_violations` and written to the
  `serving.request` audit row; the answer itself is still returned.
- `parse_api_keys` documents the group separator it actually parses, `|`,
  because commas separate entries.

## Proof

- `tests/test_serving_promotion.py` shows a candidate is never served and never
  activated, including through an active pointer, and that a promoted model is.
- `tests/test_evals_training_serving.py` shows `set_active` on a candidate is
  refused and the served pointer stays where it was.
- `tests/test_serving_guardrails.py` shows the scan flags each pattern, an
  answer naming a secret keeps the violation in its response and audit row, a
  slow answer gets `408`, and a ceiling of one keeps three answers from
  overlapping.

## Out of scope

- Guardrails warn; they do not rewrite or refuse an answer.
- The ceiling is per process, not a per-tenant quota.
