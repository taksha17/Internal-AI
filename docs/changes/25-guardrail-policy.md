## What is left

The output scan warned and nothing else. A deployment that would rather not
return text naming a secret had no way to say so, and a mistyped policy would
have had no way to be noticed.

## Change

- A served model's config may set `guardrail_policy` to `warn` (the default,
  unchanged) or `refuse`.
- Under `refuse`, a flagged answer is withheld: the endpoint answers `422` with
  `answer withheld by guardrail policy` and the violations, and never returns
  the text. The `serving.request` row records status `422`, the policy and the
  violations.
- Under `refuse`, clean text is answered normally, with an empty
  `guardrail_violations` list.
- Any other value is refused closed with `500` and names the accepted values,
  rather than quietly warning.

## Proof

- `tests/test_serving_guardrails.py` shows the default still returns a flagged
  answer with the violation on record, a refusing model withholds one and logs
  the `422`, a refusing model still answers clean text, and `Refuse` (wrong
  case) fails closed with `500` and no text.

## Out of scope

- The policy is per model version, not per project or per caller.
- Withholding does not rewrite the answer or ask the runner for another one.
