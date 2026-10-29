## What is left

The policy was per model only, so an install that wants flagged text withheld
had to set it on every model, and a model that said nothing fell back to warning
no matter what the operator wanted.

## Change

- The install has a default, `INTERNAL_AI_GUARDRAIL_POLICY` (`warn` when unset).
- A model's own `guardrail_policy` still wins, and an empty one counts as unset,
  so choosing per model stays optional.
- An unrecognised value in either place fails closed with `500` naming the
  accepted policies.

## Proof

- `tests/test_serving_guardrails.py` shows the install default of `refuse`
  withholds a flagged answer for a model that does not choose, a model that sets
  `warn` still returns the text under that same default, and an unrecognised
  install default fails closed.

## Out of scope

- The default is read from the environment, so changing it needs a restart; the
  console does not write it.
