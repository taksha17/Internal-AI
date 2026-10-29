## What is left

The install-wide default for flagged answers could only be changed through
`INTERNAL_AI_GUARDRAIL_POLICY`, which meant an operator editing an environment
file and restarting the process to make a governance change.

## Change

- `serving.guardrails.configured_install_policy` reads the setting from the
  store first, so a choice holds from the next request and needs no restart. An
  install that never chose still falls back to `INTERNAL_AI_GUARDRAIL_POLICY`,
  which keeps a fresh install seeded from its environment.
- The models page carries a `Flagged answers across the install` control, shown
  to whoever may set it and read as text to everyone else, beside the per-model
  choice it sits above.
- `POST /console/models/guardrail-default` calls
  `build.set_install_guardrail_policy`, which requires `set_serving_policy`
  (ml_lead), checks the value against the accepted policies before writing it,
  and records `serving.guardrail_default_set` with the before and after values
  and whether the install now follows the environment.
- An empty choice clears the stored setting, so the environment decides again.
- Reading the default at request time also repairs an install whose stored value
  is unrecognised: it fails closed per answer, as an unrecognised environment
  value already did.

## Proof

- `tests/test_serving_policy_console.py` shows an ml_lead setting `refuse` and
  handing it back to the environment with both values in the audit, an unknown
  value refused with nothing written, and a domain_expert refused with
  `console.denied` recorded and no control on the page.
- `tests/test_serving_guardrails.py` shows a stored default taking effect on the
  next answer of a running server, a cleared default falling back to the
  environment, and an unrecognised stored default failing closed.

## Out of scope

- The environment variable is not removed: it seeds an install that has never
  chosen, and the console setting wins once it is set.
- Changing the default does not re-run an evaluation or re-promote a model.
