## What is left

`guardrail_policy` could only be set by editing a model row by hand, and the
rules that read it lived in the serving web app, so a console service could not
validate a choice before storing it.

## Change

- `guardrail_policy`, the scan and the accepted policies move to
  `serving/guardrails.py`; `serving/app.py` imports them, so the endpoint
  behaves exactly as before.
- A model card carries a `Flagged answers` control: follow the install default,
  warn and return, or withhold. It is shown to whoever may set it and read as
  text to everyone else.
- `POST /console/models/{id}/guardrail-policy` calls
  `build.set_guardrail_policy`, which requires `set_serving_policy` (ml_lead),
  resolves the candidate config against the install default before storing it,
  and records `serving.guardrail_policy_set` with the before and after values.
- An empty choice removes the key, so the model goes back to following
  `INTERNAL_AI_GUARDRAIL_POLICY`.
- `Store.update_model_config` writes config only; the artifact's content hash
  does not move, because how a model is served is not what the model is.

## Proof

- `tests/test_serving_policy_console.py` shows an ml_lead setting `refuse`,
  clearing it back to the install default with both values in the audit, an
  unknown policy refused with the config unchanged, and a domain_expert refused
  with `console.denied` recorded and no control on the page.

## Out of scope

- The install default still comes from the environment; the console sets a
  model's own choice only.
- Changing the policy does not re-run an evaluation or re-promote the model.
