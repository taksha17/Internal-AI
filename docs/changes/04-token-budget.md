## What failed

Checklist D-05. `ds_4405d8cbf22f` stores `token_count` 40,231,401 and `token_count_kind` is unset, so the figure is still the character estimate (`len(text)//4`). The promoted encoder is `d_model` 128, 4 layers, 4 heads, `d_ff` 512, context 256. Nothing compares tokens to parameters against the 20-tokens-per-parameter floor.

`measure_token_count` can replace the estimate after a tokenizer exists. Training does not require that replacement, and it does not compute the ratio.

## Change

- `train_embedding_model` and `train_classifier_model` require `stats["token_count_kind"] == "measured"`. An estimate raises `PermissionError` naming the dataset.
- Record `parameter_count` from the built module onto the run's reproducibility file.
- Compute `tokens_per_parameter = measured_tokens / parameter_count`. The default floor is 20. The project brief may set a higher floor as a metric named `tokens_per_parameter`. Below the floor, training raises and the audit event carries both numbers.
- The dataset page shows "estimated" or "measured" next to the token count, and the ratio once a run has a parameter count.

## Proof

- Training on an estimate-only dataset is denied.
- A measured dataset under the floor is denied, and the message contains both numbers.
- A measured dataset above the floor reaches the trainer.

## Out of scope

Do not retokenize the 12,721-record corpus in this change. The check runs when someone trains.
