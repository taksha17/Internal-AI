## What failed

Checklist R-04, R-05, R-07. Training loss falls and a spike aborts the run. There is no validation curve, so a run can look healthy while it is memorising. Evals run after the run. A checkpoint stores the config, the dataset hash, and the commit. Nothing reloads it and reproduces the logged number.

## Change

- When the dataset has a `validation` split, every `log_interval` steps compute the loss on one fixed validation batch (seeded, not shuffled). Append `validation_loss` to `metrics.jsonl`. A run whose validation loss rises for three consecutive logs sets the run status to `diverged` and stops. The alert text names the step.
- After the run, reload `latest.pt`, recompute that same validation batch, and require the loss to match the last logged `validation_loss` within 1e-4. Store `checkpoint_reloaded: true` on the run result.
- The model page plots both series from `metrics.jsonl`. No new frontend library: two columns of numbers are enough if a plot is hard. Prefer a small inline SVG built on the server with no script.

## Proof

- A fixture with a validation split writes `validation_loss` on more than one row.
- Corrupting the reloaded weights fails the match.
- A dataset with no validation split records `validation_loss: not_applicable` and does not pretend the training loss is a validation loss.

## Out of scope

Do not run the full frozen-set retrieval eval on every step. One batch loss during training, the retrieval eval after.
