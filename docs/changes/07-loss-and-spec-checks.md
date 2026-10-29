## What failed

Checklist M-04, M-07, M-08. `TransformerConfig` is the spec and it is copied into `reproducibility.json`. Nothing asserts the built module's parameter count and layer shapes against that config. Loss is logged from step 0. Nothing compares it with `ln(vocab_size)`. The loop stops on a non-finite loss and does not assert that a test batch in fp32 is finite in the forward and the backward.

The overfit check in `src/internal_ai/training/overfit.py` already covers M-06. Leave it.

## Change

- After `build_model`, compare `parameter_count` and each layer shape with the config. A mismatch raises before the first step.
- On step 0 of a from-scratch run, if the objective is cross-entropy, require the loss to sit within 20 percent of `ln(vocab_size)`. Record `initial_loss` and `ln_vocab` on the first metrics row. A contrastive embedding loss is not cross-entropy: record `initial_loss_check: not_applicable` rather than forcing `ln(vocab)`.
- Add a unit test that runs one fp32 batch and asserts no NaN or Inf in logits or gradients.

## Proof

- A config with the wrong `d_model` for the saved state raises.
- A toy vocabulary whose first loss is far from `ln(vocab)` fails the check.
- The finite-batch test passes on CPU.

## Out of scope

Do not add mixed precision or FlashAttention here. That is a separate change.
