## What is left

Supervision stores a pair a person wrote and refuses a model author. Nothing stores which of two answers a person prefers. Nothing keeps a later training step closed until that set is released and a backend is attached.

## Change

- A person with `domain_expert` files two answers and marks which one is better. `authored_by` is that person.
- `authored_by` of `model`, or a source id that starts with `alpha`, is refused and audited.
- The rows are not training data. Export writes an unreleased dataset. Release uses the existing release gate.
- A preference training command refuses until that dataset is released and a compute backend is attached. This change does not apply a preference update.

## Proof

- A model author is refused and an audit row records the refusal.
- Export stays unreleased.
- The training command raises before release, and it raises after release when no backend is attached.

## Out of scope

No DPO or RLHF update. Do not train on the 1650.
