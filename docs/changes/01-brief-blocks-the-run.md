## What failed

Checklist F-01, F-02, F-03, F-04, F-05. The brief record exists in `src/internal_ai/brief.py` and the console can edit it, but nothing in the run path reads `ProjectBrief.complete()`. On the live install the five policy questions are still `undecided`, and the roster has no named owner for infra or evaluation. A plan can still be dispatched and a dataset can still be released.

## Change

- `train_embedding_model`, `train_classifier_model`, and `release_dataset` call `brief.load` and raise `PermissionError` when `complete()` is false. The error lists `brief.missing` in the words already used there.
- A policy question that chooses data classes (`What data classes may be used for training`) must be `decided` before release. An undecided question is not a default of "allowed".
- Owners stored on the brief must be person ids that exist, are not placeholders, and hold the role named in `OWNER_ROLES`. A typed name that matches nobody is a missing owner.
- The overview page already shows the gaps. The jobs page must show the same sentence when a train or release button is refused, so the person sees the gap where they clicked.

## Proof

- A test with an empty brief: release and both train entry points raise, and the audit outcome is denied.
- A test with every field set and a decided data-class policy: those calls proceed to the checks they already have.
- A placeholder person cannot satisfy `infra` or `evaluation`.

## Out of scope

Do not invent the company's target, budget, or owners. The record stays empty until a person writes it.
