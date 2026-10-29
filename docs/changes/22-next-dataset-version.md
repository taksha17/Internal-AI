## What is left

`feedback export` writes an unreleased dataset, and the loop stage closes when that lineage exists. The export is stored with `training_data` false. Nothing walks that set back through the data stage as the corpus a later run is allowed to use.

## Change

- Releasing the exported dataset, through the existing release gate, sets `training_data` true and keeps `kind` as `feedback`.
- An unreleased export stays `training_data` false.
- The data stage links to the newest released dataset. A released feedback export is that dataset when it is the newest.
- The next from-scratch run may take that dataset id. Nothing starts the run on its own.

## Proof

- An unreleased export stays `training_data` false.
- After the release gate, `training_data` is true.
- The data stage href is that released dataset.

## Out of scope

Do not train on the exported rows automatically. Do not edit a dataset that is already released.
