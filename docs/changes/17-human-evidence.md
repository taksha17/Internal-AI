## What failed

Checklist C-01, C-02 is already implemented (`training/cost.py`). C-08 passes because offline mode refuses a third-party call. Still missing as evidence, not as code guesses:

- C-01. No record that GPU capacity is reserved, with dates. The 1650 is a machine that happens to be on.
- R-03 and R-06. No instability playbook and no GPU-hours-against-plan figure.
- R-08. No ML-lead sign-off that a run is healthy.
- G-13. No signed review of the training and serving machines.
- C-09. No facilities note. Power and cooling stay a sentence a person writes, not a sensor integration.

## Change

- Add an `evidence` record: `kind` in `compute_reservation`, `run_health`, `security_review`, `instability_playbook`, `facilities`. Fields: `body`, `signed_by`, `signed_at`, `subject` (a run id or `programme`). The signer must hold `ml_lead` for run health, `security` for the security review, and `program_lead` for the reservation and the facilities note. The signer cannot be the person who started the run, for run health.
- A training run whose estimated GPU hours from `estimate_gpu_run` exceed 1 hour requires a `compute_reservation` whose dates cover today. Under 1 hour, the local machine is accepted and the audit says `compute_reservation: local`.
- The run page shows GPU hours used against the estimate. Crossing the estimate opens a `training_run_above_threshold` gate. It does not kill a run that is under the existing dollar cap.
- The instability playbook is a markdown body stored on the evidence row. The run page links to it. An empty body does not block a short run. It blocks promotion.

## Proof

- A run-health signature by the person who started the run is denied.
- A long estimate without a reservation is denied.
- A short CPU run records `compute_reservation: local`.

## Out of scope

Do not reserve a cloud cluster or SSH to the Windows machine from this change.
