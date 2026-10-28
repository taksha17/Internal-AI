## What failed

Checklist D-01, D-03, D-04. `data_sources` stores path, sensitivity, owner, and secret hits. It has no legal status and no license. The live source `src_d5bea3c3f593` (new-volume-3) was approved with `legal` empty. Sensitivity exclusion drops configured classes at ingest, and nothing samples a released corpus afterwards to prove a regulated or customer-confidential class is absent.

## Change

- Add `legal_status` (`internal`, `licensed`, `public`, `refused`) and `license` (name, url, compatibility note) on the source. `licensed` and `public` require the license fields. `refused` can never be ingested.
- The new-data-source gate evidence must include those fields. `request_gate` rejects the gate without them.
- At release, draw a seeded sample of 50 records (the same sampler as `pii_spot_check`) and fail the release if any record's sensitivity is in the excluded set or its source `legal_status` is `refused`. Write the sample size, the hits, and the seed onto the dataset stats as `exclusion_audit`.
- The sources page shows legal status and license next to the approval.

## Proof

- A source with no legal status cannot be approved.
- A released dataset whose sample contains an excluded class is refused, and the audit event names the class.
- A clean sample records `exclusion_audit.passed = true`.

## Out of scope

Do not delete `ds_4405d8cbf22f`. A fix is a new dataset version.
