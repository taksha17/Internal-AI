## What failed

Checklist D-11, D-12. Measured on released `ds_4405d8cbf22f` (12,721 records, 162,674,575 bytes) on 2026-10-28:

- 300 exact-duplicate groups, 484 extra copies. Examples: the same OSMOS file under `OSMOS/src` and `OSMOS/release/.../unpacked`, and identical Gemma error logs.
- At least 3,930 near-duplicate pairs sharing a 16-bit SimHash band at Hamming distance 3 or less. That is a lower bound.
- The dataset's recorded steps are only `pii_scrub` and `critical_redaction`. `DataPipeline` now calls `deduplicate` during a build, and this release never did. A build-time call does not protect a dataset that was assembled another way.

## Change

- In `DataPipeline.release`, before the PII audit, hash every payload with `Record.fingerprint()` (SHA-256 of the stripped text). If any hash appears twice, raise and do not set `released`. The error includes the count and up to five path pairs.
- Run the existing SimHash near-duplicate pass on the same rows. Add `dedup_residual_max` to the pipeline config, default `0.01` (one percent of rows). A residual above that refuses release. Record `exact_duplicate_groups`, `near_duplicate_pairs`, and `residual_rate` on the dataset stats.
- Add a console action on the dataset page, "Drop duplicate copies into a new version", which builds the next version with the duplicates removed and leaves the released version immutable. The new version still needs its own release gate.

## Proof

- A two-record corpus with identical text fails release.
- A corpus under the residual cap passes, and the stats contain the counts.
- The live dataset is not edited by the test.

## Out of scope

Do not rewrite `ds_4405d8cbf22f` in place.
