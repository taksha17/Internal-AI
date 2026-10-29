## What failed

Checklist D-19 and the remainder of D-13. Dataset lineage on `ds_4405d8cbf22f` has the parent dataset, the source id, the split seed, and the scrub steps. It has no code commit. A record can be joined to `source_id` and `record_key`, and nobody has run a sample that must succeed. The PII scrub and the full-corpus rescan found zero critical leaks after redaction, and the audit does not record a recall estimate for the detectors.

## Change

- Stamp `code_commit` from `git rev-parse HEAD` onto dataset lineage at build time. A missing git directory stores `unknown` and release refuses that dataset.
- Add `internal-ai dataset trace <dataset-id> --sample 20`. For each sampled record print `record_key`, `source_id`, the lineage step names, and `code_commit`. Exit non-zero if any sample is missing a source, a step list, or a commit. `DataPipeline.release` runs this and stores the result as `lineage_sample`.
- Add a fixture corpus with one known email, one known API token, and one known private-key block. Run `scrub_pii` and `secret_scan`. The release stats gain `detector_recall`: hits found divided by planted hits, per detector. A detector under 1.0 on the fixture fails the test, not the customer's corpus.

## Proof

- `dataset trace` fails when lineage has no commit.
- The planted fixture scores recall 1.0 for email, api token, and private key.
- A released dataset's stats include `lineage_sample.passed` and `code_commit`.

## Out of scope

Do not claim the detectors catch every real-world secret. The recall is against the planted fixture.
