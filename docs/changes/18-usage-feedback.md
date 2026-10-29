## What failed

Checklist order-of-operations step 12, G-14, A-09. `serving.request` events store latency, hit count, model, and whether the citation passed. Nothing reads them for drift. Nothing turns a bad answer into a new training example. The console shows spend and jobs, and it does not show GPU utilization.

## Change

- `internal-ai feedback list` prints recent `serving.request` rows where `citation_passed` is false: time, model, hit count, latency. It does not print the question, because the audit does not store it.
- A person with `data_owner` can file a feedback row by hand: the question they asked, the expected file and line, and the dataset they want it considered for. That row is not training data. `internal-ai feedback export` writes a new unreleased dataset version containing only those rows, with lineage pointing at the feedback ids. Release still requires the release gate, the dedup gate, and the PII audit.
- Drift: if the seven-day rate of `citation_passed = false` is more than twice the previous seven days, raise an alert of kind `quality_drift`.
- GPU utilization stays a single integer the trainer already can know. Write `gpu_busy` as 0 or 1 on the metrics row when CUDA is available, and 0 on CPU. The jobs page shows the latest value. Do not scrape `nvidia-smi` from the Linux laptop for the Windows card.

## Proof

- Export of two feedback rows creates an unreleased dataset of two records and does not mark it released.
- A fixture audit log with a doubled failure rate creates one `quality_drift` alert.
- The list command's output does not contain a planted question that was never stored.

## Out of scope

Do not automatically train on feedback. A person releases the new version.
