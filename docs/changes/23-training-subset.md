## What is left

A from-scratch run encodes every record in the training split. A corpus with
boilerplate, short stubs and off-language rows spends its steps on the worst of
it, and nothing records that a run used less than the whole split.

## Change

- `TrainConfig` takes `train_subset_size`, `train_subset_strategy`
  (`quality`, `random`, `threshold`), `train_subset_threshold` and
  `min_quality_score`.
- The subset is chosen once, before encoding, from the training records only.
  Validation records are never subset, and the resume offsets are computed from
  the same list that trains.
- A chosen subset writes an audit row, `train.subset_selected`, with the counts,
  the strategy and the average score. The outcome carries the same counts, so the
  model card and the run row say how many records trained.
- The console training form and `internal-ai train run` both take a size, a
  strategy and a threshold. An empty size means every record.
- `quality` sorts by language, boilerplate ratio, length and letter ratio;
  `random` is seeded from the run seed; `threshold` keeps rows at or above the
  score and caps at the requested size.

## Proof

- A real one-step run with `train_subset_size=2` over eight records encodes only
  the two clean ones, leaves the validation record intact, and records both
  counts in the audit row.
- The helpers are covered for boilerplate markers, quality ordering, all three
  strategies, a size larger than the corpus, and empty text.
- A registered model card from a subset run reports `training_records_used` and
  the `training_subset` counts, and the console form and CLI flags are checked.

## Out of scope

Do not change the tokenizer. Do not pick a subset on the caller's behalf: with
no size, a run still uses every record.
