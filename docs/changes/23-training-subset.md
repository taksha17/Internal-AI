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
  the strategy and the average score.
- `quality` sorts by language, boilerplate ratio, length and letter ratio;
  `random` is seeded from the run seed; `threshold` keeps rows at or above the
  score and caps at the requested size.

## Proof

- A real one-step run with `train_subset_size=2` over eight records encodes only
  the two clean ones, leaves the validation record intact, and records both
  counts in the audit row.
- The helpers are covered for boilerplate markers, quality ordering, all three
  strategies, a size larger than the corpus, and empty text.

## Out of scope

Do not change the tokenizer. Do not add a console or CLI switch for the subset
here; this only lets a caller ask for one.
