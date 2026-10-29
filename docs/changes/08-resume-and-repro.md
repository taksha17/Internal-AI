## What failed

Checklist S-04, S-05, S-07. `latest.pt` stores the model, the optimizer, and the step, and a run can resume. No test kills a run and shows the loss trajectory continues. The batch index follows the step, and the offset is not logged. The seed, dataset hash, tokenizer hash, and commit are recorded. Two runs are not compared.

## Change

- Log `data_offset` (dataset id, split, index of the first row in the batch) on every metrics row.
- `tests/test_resume_matches.py`: train 4 steps, stop, train 4 more with `resume=True`. Assert the step counter continues at 4, `data_offset` does not repeat step 3's index, and the loss at step 4 equals a control run that trained 8 steps without stopping, within 1e-5.
- `tests/test_two_runs_match.py`: two trainers, same seed, same four-record corpus, 6 steps. The loss lists are equal within 1e-5.
- Both tests are CPU and use the tiny config already used by the overfit check.

## Proof

- The resume test fails if the second process starts again at step 0.
- The two-run test fails if the seed is ignored.

## Out of scope

Do not add a process supervisor or a multi-node restart. That is the failure-recovery change, and this product has one process.
