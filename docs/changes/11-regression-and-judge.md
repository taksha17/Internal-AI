## What failed

Checklist E-10, E-11, E-12. Evals can be re-run by hand. Nothing runs them when a candidate is registered. Nothing checks that a second run of the same eval reproduces the score. The evaluation role is separate on paper, and the person who trained the model can still record the report.

## Change

- `register_model` looks up the newest frozen eval set for the dataset. If one exists, it queues `evaluate_model` before the model can be requested for promotion. Promotion already demands a passing report; this makes the report exist without a person remembering.
- Store `evaluated_by` on the report. `record_report` refuses when `evaluated_by` equals the model's `registered_by`.
- `tests/test_eval_repro.py` runs `evaluate_retrieval` twice on the hash embedder fixture and asserts the metric dicts are equal.
- A second candidate whose recall@10 drops by more than 0.05 against the previous candidate's report on the same frozen set is marked `regressed` on the model page. Promotion of a regressed model requires the note to contain the word `accepted` from a person who is not the builder.

## Proof

- Registering a model against a dataset that has a frozen set creates an eval job.
- The builder recording their own report is denied.
- Two eval calls on the fixture return the same recall@10.

## Out of scope

Do not add MMLU, HellaSwag, or a public benchmark runner.
