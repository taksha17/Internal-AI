## What failed

Checklist T-02, T-05, T-06. A tokenizer is trained from the released corpus and its hash is written into `reproducibility.json`. Retraining overwrites the file. Nothing records why the vocabulary size was chosen. Encode and decode exist, and there is no test that a diverse sample, including whitespace and non-ASCII, survives a round trip. Training does not refuse a tokenizer whose file changed after the dataset was released.

## Change

- On `train_tokenizer_for`, write `tokenizer.json` and `tokenizer.sha256` beside it. A second call for the same dataset raises unless the caller passes a new dataset version. The hash is the freeze.
- `TrainConfig` stores `vocab_rationale`, a non-empty string. The console field is required. The run refuses a blank rationale.
- Add `tests/test_tokenizer_roundtrip.py`. The sample is: leading and trailing spaces, a blank line, `café`, a CJK character, a digit run, and an internal token from the existing jargon test. `decode(encode(text))` equals `text` for each.
- `train_embedding_model` loads the frozen hash and raises if the file's hash differs.

## Proof

- Retraining a tokenizer for a dataset that already has `tokenizer.sha256` raises.
- The round-trip test passes for the six strings.
- A train call with a mismatched hash is denied.

## Out of scope

Do not change the BPE algorithm or the default vocabulary size.
