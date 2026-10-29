## What failed

Checklist D-06, D-07, D-17, M-05. A dataset is one corpus. Lineage does not record the share of each source category. There is no ablation. Training tokenizes rows on the fly. Nothing is sharded, so a larger run cannot stream without holding the corpus.

## Change

- At build time, group records by the first path component of `record_key` (the top folder). Write `mixture` onto lineage: folder, record count, measured-or-estimated tokens, ratio. Ratios must sum to 1 within 0.001.
- An ablation is two short from-scratch runs on the tiny config, 20 steps, on two different folder subsets of a fixture, with the validation loss of each stored on the dataset as `ablations`. The console shows the two losses. This is the M-05 small-scale experiment for data mix. It does not pick a winner by itself.
- After the tokenizer is frozen, write `shards/shard-00000.jsonl` files of at most 256 records, each line `{"ids": [...], "record_key": "..."}`. The trainer reads shards in order and logs the shard file name in `data_offset`. A test builds two shards and checks the trainer's first offset names `shard-00000.jsonl`.

## Proof

- A two-folder fixture stores two mixture rows whose ratios sum to 1.
- The ablation stats contain two losses.
- The shard test does not load the 12,721-record dataset.

## Out of scope

Do not shard the live corpus in this change. The writer runs when a tokenizer is trained.
