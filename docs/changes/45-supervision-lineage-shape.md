# 45 — Supervision lineage in the shape its readers already accept

Found while building the rebase recipe (change 44), which needed a dataset that
had actually been released.

## What was wrong

`ingest_supervision` wrote `lineage["sources"]` as a list of source **ids**.
Every reader in the repo treats each entry as a record: `pipeline.sample_lineage`
calls `entry.get("source_id")`, `services/build.py` and `workflows.py` do the
same. So releasing a supervision set died with `AttributeError: 'str' object has
no attribute 'get'` on the very row the check was supposed to trace — the one
place lineage is meant to prove something.

It stayed hidden because a second gap masked it: a supervision set records an
estimated token count, and `DataPipeline.release` refuses to release a version
whose tokens were not measured. Nothing could reach the crash through the
supported path, so nothing noticed the shape was wrong.

## What changed

The producer now emits the same entry every other builder uses — `source_id`,
`path`, `content_hash`, `owner`, `approved` — with the hash taken over the rows
that came from each source, so the number means something rather than being
carried for the check's benefit. `config_hash` and `code_commit` are recorded
too, because `sample_lineage` refuses a set whose config hash is empty or whose
commit is unknown.

## What is still open

The release path itself: a supervision set still cannot be released until its
token count is measured rather than estimated. That is a decision about how
supervision corpora are counted, not a shape bug, and changing it here would
have quietly altered what `token_count` means for every dataset. It is named
here rather than fixed.
