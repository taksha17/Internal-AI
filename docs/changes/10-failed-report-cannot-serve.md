## What failed

Checklist E-04 and E-09. `promote_model` already requires `latest_frozen_pass`. The model that search is serving does not satisfy that rule anymore:

- `mdl_309a38d221fd` (`new-volume-3` v1) is `promoted` and active.
- Report `rep_2f336550b8d4` on frozen set `evs_ef6ae3cf6374`: recall@10 **0.083** (2 of 24) against a keyword baseline of **0.375**. `passed` is 0.
- That report's `permission_leak` check is the old note, "permission enforcement is verified by the retrieval service tests", with `passed: true`. `permission_probe` in `evals/harness.py` can now fail. This stored report was not produced by that probe.

A later failing report does not change `status` and `served_model` still prefers the promoted row.

## Change

- `served_model` skips a promoted model whose newest report for that model failed or whose `permission_leak` note is the legacy sentence. If none remain, search returns 409 with "no model has a passing frozen report" rather than serving the failure.
- Add `internal-ai models demote <model-id>` for a human with `PROMOTE_MODEL`. It sets status back to `candidate`, writes an audit event, and does not delete the weights.
- Re-running an eval overwrites nothing in the frozen set. It appends a new report. The model page shows the newest report's recall@10, the baseline, and whether `permission_leak` is the legacy note.
- Do not auto-demote inside this change's migration. The command and the serving skip are the behaviour. Document the live model id in the pull request so a person can run the demote.

## Proof

- A promoted model with a newest failed report is absent from `GET /v1/models` active flags and from search.
- A model whose newest report passed is served.
- Demote writes one audit event and leaves the file on disk.

## Out of scope

Do not lower the recall bar to make 0.083 pass. Do not retrain the encoder here.
