# 35: Adaptation runs under the control plane

## Why

`train_prefix` did the arithmetic and nothing else. `python -m
internal_ai.training.alpha train` called it directly, so an adapter could be
produced with no run row, no budget check, no record of which code, seed or
dataset produced it, and no audit event. The artifact was governed afterwards —
`register_alpha` checks the capability, the licence and the released dataset —
but the act that produced it was invisible.

For a product whose claim is that a company owns and can reproduce its model,
that is the wrong half to govern. A card that names the base and the adapter
hash but not the seed, the commit or the dataset hash describes an artifact
nobody can rebuild.

## What changed

- `services/alpha.py` is the governed path. `train_alpha` checks the person's
  `TRAIN_MODEL` capability, refuses an unreleased dataset through the existing
  `require_released_dataset`, resolves the base through `available_base` so a
  research-licensed pin cannot be reached, and holds a run above
  `TRAINING_COST_GATE_USD` for an approved `training_run_above_threshold` gate.
- The run is submitted through `services/jobs.JobRunner`, so it is a job: the
  kill switch, the budget headroom check and `job.submitted` / `job.finished`
  audit events all apply, and a process killed mid-run leaves a `console_jobs`
  row in `failed` rather than nothing.
- Every run writes `reproducibility.json` beside the adapter: commit, base repo
  id, revision, licence and licence class, dataset id and content hash, the
  recipe with the layers actually trained, the seed, the adapter hash and the
  python and torch versions.
- The run is recorded with `store.record_run` and audited as
  `alpha.train_finished` with the base revision and the licence class, so the
  audit trail says which base an adapter came from without opening the card.
- `register_alpha` picks the record up from the adapter's directory and copies
  the commit, seed and run id onto the model card.
- `train_prefix` takes a `seed` instead of hardcoding `torch.manual_seed(0)`,
  and stores it in the adapter file.
- The command line keeps working for a one-off experiment, but says on stdout
  that it writes no run row, no record and no audit event, and points at the
  governed function.

## Verification

`tests/test_alpha_run.py`:

- one adaptation run leaves a `runs` row, a `reproducibility.json` naming the
  base revision, dataset hash and seed, and an `alpha.train_finished` event;
- an unreleased dataset is refused and records no run;
- a `data_owner` is denied `TRAIN_MODEL`;
- no allowed base raises `BaseRefused` before anything is queued;
- registering the adapter puts the seed, commit and run id on the card.

`ruff check .` and `ruff format --check .` stay clean; the existing alpha,
licence and offline tests are unchanged and still pass.

## Not changed

`train_prefix` still trains only rank-8 `q_proj` on the first layers with a
hand-written backward pass, and still needs torch. This change governs the run;
it does not make the run a real fine-tune. That is step 4, with a `peft`
backend and a CPU-sized fixture.
