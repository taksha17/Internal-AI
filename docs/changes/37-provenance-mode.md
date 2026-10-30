# 37: Provenance mode

## Why

`AGENTS.md` non-negotiable 1 said the from-scratch track has no code path that
reads pretrained parameters, and `PRODUCT.md` repeated it as a binding rule. The
adapter track existed anyway, reachable from the command line, so the absolute
was already untrue: the guarantee was a sentence in a file rather than a
property of the code. The owner approved replacing the absolute with an explicit
setting and reviewed the exact wording in
`docs/yc-plan/PROVENANCE-MODE-DRAFT.md`.

What must not move is the guarantee itself, and it does not: the from-scratch
training path still has no route to pretrained parameters, and the default
install still loads none.

## What changed

- `AGENTS.md` non-negotiable 1 now reads as provenance mode: `scratch_only` is
  default and preserves the old promise, `open_weight_allowed` is a per-install
  decision recorded through the `policy_decision` gate on `P6`, and only
  `internal_ai/training/alpha/` may load the allowlisted, hash-pinned base. A
  guardrail row points at `training/provenance.py`.
- `PRODUCT.md` states the same as a binding rule, including that nothing outside
  the adapter package may load those weights.
- `policy.py` gains item `P6`, "May this install load a pretrained open base to
  adapt?", defaulting to `scratch_only`, decided by `program_lead` and `legal`
  through the existing gate.
- `training/provenance.py` reads the mode. Only a `decided` policy switches the
  install on; a pending proposal, a missing row and an unrecognised value all
  mean `scratch_only`. Change 36 fixed the policy registry so a proposal no
  longer applies before its gate; this module refuses to look at status at all.
- `services.alpha.train_alpha` calls `require_open_weights` before it touches a
  dataset, so a `scratch_only` install refuses to adapt anything and leaves no
  run row. The reproducibility record and the audit event carry the mode.
- The bare command line requires `--open-weights` for `fetch`, `smoke` and
  `train`, and says that it records no run, audit event or licence decision.
  That path stays for experiments, not for anything kept.
- Every model card prints `Provenance: scratch_only` or
  `Provenance: open_weight_allowed`; `register_alpha` writes the mode onto the
  card from the install's policy, not from a caller-supplied argument.

## Verification

`tests/test_provenance.py`:

- the default install reports `scratch_only` and refuses;
- a pending proposal does not switch it on (the change 36 property, now load-bearing);
- a decided policy switches it on;
- an unrecognised value falls back to `scratch_only`;
- `train_alpha` is refused under `scratch_only` and records no run;
- the card prints the mode, before and after the switch;
- the licence and dataset checks still run in their old order for a caller who
  is allowed to adapt.

`tests/test_alpha_run.py` now switches the mode on in its fixture, which is the
point of the change: those runs were previously possible on an install that had
decided nothing.

`ruff check .` and `ruff format --check .` stay clean.

## Not changed

The refusal order in `train_alpha` is capability, then provenance mode, then the
released dataset, then the licence allowlist. Registration and serving keep
their own checks: an adapter on a research-licensed base is still refused by
change 32 regardless of the mode. `base_model_exception` is not added here; the
allowlist remains Apache-2.0 and MIT, so there is nothing for it to lift yet.
