# 36: A proposed policy value no longer applies before it is decided

## Why

`PolicyRegistry.propose` set `current_value = value` at proposal time, then
opened a `policy_decision` gate. Anything reading the policy saw the new value
while the gate was still pending, so the gate only recorded an approval after
the fact. `confirm` then wrote the status and the approver, which is why the
hole never showed: the audit trail looked correct while the behaviour had
already changed.

That is a display quirk for P1–P5 and a safety bug the moment a policy governs
something that can be loaded. The next policy item is provenance mode — may this
install load pretrained weights at all — and under the old behaviour one person
could switch it to `open_weight_allowed` and have every entry point honour it
while `program_lead` and `legal` still had the gate open.

`confirm` also could not find its gate: `propose` returned the gate id but never
stored it, and `confirm` reads `policy_gate:<id>` from the key-value store, so a
caller that trusted the documented default got "no gate recorded". The console
and CLI paths passed the id explicitly and hid it.

## What changed

- `propose` records the proposed value in `policy_proposed:<id>` and the gate in
  `policy_gate:<id>`, sets the status to `pending_approval`, and leaves
  `current_value` and `rationale` alone. The rationale a person supplies lives
  in the gate's evidence, where the approver reads it.
- `confirm` requires an approved gate and a pending proposal, then applies the
  proposed value and clears it. It can no longer decide a policy against a gate
  that was never recorded, or decide one nobody proposed.
- A second proposal for the same policy replaces the pending value and the gate
  id; the first gate, if it is later approved, cannot then be used to apply the
  second value, because `confirm` reads the value recorded with the current
  proposal.

## Verification

`tests/test_policy_decisions.py`:

- a proposal leaves `effective()` unchanged and `pending_approval` set;
- confirming an unapproved gate raises and changes nothing;
- an approved gate applies exactly the proposed value;
- a second proposal switches nothing until its own gate is approved;
- confirming with no proposal refuses and invents nothing;
- the governed path (`governance.propose_policy` / `confirm_policy`) still
  proposes, decides, and audits both steps.

`ruff check .` and `ruff format --check .` stay clean.

## Review

`policy.py` defines a guardrail, so `AGENTS.md` asks for a second reviewer on
this change. The behaviour it fixes is a widening of what a single person could
do, not a narrowing: no policy becomes harder to change through the governed
path, only easier to change *before* approval.
