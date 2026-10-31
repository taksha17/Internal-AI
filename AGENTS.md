# AGENTS.md

Rules for agents working in this repository. Humans contributing a branch and a
pull request read `CONTRIBUTING.md`; where the two differ, this file wins.

## What this is

The control plane a company runs on its own machines. Two tracks:

- **From scratch.** A model trained from random initialisation on approved
  internal data only.
- **Alpha.** One pinned open base that is already on disk, adapted into an
  adapter the company owns.

Offline is the default: no outbound call unless a lab sets
`INTERNAL_AI_OFFLINE=0`. The licence is PolyForm Noncommercial 1.0.0 in
`LICENSE`; never restore Apache or any permissive grant.

What you write here changes how other agents behave, so the bar is higher than
for ordinary application code.

## Non-negotiables

1. **A base model is loaded only in provenance mode, and the default mode
   loads none.** `scratch_only` is the default and keeps the old guarantee
   exactly: the from-scratch track has no code path that reads pretrained
   parameters, and none may be added. `open_weight_allowed` is a per-install
   setting decided through a `policy_decision` gate on `P6` and recorded in the
   audit log; in that mode the adapter track may load a base that is on the
   allowlist, hash-pinned, and recorded on every model card by repo id,
   revision and licence. Only `internal_ai/training/alpha/` may load those
   weights. A base whose licence class is not `permissive` needs its own
   `base_model_exception` gate. Alpha output is never training data for the
   from-scratch track. Weight files found in a data source are refused.
2. **Never train on third-party model output.** Distillation is a non-goal.
   Task data comes from internal documents, tickets and resolved questions.
3. **Never send raw records to an external API.** Schemas, aggregate
   statistics, redacted samples, code and logs may leave; records may not. See
   PRD 6.3.
4. **Never write to a frozen eval set.** Frozen sets are write-protected from
   every agent. A change is a new version behind a new `eval_set_freeze` gate.
5. **Never decide a gate as an agent.** Gates are human-only, the requester
   cannot approve their own, and a gate passes only when the approvers' roles
   cover every required role.
6. **Never write to `audit_events`.** It is append-only, enforced by triggers
   and chained by hash. Use `AuditLog.record`.

## Where the guardrails live

Change the rule in its own file, never around it:

| Rule | Home | A change needs |
| --- | --- | --- |
| Who may do what | `internal_ai/roles.py` (agents), `services/permissions.py` (people) | the role that needs it and nothing else |
| Every agent action | `internal_ai/agent.py`, `internal_ai/tools.py` | capability check plus an audit event |
| Task states | `internal_ai/store.py:ALLOWED_TRANSITIONS` | the map entry is the specification |
| Spend | `internal_ai/budget.py` | every new spend path calls `record_spend` or `check`; bypassing it is a bug |
| Datasets | `internal_ai/data/` | released versions are immutable; a change is a new version with a new hash and lineage entry |
| Promotion | `internal_ai/workflows.py:promote_model` | a passing eval report against a frozen set and an approved `model_promotion` gate |
| Gates | `internal_ai/gates.py`, `GateType` in `domain.py` | a new type plus the roles that may decide it |
| Base models | `internal_ai/training/alpha/pin.py` | a licence class and an allowlist entry, never an unchecked pin |
| Provenance mode | `internal_ai/training/provenance.py`, policy item `P6` | the mode reads only a decided policy; `scratch_only` refuses every load |

## Console

`/console` is a third client of the same services as the CLI and the JSON API:
server-rendered Jinja pages, htmx vendored under
`src/internal_ai/console/static/`, no CDN, no inline script, no inline style.
The build and its safety checks are documented in `docs/phase-2-console.md`.

Sign-in is a safety surface. The session cookie and the form token are both
`SameSite=Strict`, `HttpOnly`, and `Secure` when TLS is on. A missing `Origin`
or `Origin: null` is accepted, because some browsers send `null` for a
same-site form and the cookie is what proves the post. `Sec-Fetch-Site:
cross-site` and a named foreign origin are refused. Treat a regression here as
a safety bug. The failure text is identical for an unknown account and a wrong
password.

The kill switch sits on every signed-in page and needs a typed reason.
Engaging is immediate; releasing needs a different person.

Brand lives in `PRODUCT.md`. Two invariants the tests hold: colours appear only
in the token block at the top of `console.css`, so a rule carrying its own hex
is a bug; and body text stays at or above 4.5:1 (`tests/test_console_palette.py`).

## Style

- Python 3.12. Dataclasses for domain objects, `StrEnum` for closed sets, type
  hints on public functions.
- No comments unless the reasoning is not obvious; docstrings explain why, not
  what.
- Determinism where it is visible: seeded randomness, sorted iteration, content
  hashes over the fields that define an artifact.

## Checks

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

A commit leaves all three green. Every new capability, gate type or state needs
a test showing it is **denied** to the roles that must not hold it, not only a
test showing it works. A restricted state also needs a test that the state
machine refuses the illegal move.

## Git

- One change is one branch. The change record is a working note at
  `docs/changes/NN-<slug>.md`, kept out of git like the plans and the PRD; the
  pull request body carries what a reviewer needs.
- Agents do not push to `main`. Only the owner (`taksha17`) merges there, so an
  agent commits on its branch, pushes that branch, and opens or updates one
  pull request.
- Agent commits use the agent identity, so machine work stays distinguishable
  from the owner's:

  ```bash
  git -c user.name='Cline Agent' -c user.email='cline.agent@internal-ai.local' commit -m '...'
  ```

  Leave the repository's default identity as the owner's.
- Commit messages say why the change exists, not what the diff contains.

## Review

- A builder never reviews its own work; `roles.py:reviewer_for` names the
  reviewer for each agent role.
- `gates.py`, `budget.py`, `audit.py`, `policy.py` and `roles.py` define the
  guardrails themselves and need a second reviewer.
