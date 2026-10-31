# Draft: provenance mode, for approval before it is applied

Not applied. `AGENTS.md` and `PRODUCT.md` are untouched. Read this, then say
apply, and the two edits below go in with a change record and a test that
`scratch_only` refuses every adapter command.

## What problem this solves

`AGENTS.md` non-negotiable 1 says the from-scratch track has no code path that
reads pretrained parameters, and `PRODUCT.md` repeats "no pretrained weights" as
a binding rule. Phase 1 cannot exist under that as written: an adapter system is
by definition pretrained weights plus a delta. You approved turning the absolute
into a setting during this session; the wording below is how.

## The guarantee that must not move

Three sentences survive verbatim in spirit and stronger in form:

- A from-scratch model still trains from random initialisation on approved
  internal data only, and the from-scratch package still has **zero** code
  paths that read pretrained parameters.
- Alpha output is still never training data for the from-scratch track.
- Third-party model output is still never training data, in either mode.

What changes is only whether *a gated adapter track may load a licence-checked
base*, and that becomes an explicit, recorded, reviewable decision per install
instead of a permanent no.

## Edit 1: `AGENTS.md`, non-negotiable 1

Replace the current item 1 with:

> 1. **A base model is loaded only in provenance mode, and the default mode
>    loads none.** `scratch_only` is the default and keeps today's guarantee
>    exactly: the from-scratch track has no code path that reads pretrained
>    parameters, and none may be added. `open_weight_allowed` is a per-install
>    setting decided through a `policy_decision` gate and recorded in the audit
>    log; in that mode the adapter track may load a base that is on the
>    allowlist, hash-pinned, and recorded on every model card by repo id,
>    revision and licence. Only `internal_ai/training/alpha/` and its successor
>    `internal_ai/training/adapt/` may load those weights. A base whose licence
>    class is not `permissive` needs its own `base_model_exception` gate. Alpha
>    and adapter output must not be used to train the from-scratch track. Weight
>    files found in a data source are still refused.

## Edit 2: `AGENTS.md`, the change-rules list

Add one bullet after "Anything an agent can trigger is capability checked":

> - A base model is a governed artifact: registering one needs the capability
>   that names it, its licence class must be on the allowlist or covered by an
>   approved `base_model_exception`, and loading it writes an audit event with
>   the repo id, revision and licence. There is no code path that loads a base
>   in `scratch_only` mode, and a test walks the import graph to prove it.

## Edit 3: `PRODUCT.md`, binding rules

Replace the bullet "No pretrained weights. Models train from random
initialisation on approved internal data only." with:

> - **Provenance mode is explicit.** The default install trains from random
>   initialisation on approved internal data only and has no path that reads
>   pretrained parameters. An install that switches to `open_weight_allowed`
>   through its gate may adapt a hash-pinned base whose licence class is
>   allowlisted, and every model card then names the base, the revision and the
>   licence. Nothing outside the adapter package may load those weights.

## What the console shows

One line on the overview and on every model card: `Provenance: scratch_only` or
`Provenance: open_weight_allowed (2 bases allowlisted)`. A card for an adapter
shows `Base: Qwen/Qwen3-4B @ 1cfa9a7… (Apache-2.0, permissive)`. A person
reading a card never has to guess whether they own the weights or rent them.

## What the tests will assert

1. `scratch_only` makes every entry point in `training/alpha/` and
   `training/adapt/` raise, and the message names the mode.
2. Switching the mode needs the `policy_decision` gate; an unapproved switch
   does not take effect.
3. A `research_only` base is refused without a `base_model_exception`
   (change 32 already enforces the refusal; this adds the gate that lifts it).
4. A walk of `training/model.py`'s import graph finds no reference to
   `safe_open`, `from_pretrained`, `AutoModel` or `snapshot_download`.
5. The mode appears on the model card and in the audit event for every load.

## What I still need from you

- **Apply or reword.** Any sentence above can change; I would rather you rewrite
  a line than approve a line you dislike, because this file governs every agent
  that touches the repo afterwards.
- **The exception approver pair.** I proposed `legal` plus `program_lead` for
  `base_model_exception`, matching `policy_decision`. Say if it should be
  `compliance` or `security` instead.
- **The allowlist's starting contents.** Today `Apache-2.0` and `MIT`. Adding
  `Llama-3.x-Community` as `restricted` would widen the product to the most
  commonly deployed open bases but needs your legal position first.
