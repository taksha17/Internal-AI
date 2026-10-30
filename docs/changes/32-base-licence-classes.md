# 32 — Base licences are a class, and a research pin is never swapped in

## Why

`PinnedBase.license` was a string nothing interpreted, and `available_base()`
caught the Apache-2.0 Qwen3-4B pin failing to verify and returned the
research-licensed Qwen2.5-3B pin instead. Nothing said so: not the return value,
not an audit event, not the card. An install could train an adapter, evaluate
it, promote it and ship it while its provenance rested on a licence nobody
approved.

That is the one bug a product selling "you own what you run" cannot have, so the
licence became a classification the code enforces rather than a label it stores.

## What changed

- `LicenseClass` (`permissive`, `restricted`, `research_only`) with
  `license_class_of()`. `Apache-2.0` and `MIT` are permissive; a small known set
  is restricted; **an unrecognised licence is research-only**, so a base nobody
  reviewed can never be treated as permissive by omission.
- `PinnedBase.license_class` derives the class from the SPDX string, so the two
  existing pins keep their constructors and gained a class.
- `BaseRefused` names the class and the allowlist in its message.
- `available_base()` takes `allowed` (default: permissive only) and `bases`, and
  refuses instead of falling through: a verified base outside the allowlist
  produces `BaseRefused` naming that base, its licence and its class.
- `fetch_primary_or_fallback()` fetches the research pin only when
  `allow_research_licence=True`, exposed as `--allow-research-licence` on the
  alpha CLI. A failed 4B download no longer becomes a 3B install silently.
- `smoke()` no longer swaps to the 3B pin on an out-of-memory error; it refuses
  and names the missing `base_model_exception`.
- `register_alpha()` refuses a base that is not permissive, before the adapter
  can be registered against it.

## Proof

`tests/test_base_license.py`:

- the two pins carry permissive and research-only classes, and the research one
  is outside the default allowlist;
- an unreviewed licence classifies as research-only;
- a permissive base that verifies is returned;
- **a research base is refused, not swapped in**, and the message names the
  repo, the licence class and the allowlist;
- nothing on disk names every candidate in the refusal;
- a tampered shard is not a candidate;
- `fetch` refuses the research pin unless asked, and fetches it when asked;
- `register_alpha` on the research base raises.

`tests/test_alpha_track.py`, `tests/test_offline_owned.py` and the rest of the
suite stay green: the defaults are stricter, and the only caller that wanted the
fallback now asks for it.

## Risks

- A 4 GB card that previously fell back to the 3B pin now refuses. That is the
  intended behaviour; the remedy is an approved `base_model_exception`, which
  change 1.1 adds.
- `RESTRICTED_LICENSES` is a starting list, not a legal opinion. Adding a base
  means adding its SPDX string and, if it is not permissive, the gate that
  allows it.
