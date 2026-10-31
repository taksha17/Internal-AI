"""Whether this install may load a pretrained base at all.

The default is `scratch_only`: no code path reads pretrained parameters, which
is the promise the product makes with a straight face. An install that wants to
adapt a base switches to `open_weight_allowed` through policy item `P6`, which
is a `policy_decision` gate decided by `program_lead` and `legal`.

The mode is read from a *decided* policy and from nothing else. A proposal that
is still pending must not switch the install on, or the gate would be a
formality — `PolicyRegistry` already keeps a proposal out of `current_value`,
and this module refuses to look at `status` anything other than `decided`.
"""

from __future__ import annotations

from internal_ai.context import Platform

POLICY_ID = "P6"
SCRATCH_ONLY = "scratch_only"
OPEN_WEIGHT_ALLOWED = "open_weight_allowed"
MODES: tuple[str, ...] = (SCRATCH_ONLY, OPEN_WEIGHT_ALLOWED)


class ProvenanceRefused(PermissionError):
    """This install may not load pretrained weights. The message says how to change that."""


def provenance_mode(platform: Platform) -> str:
    """The decided mode, or `scratch_only` for anything else.

    An undecided policy, a pending proposal, a missing row and an unrecognised
    value all mean the same thing here: no base may be loaded.
    """
    try:
        decision = platform.policy.get(POLICY_ID)
    except KeyError:
        return SCRATCH_ONLY
    if decision.status != "decided":
        return SCRATCH_ONLY
    return decision.current_value if decision.current_value in MODES else SCRATCH_ONLY


def allow_open_weights(platform: Platform) -> bool:
    return provenance_mode(platform) == OPEN_WEIGHT_ALLOWED


def require_open_weights(platform: Platform, what: str) -> str:
    """Allow a load, or refuse it naming the mode and the gate that lifts it."""
    mode = provenance_mode(platform)
    if mode == OPEN_WEIGHT_ALLOWED:
        return mode
    raise ProvenanceRefused(
        f"{what} would load pretrained weights, and this install is {SCRATCH_ONLY}. "
        f"Switch policy {POLICY_ID} to {OPEN_WEIGHT_ALLOWED} through a "
        "policy_decision gate; a pending proposal does not count."
    )
