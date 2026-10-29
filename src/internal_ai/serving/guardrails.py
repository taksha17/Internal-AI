"""What a served answer may say, and who decides what happens when it does.

A scan that only warns is useless to an install that refuses to return text
naming a secret, and a policy that falls back to warning on a typo is worse than
none. So the policy is closed: `warn` or `refuse`, an unset one takes the
install default, and anything else is an error rather than a quiet downgrade.
"""

from __future__ import annotations

import re
from typing import Any

from internal_ai.config import installed_guardrail_policy
from internal_ai.store import Store

GUARDRAIL_WARN = "warn"
GUARDRAIL_REFUSE = "refuse"
GUARDRAIL_POLICIES = (GUARDRAIL_WARN, GUARDRAIL_REFUSE)
INSTALL_POLICY_KEY = "serving.guardrail_policy"

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpassword\b", re.IGNORECASE), "potential password disclosure"),
    (re.compile(r"\bapi[\s_-]*key\b", re.IGNORECASE), "potential API key disclosure"),
    (re.compile(r"\bconfidential\b", re.IGNORECASE), "confidential term used"),
]


def guardrail_policy(config: dict[str, Any], default: str = GUARDRAIL_WARN) -> str:
    """What a model does with flagged text: warn and return it, or refuse it.

    A model's own ``guardrail_policy`` wins; an unset or empty one takes the
    install default, which itself warns unless an operator says otherwise. An
    unrecognised policy fails closed instead of quietly falling back to warning.
    """
    policy = str(config.get("guardrail_policy") or default)
    if policy not in GUARDRAIL_POLICIES:
        raise ValueError(
            f"unknown guardrail_policy {policy!r}; use one of {', '.join(GUARDRAIL_POLICIES)}"
        )
    return policy


def configured_install_policy(store: Store) -> str:
    """What the install does when a model does not choose for itself.

    The console's setting wins, so an operator's choice holds from the next
    request and needs no restart. An install that never chose falls back to
    ``INTERNAL_AI_GUARDRAIL_POLICY``, which is how a fresh one is seeded.
    """
    return store.kv_get(INSTALL_POLICY_KEY, "") or installed_guardrail_policy()


def validate_output_guardrails(text: str) -> list[str]:
    """Every rule the text breaks, in rule order. Empty means it is clean."""
    return [message for pattern, message in _PATTERNS if pattern.search(text)]
