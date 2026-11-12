"""Which engine answers, and the boundary that decides it.

The harness is inference-agnostic in shape: one OpenAI-compatible call whichever
engine sits behind it. It is not agnostic about the boundary. A record leaving
the machine is what rule 3 forbids, so an external adapter is refused unless the
caller says the work is code or configuration and the install has already left
offline mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from internal_ai.config import offline_mode
from internal_ai.llm import LLMAdapter, build_adapter

EXTERNAL = frozenset({"anthropic", "openai", "openrouter", "deepinfra"})
"""Names that carry a prompt off this machine. A gateway is not a backend choice."""

KINDS = ("auto", "scripted", "local", "anthropic")


class BoundaryRefusal(PermissionError):
    """The chosen engine would carry a record past the boundary."""


@dataclass(frozen=True, slots=True)
class BackendChoice:
    name: str
    adapter: LLMAdapter
    external: bool


def is_external(adapter: LLMAdapter) -> bool:
    return any(part in EXTERNAL for part in adapter.name.split("+"))


def choose_backend(kind: str = "auto", *, allow_external: bool = False) -> BackendChoice:
    """Pick an engine, or refuse one that would leave the boundary."""
    if kind not in KINDS:
        raise ValueError(f"unknown backend {kind!r}: this install has {', '.join(KINDS)}")
    adapter = build_adapter(kind)
    external = is_external(adapter)
    if external and not allow_external:
        raise BoundaryRefusal(
            f"{adapter.name} sends the prompt to a third party; this install refuses it "
            "for records, and allow_external is only for code and configuration work"
        )
    if external and offline_mode():
        raise BoundaryRefusal(
            f"offline mode refuses {adapter.name} before any HTTP call, whatever the caller allows"
        )
    return BackendChoice(adapter.name, adapter, external)
