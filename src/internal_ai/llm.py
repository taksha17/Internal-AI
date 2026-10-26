"""LLM adapters for role agents.

Policy P2 governs this layer: agents may use third-party LLM APIs for code and
configuration work, but never for raw internal data. Adapters therefore receive
prompts that are scrubbed of record content by the caller, and every call is
audited with its token cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from internal_ai.secrets import ANTHROPIC_API_KEY, operator_secret, refuse_secret_in_prompt

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5-20250929": (0.003 / 1_000_000, 0.015 / 1_000_000),
    "claude-opus-4-1-20250805": (0.015 / 1_000_000, 0.075 / 1_000_000),
    "claude-haiku-4-5-20251001": (0.001 / 1_000_000, 0.005 / 1_000_000),
}


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any] | None:
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


class LLMAdapter(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Completion: ...


class ScriptedAdapter:
    """Deterministic offline adapter.

    Used for the Phase 0 no-op loop and for tests. It produces a structured
    response derived from the prompt so runs are byte-for-byte reproducible.
    `nominal_cost_usd` lets offline runs still exercise budget accounting.
    """

    name = "scripted"

    def __init__(self, nominal_cost_usd: float = 0.0) -> None:
        self.nominal_cost_usd = nominal_cost_usd

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Completion:
        payload = {
            "status": "ok",
            "summary": "Deterministic offline execution of the agent contract.",
            "artifacts": [],
            "acceptance_criteria_met": True,
            "next_state": "in_review",
            "notes": "No external model was called (policy P2 development mode).",
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        return Completion(
            text=text,
            model="scripted",
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            cost_usd=self.nominal_cost_usd,
            raw={"prompt_chars": len(prompt), "system_chars": len(system)},
        )


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Completion:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        refuse_secret_in_prompt(self.api_key, prompt, system)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        in_price, out_price = MODEL_PRICING.get(self.model, (0.0, 0.0))
        return Completion(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=input_tokens * in_price + output_tokens * out_price,
            raw=data,
        )


class FallbackAdapter:
    """Tries the primary adapter and falls back to the deterministic one."""

    def __init__(self, primary: LLMAdapter, fallback: LLMAdapter | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or ScriptedAdapter()
        self.name = f"{primary.name}+{self.fallback.name}"

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Completion:
        try:
            return self.primary.complete(prompt, system, max_tokens, temperature)
        except Exception:
            return self.fallback.complete(prompt, system, max_tokens, temperature)


def build_adapter(kind: str = "auto", home: Path | None = None) -> LLMAdapter:
    if kind == "scripted":
        return ScriptedAdapter()
    key = operator_secret(home, ANTHROPIC_API_KEY) if home is not None else ""
    if kind == "anthropic":
        return AnthropicAdapter(api_key=key or None)
    anthropic = AnthropicAdapter(api_key=key or None)
    if anthropic.available:
        return FallbackAdapter(anthropic, ScriptedAdapter())
    return ScriptedAdapter()
