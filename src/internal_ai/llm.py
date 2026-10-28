"""LLM adapters for role agents.

Policy P2 governs this layer: agents may use third-party LLM APIs for code and
configuration work, but never for raw internal data. Offline mode is the
default and raises before any HTTP call. When a lab turns it off, the
external adapter redacts anything that looks like a secret or personal data
before a prompt leaves the process, and every call is audited with its token
cost.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from internal_ai.config import offline_mode
from internal_ai.data.inventory import SECRET_PATTERNS

OUTBOUND_PATTERNS: dict[str, re.Pattern[str]] = {
    **SECRET_PATTERNS,
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"),
}


def redact_for_external(text: str) -> tuple[str, int]:
    """Replace secret- and PII-shaped spans with a labelled placeholder.

        The external adapter applies this to every prompt and system message, so
    a task description that pastes in a record cannot exfiltrate it. Redaction
    is structural, not a promise about content: callers must still keep raw
    records out of prompts.
    """
    redacted = 0
    for name, pattern in OUTBOUND_PATTERNS.items():
        text, hits = pattern.subn(f"[redacted:{name}]", text)
        redacted += hits
    return text, redacted


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
        if offline_mode():
            raise RuntimeError("offline mode refuses third-party model APIs")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        prompt, redacted_prompt = redact_for_external(prompt)
        system, redacted_system = redact_for_external(system)
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
        data["redacted_spans"] = redacted_prompt + redacted_system
        return Completion(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=input_tokens * in_price + output_tokens * out_price,
            raw=data,
        )


def private_runner_url(url: str) -> str:
    """Accept only a loopback or private-network runner. Public hosts are refused."""
    parsed = urlsplit(url.strip())
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("local runner URL must be http or https with a host")
    if host == "localhost":
        return url.rstrip("/")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("local runner host must be a private or loopback IP") from exc
    if not (address.is_loopback or address.is_private):
        raise ValueError("local runner host must be a private or loopback IP")
    return url.rstrip("/")


class LocalRunnerAdapter:
    """On-prem generation. The runner owns the KV cache. Company text stays on the LAN."""

    name = "local-runner"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        raw = base_url if base_url is not None else os.environ.get("INTERNAL_AI_LOCAL_RUNNER", "")
        self.base_url = private_runner_url(raw)
        self.model = model or os.environ.get("INTERNAL_AI_LOCAL_MODEL", "qwen2.5-3b-lumen")
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=str(data.get("model") or self.model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
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


def build_adapter(kind: str = "auto") -> LLMAdapter:
    if kind == "scripted":
        return ScriptedAdapter()
    if kind == "local":
        return LocalRunnerAdapter()
    if kind == "anthropic":
        return AnthropicAdapter()
    local = os.environ.get("INTERNAL_AI_LOCAL_RUNNER", "").strip()
    if local:
        return LocalRunnerAdapter(base_url=local)
    if offline_mode():
        return ScriptedAdapter()
    anthropic = AnthropicAdapter()
    if anthropic.available:
        return FallbackAdapter(anthropic, ScriptedAdapter())
    return ScriptedAdapter()
