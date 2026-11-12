"""The harness: one layer in front of the model, and the record it leaves.

The harness holds what a prompt retyped per call site cannot: the prompt variant
that was measured, the client context that fits this model's window, the bounded
repair of an answer that fails its own check, the tool surface a gate admitted,
and the backend that answered. An answer reports all of it beside the hash of the
prompt it was asked and what the call cost, because a result that cannot name its
harness is a result nobody can reproduce.

Spending and recording are part of the call rather than a courtesy of the caller:
a harness given a budget or a log writes to it every time it generates, and
writes a denial when a cap or the kill switch stops it before it does.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.budget import BudgetManager
from internal_ai.domain import AuditOutcome, RoleSpec
from internal_ai.harness.backends import BackendChoice
from internal_ai.harness.conditioning import (
    VARIANTS,
    ConditionedPrompt,
    ModelSignature,
    TokenCounter,
    condition,
    estimated_tokens,
    variant_for,
)
from internal_ai.harness.mcp import McpDenied, McpError, McpToolkit
from internal_ai.harness.repair import RepairOutcome, run_with_repair
from internal_ai.training.alpha.answer import selection_candidates
from internal_ai.training.alpha.citations import AskHit, score_answer
from internal_ai.training.alpha.memorisation import probe

HARNESS_VERSION = "1"
"""What this layer promises, named on every answer beside the model and variant."""

MAX_QUESTION_CHARS = 4000
"""A longer string is a document. Question-shaped input arrives through retrieval."""

MAX_HISTORY_TURNS = 20
"""History is carried verbatim, so its length is capped rather than trimmed."""

DEFAULT_DEADLINE_S = 120.0

MAX_TOOL_ROUNDS = 2
"""A tool result restarts the answer once; two rounds is the whole allowance."""

MAX_OBSERVATION_CHARS = 2000
"""What a tool returns joins the client context, so its length is capped there."""

TOOL_PROTOCOL = (
    'To use one, reply with only `TOOL: <name> {"argument": "value"}` and nothing else; '
    "otherwise answer the question as usual."
)
"""The one wording the harness parses, so a model is told what it may ask for."""

_TOOL_LINE = re.compile(
    r"^\s*TOOL:\s*(?P<name>[A-Za-z0-9_.\-]+)\s*(?P<arguments>\{.*\})?\s*$", re.DOTALL
)


class HarnessRefused(RuntimeError):
    """The harness did not call the model, and this is why."""


class RequestRefused(HarnessRefused):
    """The request is outside what this harness will put in front of a model."""


class BudgetRefused(HarnessRefused):
    """A cap or the kill switch stopped the call before it was paid for."""


class HarnessDeadlineExceeded(HarnessRefused):
    """The answer ran past its deadline, so it is reported as incomplete, not late."""


@dataclass(frozen=True, slots=True)
class HarnessRequest:
    question: str
    hits: tuple[AskHit, ...] = ()
    excerpts: tuple[str, ...] = ()
    system: str = ""
    history: tuple[tuple[str, str], ...] = ()
    max_tokens: int = 400
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class HarnessAnswer:
    text: str
    passed: bool
    problems: tuple[str, ...]
    model: str
    variant: str
    backend: str
    attempts: int
    exhausted: bool
    prompt_tokens: int
    prompt_sha256: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "passed": self.passed,
            "problems": list(self.problems),
            "model": self.model,
            "variant": self.variant,
            "backend": self.backend,
            "attempts": self.attempts,
            "exhausted": self.exhausted,
            "prompt_tokens": self.prompt_tokens,
            "prompt_sha256": self.prompt_sha256,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "tool_calls": self.tool_calls,
        }


def prompt_digest(text: str) -> str:
    """The prompt as a report may carry it: an identifier, never the text itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def problems_with(text: str, hits: list[AskHit], training_texts: tuple[str, ...]) -> str:
    """What is wrong with an answer, as one correction sentence, or ``\"\"``."""
    problems = list(score_answer(text, hits).mismatches)
    if training_texts:
        recall = probe(text, list(training_texts))
        problems.extend(f"repeats {span!r} from the training data" for span in recall.repeated[:3])
    return "; ".join(problems)


class Harness:
    """Conditioning, generation, checking, repair and accounting for one model signature."""

    def __init__(
        self,
        backend: BackendChoice,
        signature: ModelSignature,
        *,
        variant: str = "selection-v1",
        max_attempts: int = 2,
        counter: TokenCounter | None = None,
        training_texts: tuple[str, ...] = (),
        tools: McpToolkit | None = None,
        audit: AuditLog | None = None,
        budget: BudgetManager | None = None,
        actor: str = "harness",
        actor_kind: str = "agent",
        deadline_s: float = DEFAULT_DEADLINE_S,
    ) -> None:
        if deadline_s <= 0:
            raise ValueError("a harness deadline has to be positive seconds")
        variant_for(variant)
        self.backend = backend
        self.adapter = backend.adapter
        self.signature = signature
        self.variant = variant
        self.max_attempts = max_attempts
        self.counter = counter or estimated_tokens
        self.training_texts = tuple(training_texts)
        self.tools = tools
        self.audit = audit
        self.budget = budget
        self.actor = actor
        self.actor_kind = actor_kind
        self.deadline_s = deadline_s

    def describe(self) -> dict[str, Any]:
        """What answered, in the terms a report has to print beside a score."""
        return {
            "harness": HARNESS_VERSION,
            "model": self.signature.name,
            "model_family": self.signature.family,
            "window_tokens": self.signature.window_tokens,
            "variant": self.variant,
            "variant_source": VARIANTS[self.variant].source,
            "backend": self.backend.name,
            "external": self.backend.external,
            "max_attempts": self.max_attempts,
            "deadline_s": self.deadline_s,
            "tools": (
                None
                if self.tools is None
                else {"server": self.tools.server.name, "gate_id": self.tools.server.gate_id}
            ),
        }

    def answer(
        self,
        request: HarnessRequest,
        *,
        audit: AuditLog | None = None,
        actor: str | None = None,
        actor_kind: str | None = None,
    ) -> HarnessAnswer:
        log = audit if audit is not None else self.audit
        who = actor if actor is not None else self.actor
        who_kind = actor_kind if actor_kind is not None else self.actor_kind
        started = time.monotonic()
        self._guard(request)
        hits = list(request.hits)
        self._require_budget(log, who, who_kind)
        answered: dict[str, str] = {}
        meter = {"usd": 0.0}

        def generate(prompt: str) -> str:
            elapsed = time.monotonic() - started
            if elapsed > self.deadline_s:
                raise HarnessDeadlineExceeded(
                    f"the harness deadline of {self.deadline_s:.1f}s passed after {elapsed:.1f}s; "
                    "an incomplete answer is reported rather than a late one"
                )
            completion = self.adapter.complete(
                prompt, max_tokens=request.max_tokens, temperature=request.temperature
            )
            answered["model"] = completion.model
            if completion.cost_usd:
                meter["usd"] += completion.cost_usd
                if self.budget is not None:
                    self.budget.record_spend(
                        completion.cost_usd,
                        who,
                        notes=f"harness:{self.variant}:{self.signature.name}",
                    )
            return completion.text

        observations: list[str] = []
        conditioned = self._condition(request, hits, observations)

        def validate(text: str) -> str | None:
            # A tool request is an attempt that is over, not an answer that is wrong.
            if self.tools is not None and self._tool_request(text) is not None:
                return None
            return problems_with(text, hits, self.training_texts) or None

        outcome: RepairOutcome | None = None
        for round_index in range(MAX_TOOL_ROUNDS + 1):
            outcome = run_with_repair(generate, conditioned.text, validate, self.max_attempts)
            call = self._tool_request(outcome.text) if self.tools is not None else None
            if call is None or round_index == MAX_TOOL_ROUNDS:
                break
            observations.append(
                self._observe(
                    *call, log=log, who=who, who_kind=who_kind, round_index=round_index + 1
                )
            )
            conditioned = self._condition(request, hits, observations)
        if outcome is None:  # pragma: no cover - the loop runs at least once
            raise HarnessRefused("the harness produced no attempt")
        found = problems_with(outcome.text, hits, self.training_texts)
        if self.tools is not None and self._tool_request(outcome.text) is not None:
            # A request for a tool is not an answer, whatever the implies about citations.
            found = "; ".join(part for part in (found, "a tool request is not an answer") if part)
        result = HarnessAnswer(
            text=outcome.text.strip(),
            passed=not found,
            problems=tuple(part for part in found.split("; ") if part),
            model=answered.get("model", self.adapter.name),
            variant=conditioned.variant,
            backend=self.backend.name,
            attempts=outcome.used,
            exhausted=outcome.exhausted,
            prompt_tokens=conditioned.prompt_tokens,
            prompt_sha256=prompt_digest(conditioned.text),
            latency_ms=int((time.monotonic() - started) * 1000),
            cost_usd=round(meter["usd"], 8),
            tool_calls=len(observations),
        )
        if log is not None:
            log.record(
                actor=who,
                actor_kind=who_kind,
                action="harness.answer",
                subject=self.signature.name,
                outcome=AuditOutcome.SUCCESS if result.passed else AuditOutcome.FAILURE,
                details={
                    "harness": HARNESS_VERSION,
                    "variant": result.variant,
                    "backend": result.backend,
                    "model": result.model,
                    "attempts": result.attempts,
                    "exhausted": result.exhausted,
                    "passed": result.passed,
                    "problems": len(result.problems),
                    "prompt_tokens": result.prompt_tokens,
                    "window_tokens": self.signature.window_tokens,
                    "prompt_sha256": result.prompt_sha256,
                    "latency_ms": result.latency_ms,
                    "cost_usd": result.cost_usd,
                    "tool_calls": result.tool_calls,
                    "tools": None if self.tools is None else self.tools.server.name,
                },
                cost_usd=result.cost_usd,
            )
        return result

    @staticmethod
    def _guard(request: HarnessRequest) -> None:
        if not request.question.strip():
            raise RequestRefused("a harness question cannot be empty")
        if len(request.question) > MAX_QUESTION_CHARS:
            raise RequestRefused(
                f"the question is {len(request.question)} characters; this harness asks at most "
                f"{MAX_QUESTION_CHARS}, because a longer one is a document, not a question"
            )
        if len(request.history) > MAX_HISTORY_TURNS:
            raise RequestRefused(
                f"{len(request.history)} turns of history; this harness carries at most "
                f"{MAX_HISTORY_TURNS} and will not drop the rest silently"
            )
        if request.max_tokens <= 0:
            raise RequestRefused("max_tokens must leave the answer somewhere to go")

    def _require_budget(self, log: AuditLog | None, actor: str, actor_kind: str) -> None:
        """Ask the ledger before a call, not after: a refusal costs nothing."""
        if self.budget is None:
            return
        decision = self.budget.check(actor)
        if decision.allowed:
            return
        if log is not None:
            log.record(
                actor=actor,
                actor_kind=actor_kind,
                action="harness.answer",
                subject=self.signature.name,
                outcome=AuditOutcome.DENIED,
                details={
                    "harness": HARNESS_VERSION,
                    "variant": self.variant,
                    "backend": self.backend.name,
                    "reason": decision.reason,
                },
            )
        raise BudgetRefused(decision.reason)

    def _condition(
        self, request: HarnessRequest, hits: list[AskHit], observations: Sequence[str]
    ) -> ConditionedPrompt:
        return condition(
            request.question,
            selection_candidates(hits, list(request.excerpts)),
            self.signature,
            system=self._context(request.system, observations),
            history=request.history,
            variant=self.variant,
            counter=self.counter,
        )

    def _context(self, system: str, observations: Sequence[str] = ()) -> str:
        """Tools join the client context, which is conditioned, never the measured tail."""
        parts: list[str] = []
        if self.tools is not None:
            fragment = self.tools.prompt_fragment()
            if fragment:
                parts.extend((fragment, TOOL_PROTOCOL))
        parts.extend(observations)
        if system.strip():
            parts.append(system.strip())
        return "\n".join(parts)

    def _tool_request(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """The tool an answer is asking for, or ``None`` when it is an answer."""
        match = _TOOL_LINE.match(text.strip())
        if match is None:
            return None
        raw = (match.group("arguments") or "").strip() or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        return match.group("name"), arguments

    def _observe(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        log: AuditLog | None,
        who: str,
        who_kind: str,
        round_index: int,
    ) -> str:
        """Run one tool and turn its result into the context of the next attempt."""
        if self.tools is None:
            raise McpDenied("this harness has no MCP server configured")
        offered = {offer.name for offer in self.tools.offered_tools()}
        if name not in offered:
            return self._observed(
                log, who, who_kind, name, round_index, False, f"there is no tool named {name!r}"
            )
        try:
            body = self.tools.call(name, arguments, actor=who)
        except McpError as exc:
            return self._observed(
                log, who, who_kind, name, round_index, False, f"the tool failed: {exc}"
            )
        except McpDenied:
            self._observed(
                log, who, who_kind, name, round_index, False, "the gate no longer admits it"
            )
            raise
        return self._observed(
            log,
            who,
            who_kind,
            name,
            round_index,
            True,
            f"Tool {name} returned: {body[:MAX_OBSERVATION_CHARS]}",
        )

    def _observed(
        self,
        log: AuditLog | None,
        who: str,
        who_kind: str,
        name: str,
        round_index: int,
        ok: bool,
        observation: str,
    ) -> str:
        if log is not None:
            log.record(
                actor=who,
                actor_kind=who_kind,
                action="harness.tool",
                subject=f"{self.tools.server.name}.{name}" if self.tools else name,
                outcome=AuditOutcome.SUCCESS if ok else AuditOutcome.FAILURE,
                details={
                    "harness": HARNESS_VERSION,
                    "round": round_index,
                    "ok": ok,
                    "gate_id": self.tools.server.gate_id if self.tools else None,
                },
            )
        return observation

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        role: RoleSpec | None = None,
        actor: str = "harness",
    ) -> str:
        if self.tools is None:
            raise McpDenied("this harness has no MCP server configured")
        return self.tools.call(tool_name, arguments, role=role, actor=actor)
