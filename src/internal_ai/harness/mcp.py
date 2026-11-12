"""Model Context Protocol client: tools behind a gate, never before it.

A connected server is a data source. It reads records this install has not
reviewed, and its tool descriptions are text a model will follow, so the same
``new_data_source_access`` gate that admits a file admits a server: two people
approve it, the approval names one server, and the check runs again on every
call, because a listing of tools is not a licence to use one.

Only stdio servers are reachable. A remote endpoint is a network egress and would
need ``NET_EGRESS``, an allowlist and a redaction plan; it is refused here rather
than half-supported.
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from internal_ai.audit import AuditLog
from internal_ai.domain import (
    AuditOutcome,
    Capability,
    GateRequest,
    GateStatus,
    GateType,
    RoleSpec,
)
from internal_ai.store import Store

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT_S = 20.0
MAX_DESCRIPTION_CHARS = 400

ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV", "SYSTEMROOT")
"""What a server needs to run. An API key in the parent's environment is not a grant."""

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class McpError(RuntimeError):
    """The server could not be started, answered badly, or reported a failure."""


class McpDenied(PermissionError):
    """The gate, the server, the tool or the role does not permit this call."""


def approved_server_gate(store: Store, gate_id: str, server: str) -> GateRequest:
    """The gate that admits this server, or a refusal naming what is wrong with it."""
    gate = store.get_gate(gate_id)
    if gate is None:
        raise McpDenied(f"no gate covers MCP server {server!r}: a server is a data source")
    if str(gate.gate_type) != str(GateType.NEW_DATA_SOURCE):
        raise McpDenied(
            f"gate {gate_id} is a {gate.gate_type}; an MCP server is admitted by "
            "new_data_source_access and nothing else"
        )
    if gate.subject != server:
        raise McpDenied(
            f"gate {gate_id} covers {gate.subject!r}, not {server!r}: one approval is one server"
        )
    if str(gate.status) != str(GateStatus.APPROVED):
        raise McpDenied(
            f"MCP server {server!r} is {gate.status}: its tools are not offered before "
            "the gate passes"
        )
    return gate


def clean_description(text: Any) -> str:
    """A tool description is untrusted text, never an instruction.

    Control characters are dropped and the length capped, so a server cannot use
    its own metadata to flood a window or smuggle terminal escapes into a log.
    """
    return " ".join(_CONTROL.sub(" ", str(text)).split())[:MAX_DESCRIPTION_CHARS]


def scrubbed_env(source: dict[str, str] | None = None) -> dict[str, str]:
    """The child's environment: what it needs to run, and no credential."""
    env = source if source is not None else dict(os.environ)
    return {name: env[name] for name in ENV_ALLOWLIST if name in env}


class Transport(Protocol):
    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    def notify(self, method: str, params: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class StdioTransport:
    """One server process, newline-delimited JSON-RPC, a deadline on every read."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> None:
        if not command:
            raise ValueError("an MCP server needs a command to run")
        self.timeout = timeout
        self._next_id = 0
        try:
            self._process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=scrubbed_env(env),
                cwd=str(cwd) if cwd is not None else None,
            )
        except OSError as exc:
            raise McpError(f"could not start MCP server {command[0]!r}: {exc}") from exc

    def _write(self, message: dict[str, Any]) -> None:
        stream = self._process.stdin
        if stream is None:
            raise McpError("the MCP server has no input stream")
        stream.write(json.dumps(message, sort_keys=True) + "\n")
        stream.flush()

    def _readline(self) -> str:
        stream = self._process.stdout
        if stream is None:
            raise McpError("the MCP server has no output stream")
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpError(f"the MCP server did not answer within {self.timeout}s")
            ready, _, _ = select.select([stream], [], [], remaining)
            if not ready:
                continue
            line = stream.readline()
            if not line:
                raise McpError("the MCP server closed its output")
            return line

    def _reply(self, want_id: int) -> dict[str, Any]:
        while True:
            line = self._readline()
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise McpError(
                    f"the MCP server wrote a line that is not JSON: {line[:80]!r}"
                ) from exc
            if message.get("id") != want_id:
                continue
            if "error" in message:
                error = message["error"]
                detail = error.get("message") if isinstance(error, dict) else error
                raise McpError(f"the MCP server refused {message.get('id')}: {detail}")
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        return self._reply(self._next_id)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)


@dataclass(frozen=True, slots=True)
class ToolOffer:
    server: str
    name: str
    description: str

    def as_line(self) -> str:
        return f"- {self.name}: {self.description}"


@dataclass(frozen=True, slots=True)
class McpServer:
    """One connected server, and the gate that has to admit it first."""

    name: str
    command: tuple[str, ...]
    gate_id: str
    capability: Capability = Capability.DATA_READ

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError(f"MCP server {self.name!r} has no command")
        if self.capability is Capability.NET_EGRESS:
            raise McpDenied(
                "a stdio server does not need NET_EGRESS; a remote endpoint would, and "
                "this client does not speak to one"
            )


class McpToolkit:
    """One server's tools, offered only after its gate passes and checked on each call."""

    def __init__(
        self,
        server: McpServer,
        store: Store,
        *,
        transport: Transport | None = None,
        transport_factory: Callable[[Sequence[str]], Transport] | None = None,
        audit: AuditLog | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.server = server
        self.store = store
        self.audit = audit
        self._transport = transport
        self._factory = transport_factory or (
            lambda command: StdioTransport(command, timeout=timeout)
        )
        self._offers: list[ToolOffer] | None = None
        # One pipe carries every request and reply, so two callers reading it at
        # once would each take the other's answer. The lock is re-entrant because
        # a call asks the offer list before it asks the server.
        self._lock = threading.RLock()

    def __enter__(self) -> McpToolkit:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._transport is not None:
                self._transport.close()
                self._transport = None

    def _started(self) -> Transport:
        approved_server_gate(self.store, self.server.gate_id, self.server.name)
        if self._transport is None:
            self._transport = self._factory(self.server.command)
            self._transport.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "internal-ai-harness", "version": "0.1"},
                },
            )
            self._transport.notify("notifications/initialized", {})
            self._offers = None
        return self._transport

    def offered_tools(self) -> list[ToolOffer]:
        """What this server offers. Nothing at all until a person has admitted it."""
        with self._lock:
            return self._listed()

    def _listed(self) -> list[ToolOffer]:
        approved_server_gate(self.store, self.server.gate_id, self.server.name)
        if self._offers is not None:
            return list(self._offers)
        result = self._started().request("tools/list", {})
        offers: list[ToolOffer] = []
        for item in result.get("tools") or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            offers.append(
                ToolOffer(
                    self.server.name,
                    str(item["name"]),
                    clean_description(item.get("description", "")),
                )
            )
        offers.sort(key=lambda offer: offer.name)
        self._offers = offers
        return list(offers)

    def prompt_fragment(self) -> str:
        """The tool list a model may be shown, empty when the gate has not passed."""
        offers = self.offered_tools()
        if not offers:
            return ""
        lines = [f"Tools on {self.server.name}:"]
        lines.extend(offer.as_line() for offer in offers)
        return "\n".join(lines)

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        role: RoleSpec | None = None,
        actor: str = "harness",
    ) -> str:
        """Call one tool. The gate and the capability are checked here, not at listing."""
        with self._lock:
            return self._called(tool_name, arguments, role=role, actor=actor)

    def _called(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        role: RoleSpec | None,
        actor: str,
    ) -> str:
        approved_server_gate(self.store, self.server.gate_id, self.server.name)
        offered = {offer.name for offer in self._listed()}
        if tool_name not in offered:
            raise McpDenied(
                f"MCP server {self.server.name!r} does not offer {tool_name!r}; "
                f"offered: {', '.join(sorted(offered)) or 'nothing'}"
            )
        if role is not None and not role.can(self.server.capability):
            self._record(
                actor,
                tool_name,
                AuditOutcome.DENIED,
                {"missing_capability": str(self.server.capability)},
            )
            raise McpDenied(
                f"role {role.name} lacks capability {self.server.capability} for "
                f"tool {self.server.name}.{tool_name}"
            )
        result = self._started().request("tools/call", {"name": tool_name, "arguments": arguments})
        body = "\n".join(
            str(item.get("text", ""))
            for item in result.get("content") or []
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if result.get("isError"):
            self._record(actor, tool_name, AuditOutcome.FAILURE, {"error": True})
            raise McpError(
                f"{self.server.name}.{tool_name} failed: {body or 'no detail was returned'}"
            )
        self._record(actor, tool_name, AuditOutcome.SUCCESS, {"chars": len(body)})
        return body

    def _record(
        self, actor: str, tool_name: str, outcome: AuditOutcome, details: dict[str, Any]
    ) -> None:
        if self.audit is None:
            return
        self.audit.record(
            actor=actor,
            actor_kind="agent",
            action="mcp.tool",
            subject=f"{self.server.name}.{tool_name}",
            outcome=outcome,
            details={"gate_id": self.server.gate_id, **details},
        )
