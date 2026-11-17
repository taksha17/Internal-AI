"""Tool registry with least-privilege enforcement (architecture principle 4).

Every tool call is capability-checked against the calling role and written to
the audit log, so tool use is reviewable after the fact.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from internal_ai.store import Store

from internal_ai.audit import AuditLog
from internal_ai.domain import AuditOutcome, Capability, RoleSpec
from internal_ai.ids import new_id


class ToolDenied(PermissionError):
    pass


# Which task a call belongs to, set by the orchestrator around the agent's turn. A
# context variable rather than a parameter so the agent's four invoke sites do not
# each have to carry an id they never look at.
CURRENT_TASK: ContextVar[str | None] = ContextVar("internal_ai_tool_task", default=None)


FETCH_TIMEOUT_S = 10.0
FETCH_MAX_BYTES = 1_000_000


def _fetch_url_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as response:
        body = response.read(FETCH_MAX_BYTES + 1)
    if len(body) > FETCH_MAX_BYTES:
        raise ToolDenied("response exceeds the fetch size cap")
    return body


def fetch_url_bytes(url: str) -> bytes:
    """Fetch a URL outside the agent tool path.

    The person's service path uses this after enforcing the same refusals
    as ``ToolRegistry.web_fetch``; the name is public so callers do not
    reach past it into the private fetcher.
    """
    return _fetch_url_bytes(url)


def approved_web_allowlist(store: object, gate_id: str) -> list:
    gate = store.get_gate(gate_id)  # type: ignore[union-attr]
    if gate is None or str(gate.gate_type) != "web_access":
        raise ToolDenied("no approved web-access gate covers this fetch")
    if str(gate.status) != "approved":
        raise ToolDenied("no approved web-access gate covers this fetch")
    allowlist = gate.evidence.get("url_allowlist") or []
    if isinstance(allowlist, str):
        allowlist = allowlist.replace(",", " ").split()
    return [str(host).lower() for host in allowlist]


@dataclass(slots=True)
class Tool:
    name: str
    capability: Capability
    description: str
    handler: Callable[..., Any]
    side_effecting: bool = False


class ToolRegistry:
    def __init__(
        self,
        audit: AuditLog,
        workspace_dir: Path,
        artifacts_dir: Path,
        store: Store | None = None,
    ) -> None:
        self.audit = audit
        self.workspace_dir = workspace_dir
        self.artifacts_dir = artifacts_dir
        # Without a store the registry cannot remember a call, which is what a tool
        # used outside a task gets; refusing instead would break every caller that
        # only wants to run one tool.
        self.store = store
        self._web_gate_check: Callable[[str], list[str]] | None = None
        self._tools: dict[str, Tool] = {}
        self._register_builtin()

    def _register_builtin(self) -> None:
        self.register(
            Tool(
                name="write_artifact",
                capability=Capability.TASK_WRITE,
                description="Write a versioned artifact into the shared artifact store.",
                handler=self._write_artifact,
                side_effecting=True,
            )
        )
        self.register(
            Tool(
                name="read_artifact",
                capability=Capability.TASK_READ,
                description="Read an artifact previously produced by another agent.",
                handler=self._read_artifact,
            )
        )
        self.register(
            Tool(
                name="web_fetch",
                capability=Capability.NET_EGRESS,
                description="Fetch a URL covered by an approved web-access gate.",
                handler=self._web_fetch,
                side_effecting=True,
            )
        )
        self.register(
            Tool(
                name="list_artifacts",
                capability=Capability.TASK_READ,
                description="List artifacts in the shared artifact store.",
                handler=self._list_artifacts,
            )
        )

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def set_web_gate_check(self, check: Callable[[str], list[str]]) -> None:
        self._web_gate_check = check

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolDenied(f"unknown tool: {name}") from exc

    def invoke(
        self,
        role: RoleSpec,
        tool_name: str,
        args: dict[str, Any],
        actor: str | None = None,
    ) -> Any:
        tool = self.get(tool_name)
        if not role.can(tool.capability):
            self.audit.record(
                actor=actor or role.name,
                action=f"tool.denied.{tool_name}",
                subject=tool_name,
                outcome=AuditOutcome.DENIED,
                actor_kind="agent",
                inputs={"args": args, "missing_capability": str(tool.capability)},
            )
            raise ToolDenied(
                f"role {role.name} lacks capability {tool.capability} for tool {tool_name}"
            )
        replay = self._prior_call(tool, args)
        if replay is not None:
            self.audit.record(
                actor=actor or role.name,
                action=f"tool.replayed.{tool_name}",
                subject=tool_name,
                outcome=AuditOutcome.SUCCESS,
                actor_kind="agent",
                inputs={"args": args},
                details={"task_id": CURRENT_TASK.get()},
            )
            return replay
        result = tool.handler(**args)
        self._remember_call(tool, args, result)
        self.audit.record(
            actor=actor or role.name,
            action=f"tool.{tool_name}",
            subject=tool_name,
            outcome=AuditOutcome.SUCCESS,
            actor_kind="agent",
            inputs={"args": args},
            details={"result_type": type(result).__name__},
        )
        return result

    def _call_key(self, tool: Tool, args: dict[str, Any]) -> tuple[str, str] | None:
        """The key a side effect is remembered by, or None when nothing is remembered.

        Only a tool that changes something is deduplicated, and only inside a task:
        a read must always see the current state, and a call outside a task has no
        attempt whose retry it could be duplicating.
        """
        task_id = CURRENT_TASK.get()
        if self.store is None or not task_id or not tool.side_effecting:
            return None
        body = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
        return task_id, hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _prior_call(self, tool: Tool, args: dict[str, Any]) -> Any:
        key = self._call_key(tool, args)
        if key is None:
            return None
        assert self.store is not None
        row = self.store.tool_call(key[0], tool.name, key[1])
        if row is None:
            return None
        if row["state"] != "done":
            raise ToolDenied(
                f"{tool.name} was started for this task and never finished; the task has to "
                "be resumed before the call is retried, or its side effect runs twice"
            )
        return json.loads(row["result"])

    def _remember_call(self, tool: Tool, args: dict[str, Any], result: Any) -> None:
        key = self._call_key(tool, args)
        if key is None:
            return
        assert self.store is not None
        row = self.store.tool_call(key[0], tool.name, key[1])
        payload = json.dumps(result, sort_keys=True, default=str)
        if row is None:
            call_id = new_id("tcl")
            self.store.begin_tool_call(
                call_id=call_id, task_id=key[0], tool_name=tool.name, args_hash=key[1]
            )
        else:
            call_id = row["id"]
        self.store.finish_tool_call(call_id, state="done", result=payload)

    def _write_artifact(self, name: str, content: str, kind: str = "text") -> str:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        suffix = {"text": ".md", "json": ".json", "code": ".py"}.get(kind, ".txt")
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in name)
        target = self.artifacts_dir / f"{safe}{suffix}"
        target.write_text(content, encoding="utf-8")
        return str(target)

    def _read_artifact(self, path: str) -> str:
        target = Path(path).resolve()
        store = self.artifacts_dir.resolve()
        if target != store and store not in target.parents:
            raise ToolDenied(f"artifact outside of the shared store: {path}")
        if not target.exists():
            raise FileNotFoundError(path)
        return target.read_text(encoding="utf-8")

    def _list_artifacts(self) -> str:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        entries = sorted(p.name for p in self.artifacts_dir.iterdir() if p.is_file())
        return json.dumps({"artifacts": entries}, indent=2)

    def _web_fetch(self, url: str, gate_id: str) -> bytes:
        from internal_ai.config import offline_mode

        parts = urlsplit(url)
        if parts.scheme != "https":
            raise ToolDenied("web fetch allows https URLs only")
        if self._web_gate_check is None:
            raise ToolDenied("web fetch is not wired to a gate checker on this install")
        allowlist = self._web_gate_check(gate_id)
        if (parts.hostname or "").lower() not in allowlist:
            raise ToolDenied("host is not on the approved allowlist")
        if offline_mode():
            raise ToolDenied("web fetch is disabled while the install is offline")
        return _fetch_url_bytes(url)
