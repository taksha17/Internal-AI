"""Tool registry with least-privilege enforcement (architecture principle 4).

Every tool call is capability-checked against the calling role and written to
the audit log, so tool use is reviewable after the fact.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.domain import AuditOutcome, Capability, RoleSpec


class ToolDenied(PermissionError):
    pass


@dataclass(slots=True)
class Tool:
    name: str
    capability: Capability
    description: str
    handler: Callable[..., Any]
    side_effecting: bool = False


class ToolRegistry:
    def __init__(self, audit: AuditLog, workspace_dir: Path, artifacts_dir: Path) -> None:
        self.audit = audit
        self.workspace_dir = workspace_dir
        self.artifacts_dir = artifacts_dir
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
                name="list_artifacts",
                capability=Capability.TASK_READ,
                description="List artifacts in the shared artifact store.",
                handler=self._list_artifacts,
            )
        )

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

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
        result = tool.handler(**args)
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
