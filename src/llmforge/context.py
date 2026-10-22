"""Wiring: one place that builds the platform services from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmforge.agent import AgentRuntime
from llmforge.audit import AuditLog
from llmforge.budget import BudgetManager
from llmforge.config import Settings
from llmforge.engine import EngineConfig, Orchestrator
from llmforge.gates import GateService
from llmforge.llm import LLMAdapter, build_adapter
from llmforge.policy import PolicyRegistry
from llmforge.store import Store
from llmforge.tools import ToolRegistry


@dataclass(slots=True)
class Forge:
    settings: Settings
    store: Store
    audit: AuditLog
    gates: GateService
    budget: BudgetManager
    tools: ToolRegistry
    runtime: AgentRuntime
    orchestrator: Orchestrator
    policy: PolicyRegistry

    @classmethod
    def create(
        cls,
        root: Path | None = None,
        llm: LLMAdapter | None = None,
        llm_kind: str = "auto",
        engine_config: EngineConfig | None = None,
        project: str = "default",
    ) -> Forge:
        settings = Settings.from_env(root)
        settings.ensure_dirs()
        store = Store(settings.db_path)
        audit = AuditLog(store)
        gates = GateService(store)
        budget = BudgetManager(store, project=project)
        tools = ToolRegistry(audit, settings.workspace_dir, settings.artifacts_dir)
        adapter = llm or build_adapter(llm_kind)
        runtime = AgentRuntime(store, audit, tools, budget, adapter)
        orchestrator = Orchestrator(
            store, audit, gates, budget, runtime, config=engine_config or EngineConfig()
        )
        policy = PolicyRegistry(store, gates)
        policy.seed()
        return cls(
            settings=settings,
            store=store,
            audit=audit,
            gates=gates,
            budget=budget,
            tools=tools,
            runtime=runtime,
            orchestrator=orchestrator,
            policy=policy,
        )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Forge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
