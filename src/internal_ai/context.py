"""Wiring: one place that builds the platform services from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from internal_ai import signing
from internal_ai.agent import AgentRuntime
from internal_ai.alerts import AlertEngine
from internal_ai.audit import AuditLog
from internal_ai.budget import BudgetManager
from internal_ai.config import Settings
from internal_ai.engine import EngineConfig, Orchestrator
from internal_ai.gates import GateService
from internal_ai.llm import LLMAdapter, build_adapter
from internal_ai.policy import PolicyRegistry
from internal_ai.storage import BlobStore, blob_store
from internal_ai.store import Store
from internal_ai.tools import ToolRegistry, approved_web_allowlist


@dataclass(slots=True)
class Platform:
    settings: Settings
    store: Store
    blobs: BlobStore
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
    ) -> Platform:
        settings = Settings.from_env(root)
        settings.ensure_dirs()
        store = Store(settings.database_url)
        # The signing key lives outside the database; an install without one can
        # still link events, it just cannot anchor a head.
        audit = AuditLog(
            store,
            signer=signing.signer_for(settings.home),
            anchor_dir=settings.home / "audit_anchors",
        )
        gates = GateService(store)
        budget = BudgetManager(store, project=project)
        tools = ToolRegistry(audit, settings.workspace_dir, settings.artifacts_dir)
        tools.set_web_gate_check(lambda gate_id: approved_web_allowlist(store, gate_id))
        adapter = llm or build_adapter(llm_kind)
        runtime = AgentRuntime(store, audit, tools, budget, adapter)
        orchestrator = Orchestrator(
            store, audit, gates, budget, runtime, config=engine_config or EngineConfig()
        )
        # Verified once at open: a chain that is already broken is something the
        # operator has to see before it records a day of work on top of it.
        health = audit.verify()
        if not health.ok:
            AlertEngine(store, audit, budget).raise_alert(
                "audit_integrity",
                "critical",
                "audit_events",
                f"the audit chain is broken at #{health.broken_at_seq}: {health.reason}",
                {"broken_at_seq": health.broken_at_seq, "checked": health.checked},
            )

        policy = PolicyRegistry(store, gates)
        policy.seed()
        return cls(
            settings=settings,
            store=store,
            # Local disk unless the install names an S3/MinIO bucket; a bad
            # setting refuses here rather than writing bytes somewhere else.
            blobs=blob_store(settings.home / "objects"),
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

    def __enter__(self) -> Platform:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
