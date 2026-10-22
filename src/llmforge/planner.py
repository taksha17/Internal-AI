"""Planning: turn a goal into a dependency-ordered task graph (FR-1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from llmforge.domain import TaskKind
from llmforge.llm import LLMAdapter, ScriptedAdapter


@dataclass(slots=True)
class TaskDraft:
    key: str
    title: str
    description: str
    kind: TaskKind
    owner_role: str
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    cost_estimate_usd: float = 0.0
    milestone: str = ""
    gate: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Planner:
    def plan(
        self, goal: str, context: dict[str, Any]
    ) -> list[TaskDraft]:  # pragma: no cover - interface
        raise NotImplementedError


FOUNDATION_TASKS: tuple[TaskDraft, ...] = (
    TaskDraft(
        key="policy",
        title="Resolve policy decisions P1-P5",
        description=(
            "Record an explicit human decision and rationale for each policy question that "
            "materially changes scope (PRD section 3). Until decided, platform defaults apply."
        ),
        kind=TaskKind.OPS,
        owner_role="orchestrator",
        dependencies=[],
        acceptance_criteria=[
            "P1-P5 each carry a value, a rationale and a decider",
            "Each decision is traceable to a policy_decision gate",
        ],
        cost_estimate_usd=0.2,
        milestone="Week 1",
    ),
    TaskDraft(
        key="inventory",
        title="Inventory approved data sources",
        description=(
            "Catalog sources, owners, volume, formats, sensitivity class and legal status for "
            "every candidate training source. Start with the lowest-risk sources (PRD section 11)."
        ),
        kind=TaskKind.DATA,
        owner_role="data_engineer",
        dependencies=["policy"],
        acceptance_criteria=[
            "Every source records owner, volume, format, sensitivity class and legal status",
            "Sources without an owner are flagged as blocked rather than ingested",
        ],
        cost_estimate_usd=0.5,
        milestone="Week 1",
    ),
    TaskDraft(
        key="tracker",
        title="Stand up shared tracker, task board and audit log",
        description=(
            "One monorepo, one experiment tracker, one task board, one data catalog with lineage, "
            "and an append-only audit log (architecture principle 2)."
        ),
        kind=TaskKind.INFRA,
        owner_role="data_engineer",
        dependencies=[],
        acceptance_criteria=[
            "Tasks, gates and audit events are queryable by every agent through the shared store",
            "Audit chain verification passes",
        ],
        cost_estimate_usd=0.1,
        milestone="Week 1",
    ),
    TaskDraft(
        key="harness",
        title="Build evaluation harness with placeholder eval sets",
        description=(
            "Frozen eval sets, metric definitions, baseline slot and report generation, ready for "
            "real eval sets in Week 3."
        ),
        kind=TaskKind.EVAL,
        owner_role="evaluation",
        dependencies=["tracker"],
        acceptance_criteria=[
            "A report can be produced against a placeholder eval set",
            "Metrics match the framework in PRD section 12",
        ],
        cost_estimate_usd=0.3,
        milestone="Week 1",
    ),
    TaskDraft(
        key="model_impl",
        title="Implement from-scratch embedding model architecture",
        description=(
            "Random-initialized transformer trained on internal data only. No pretrained weights, "
            "no distillation from third-party models (non-goals 2.2, policy P3)."
        ),
        kind=TaskKind.RESEARCH,
        owner_role="ml_engineer",
        dependencies=["policy"],
        acceptance_criteria=[
            "Architecture, parameter count and token budget are documented",
            "Training code contains no pretrained weight loading path",
        ],
        cost_estimate_usd=0.4,
        milestone="Week 1",
    ),
    TaskDraft(
        key="tiny_run",
        title="Validate training code on a tiny run",
        description=(
            "Short training run on a small corpus slice with fixed seed, checkpointing and metric "
            "streaming, to prove the code path before spending real GPU hours (FR-10 to FR-13)."
        ),
        kind=TaskKind.TRAIN,
        owner_role="ml_engineer",
        dependencies=["model_impl", "harness"],
        acceptance_criteria=[
            "Run reproduces from commit, config, dataset version and seed",
            "Loss decreases over the run and metrics stream to the tracker",
        ],
        cost_estimate_usd=1.5,
        milestone="Week 1",
    ),
    TaskDraft(
        key="corpus",
        title="Build cleaned corpus v1 for pilot sources",
        description=(
            "Ingest approved sources, normalize formats, remove exact and near duplicates, scrub "
            "PII and secrets, filter on quality and language, and version the result "
            "(FR-5 to FR-9)."
        ),
        kind=TaskKind.DATA,
        owner_role="data_engineer",
        dependencies=["inventory"],
        acceptance_criteria=[
            "Lineage coverage is 100 percent",
            "PII scrub statistics are attached to the release",
        ],
        cost_estimate_usd=1.0,
        milestone="Week 2",
        gate="dataset_release",
    ),
    TaskDraft(
        key="tokenizer",
        title="Train tokenizer v1 and report vocabulary coverage",
        description=(
            "Train a tokenizer on the cleaned corpus and report coverage of internal jargon and "
            "code (FR-9)."
        ),
        kind=TaskKind.DATA,
        owner_role="data_engineer",
        dependencies=["corpus"],
        acceptance_criteria=[
            "Tokenizer is trained only on released corpus data",
            "Coverage report lists unknown-token rate per source",
        ],
        cost_estimate_usd=0.4,
        milestone="Week 3",
    ),
    TaskDraft(
        key="eval_freeze",
        title="Freeze eval set v1 for the embedding pilot",
        description=(
            "Freeze the eval sets after human approval. After the freeze no agent may modify them "
            "(PRD 7.3, FR-14)."
        ),
        kind=TaskKind.EVAL,
        owner_role="evaluation",
        dependencies=["harness", "corpus"],
        acceptance_criteria=[
            "Eval set version is frozen and approved by a human",
            "Leakage check against the training corpus passed",
        ],
        cost_estimate_usd=0.3,
        milestone="Week 4",
        gate="eval_set_freeze",
    ),
    TaskDraft(
        key="embed_train",
        title="Train embedding model on corpus v1",
        description=(
            "Train the from-scratch embedding model on the released corpus and stream metrics to "
            "the tracker."
        ),
        kind=TaskKind.TRAIN,
        owner_role="ml_engineer",
        dependencies=["tokenizer", "eval_freeze"],
        acceptance_criteria=[
            "Run is reproducible and checkpointed",
            "Cost estimate was pre-registered and within cap",
        ],
        cost_estimate_usd=8.0,
        milestone="Week 5",
        gate="training_run_above_threshold",
    ),
    TaskDraft(
        key="embed_eval",
        title="Evaluate embedding model against baseline",
        description=(
            "Independent evaluation agent runs the frozen eval set and compares against the "
            "keyword-search baseline (PRD section 12)."
        ),
        kind=TaskKind.EVAL,
        owner_role="evaluation",
        dependencies=["embed_train"],
        acceptance_criteria=[
            "Recall@k, MRR and nDCG are reported against the baseline",
            "Contamination and leakage checks passed",
        ],
        cost_estimate_usd=0.5,
        milestone="Week 6",
    ),
    TaskDraft(
        key="promote",
        title="Promote embedding model to pilot",
        description=(
            "Promotion gate: eval results versus baseline, model card and red-team results "
            "(PRD section 14)."
        ),
        kind=TaskKind.OPS,
        owner_role="evaluation",
        dependencies=["embed_eval"],
        acceptance_criteria=[
            "Model card lists data sources, training config, eval results and limitations",
            "Promotion gate approved by ml lead and domain expert",
        ],
        cost_estimate_usd=0.2,
        milestone="Week 7",
        gate="model_promotion",
    ),
    TaskDraft(
        key="report",
        title="Write weekly status and risk report",
        description=(
            "Plan versus actual, blocked-task age, spend versus budget, open gates, risks and "
            "forecast to completion (PRD section 15)."
        ),
        kind=TaskKind.OPS,
        owner_role="orchestrator",
        dependencies=["embed_eval"],
        acceptance_criteria=[
            "Report includes spend by agent and open gates",
            "Risks carry an owner and a next action",
        ],
        cost_estimate_usd=0.1,
        milestone="Weekly",
    ),
)


class TemplatePlanner:
    """Deterministic planner built from the PRD milestones."""

    def __init__(self, tasks: tuple[TaskDraft, ...] = FOUNDATION_TASKS) -> None:
        self.tasks = tasks

    def plan(self, goal: str, context: dict[str, Any]) -> list[TaskDraft]:
        milestone = context.get("milestone")
        if milestone:
            prefix = f"{milestone}: "
            return [
                TaskDraft(
                    key=draft.key,
                    title=f"{prefix}{draft.title}",
                    description=draft.description,
                    kind=draft.kind,
                    owner_role=draft.owner_role,
                    dependencies=list(draft.dependencies),
                    acceptance_criteria=list(draft.acceptance_criteria),
                    cost_estimate_usd=draft.cost_estimate_usd,
                    milestone=draft.milestone,
                    gate=draft.gate,
                    metadata=dict(draft.metadata),
                )
                for draft in self.tasks
            ]
        return list(self.tasks)


class LLMPlanner:
    """Planner that asks an LLM for a task graph, with template fallback."""

    def __init__(self, llm: LLMAdapter | None = None, fallback: Planner | None = None) -> None:
        self.llm = llm or ScriptedAdapter()
        self.fallback = fallback or TemplatePlanner()

    def plan(self, goal: str, context: dict[str, Any]) -> list[TaskDraft]:
        prompt = (
            'Decompose the goal into tasks. Return JSON: {"tasks": [{"key", "title", '
            '"description", "kind", "owner_role", "dependencies": [], '
            '"acceptance_criteria": [], "cost_estimate_usd"}]}. Allowed kinds: plan, data, '
            "eval, research, train, infra, serve, review, gate, ops.\n"
            f"Goal: {goal}\nContext: {json.dumps(context, sort_keys=True, default=str)}"
        )
        try:
            completion = self.llm.complete(prompt, system="You plan work for an ML platform.")
            payload = completion.as_json() or {}
            drafts = self._parse(payload)
            if drafts:
                return drafts
        except Exception:
            pass
        return self.fallback.plan(goal, context)

    @staticmethod
    def _parse(payload: dict[str, Any]) -> list[TaskDraft]:
        drafts: list[TaskDraft] = []
        for item in payload.get("tasks", []):
            if not isinstance(item, dict):
                continue
            try:
                kind = TaskKind(str(item.get("kind", "ops")))
            except ValueError:
                kind = TaskKind.OPS
            drafts.append(
                TaskDraft(
                    key=str(item.get("key") or f"task{len(drafts)}"),
                    title=str(item.get("title", "untitled")),
                    description=str(item.get("description", "")),
                    kind=kind,
                    owner_role=str(item.get("owner_role", "orchestrator")),
                    dependencies=[str(dep) for dep in item.get("dependencies", [])],
                    acceptance_criteria=[str(item) for item in item.get("acceptance_criteria", [])],
                    cost_estimate_usd=float(item.get("cost_estimate_usd", 0.0)),
                    milestone=str(item.get("milestone", "")),
                    gate=item.get("gate"),
                )
            )
        return drafts
