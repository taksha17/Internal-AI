"""Role agent registry.

Every role agent in PRD section 7 is defined here with its mission, sub-agents,
inputs, outputs, least-privilege capabilities, approval gates and KPIs.
Appendix B defines the starter set: orchestrator, data engineer, ML engineer and
evaluation agent are enabled by default; the rest are activated when the
workload justifies it.
"""

from __future__ import annotations

from functools import lru_cache

from llmforge.domain import Capability, GateType, RoleSpec

ORCHESTRATOR = RoleSpec(
    name="orchestrator",
    mission=(
        "Turn the PRD into a dependency-ordered plan, keep it current, and keep all agents "
        "within scope, schedule and budget."
    ),
    sub_agents=(
        "planner_decomposer",
        "dependency_and_schedule_tracker",
        "budget_monitor",
        "status_and_report_writer",
    ),
    inputs=("prd", "policy_decisions", "budget", "capacity"),
    outputs=("task_graph", "weekly_status_report", "risk_register", "gate_requests"),
    capabilities=frozenset(
        {
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.TASK_BOARD_WRITE,
            Capability.BUDGET_READ,
            Capability.AUDIT_READ,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.BUDGET_INCREASE,),
    kpis=("plan_accuracy", "blocked_task_age", "reporting_latency"),
    enabled_by_default=True,
)

DATA_ENGINEER = RoleSpec(
    name="data_engineer",
    mission=(
        "Produce clean, deduplicated, PII-scrubbed, versioned training corpora with full lineage."
    ),
    sub_agents=(
        "source_ingestion",
        "format_normalization",
        "deduplication",
        "pii_secret_scrubber",
        "quality_and_language_filter",
        "tokenizer_trainer",
        "dataset_versioning_and_lineage",
        "data_split_manager",
    ),
    inputs=("approved_source_list", "data_owner_rules", "schema_docs"),
    outputs=("versioned_datasets", "tokenizer", "data_quality_report", "lineage_graph"),
    capabilities=frozenset(
        {
            Capability.DATA_READ,
            Capability.DATA_WRITE,
            Capability.DATASET_WRITE,
            Capability.CATALOG_READ,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(
        GateType.NEW_DATA_SOURCE,
        GateType.DATASET_RELEASE,
        GateType.EVAL_SET_FREEZE,
    ),
    kpis=("pii_leakage_rate", "post_dedup_duplicate_rate", "throughput", "lineage_coverage"),
    enabled_by_default=True,
)

DATA_SCIENTIST = RoleSpec(
    name="data_scientist",
    mission="Understand the data and define how success is measured.",
    sub_agents=(
        "corpus_profiler",
        "eval_set_builder",
        "baseline_builder",
        "error_analyst",
        "ablation_analyst",
    ),
    inputs=("dataset_candidate", "data_quality_report", "use_cases"),
    outputs=("corpus_profile", "draft_eval_sets", "baselines", "error_analysis"),
    capabilities=frozenset(
        {
            Capability.DATA_READ,
            Capability.CATALOG_READ,
            Capability.EVAL_READ,
            Capability.EVAL_WRITE,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.EVAL_SET_FREEZE,),
    kpis=("eval_coverage", "baseline_quality", "analysis_turnaround"),
)

RESEARCHER = RoleSpec(
    name="researcher",
    mission=(
        "Propose architectures, sizes and training recipes grounded in published research "
        "and scaling-law estimates."
    ),
    sub_agents=(
        "literature_scanner",
        "architecture_and_hyperparameter_proposer",
        "scaling_law_estimator",
        "experiment_designer",
    ),
    inputs=("prd", "corpus_profile", "compute_constraints"),
    outputs=("design_memos", "experiment_plans", "compute_estimates"),
    capabilities=frozenset(
        {
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.BUDGET_READ,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.TRAINING_RUN_COST,),
    kpis=("proposal_hit_rate", "estimate_accuracy"),
)

ML_ENGINEER = RoleSpec(
    name="ml_engineer",
    mission="Implement and run training correctly.",
    sub_agents=(
        "model_implementation",
        "training_config_writer",
        "loss_and_instability_debugger",
        "checkpoint_and_resume_manager",
        "reproducibility_checker",
    ),
    inputs=("design_memo", "dataset_version", "tokenizer", "compute_budget"),
    outputs=("training_code", "configs", "run_logs", "checkpoints", "run_summaries"),
    capabilities=frozenset(
        {
            Capability.DATA_READ,
            Capability.DATASET_WRITE,
            Capability.TRAIN_RUN,
            Capability.JOB_SUBMIT,
            Capability.MODEL_WRITE,
            Capability.MODEL_READ,
            Capability.BUDGET_READ,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.TRAINING_RUN_COST,),
    kpis=("run_success_rate", "wasted_gpu_hours", "reproducibility_pass_rate"),
    enabled_by_default=True,
)

INFRA_ENGINEER = RoleSpec(
    name="infra_engineer",
    mission="Make compute available, efficient, observable and capped.",
    sub_agents=(
        "environment_and_container_builder",
        "job_scheduler",
        "gpu_utilization_monitor",
        "cost_tracker",
        "storage_and_checkpoint_lifecycle",
        "alerting",
    ),
    inputs=("job_specs", "budget_caps", "environment_requirements"),
    outputs=("reproducible_environments", "job_submissions", "utilization_reports", "cost_reports"),
    capabilities=frozenset(
        {
            Capability.JOB_SUBMIT,
            Capability.BUDGET_READ,
            Capability.AUDIT_READ,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.BUDGET_INCREASE,),
    kpis=("gpu_utilization", "idle_gpu_cost", "job_queue_wait"),
)

AI_ENGINEER = RoleSpec(
    name="ai_engineer",
    mission="Turn trained models into useful, served capabilities.",
    sub_agents=(
        "task_data_builder",
        "fine_tuning_runner",
        "embedding_and_retrieval_builder",
        "inference_optimizer",
        "prompt_and_template_author",
    ),
    inputs=("task_model", "task_data", "serving_requirements"),
    outputs=("task_tuned_models", "retrieval_index", "inference_configs"),
    capabilities=frozenset(
        {
            Capability.DATA_READ,
            Capability.MODEL_READ,
            Capability.MODEL_WRITE,
            Capability.TRAIN_RUN,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.TRAINING_RUN_COST, GateType.MODEL_PROMOTION),
    kpis=("task_accuracy_vs_baseline", "p95_latency", "cost_per_1k_requests"),
)

SOFTWARE_ENGINEER = RoleSpec(
    name="software_engineer",
    mission="Build the services and tooling around the models.",
    sub_agents=(
        "api_service_builder",
        "test_writer",
        "ci_cd_maintainer",
        "code_reviewer",
        "documentation_writer",
    ),
    inputs=("model_artifacts", "service_requirements", "access_policy"),
    outputs=("inference_api", "retrieval_service", "ci_pipelines", "runbooks"),
    capabilities=frozenset(
        {
            Capability.MODEL_READ,
            Capability.DEPLOY,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.PRODUCTION_DEPLOYMENT,),
    kpis=("test_coverage", "change_failure_rate", "review_turnaround"),
)

EVALUATION = RoleSpec(
    name="evaluation",
    mission="Be the single source of truth on model quality.",
    sub_agents=(
        "benchmark_runner",
        "regression_detector",
        "contamination_checker",
        "report_generator",
    ),
    inputs=("model_candidate", "frozen_eval_sets", "baselines"),
    outputs=("eval_reports", "regression_alerts", "promotion_recommendation"),
    capabilities=frozenset(
        {
            Capability.EVAL_READ,
            Capability.EVAL_WRITE,
            Capability.EVAL_FREEZE,
            Capability.MODEL_READ,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(GateType.MODEL_PROMOTION,),
    kpis=("eval_reproducibility", "regression_detection_rate", "time_to_report"),
    enabled_by_default=True,
)

SECURITY_COMPLIANCE = RoleSpec(
    name="security_compliance",
    mission="Maintain the audit trail and catch policy violations early.",
    sub_agents=(
        "access_auditor",
        "license_provenance_checker",
        "red_team_tester",
        "governance_writer",
    ),
    inputs=("audit_log", "lineage_graph", "license_register"),
    outputs=("audit_logs", "license_register", "model_cards", "risk_assessments"),
    capabilities=frozenset(
        {
            Capability.AUDIT_READ,
            Capability.DATA_READ,
            Capability.CATALOG_READ,
            Capability.MODEL_READ,
            Capability.TASK_READ,
            Capability.TASK_WRITE,
            Capability.GATE_REQUEST,
        }
    ),
    gates=(),
    kpis=("audit_completeness", "time_to_flag_policy_violation"),
)

ROLES: tuple[RoleSpec, ...] = (
    ORCHESTRATOR,
    DATA_ENGINEER,
    DATA_SCIENTIST,
    RESEARCHER,
    ML_ENGINEER,
    INFRA_ENGINEER,
    AI_ENGINEER,
    SOFTWARE_ENGINEER,
    EVALUATION,
    SECURITY_COMPLIANCE,
)

STARTER_SET = tuple(role.name for role in ROLES if role.enabled_by_default)


@lru_cache(maxsize=1)
def _registry() -> dict[str, RoleSpec]:
    return {role.name: role for role in ROLES}


def get_role(name: str) -> RoleSpec:
    try:
        return _registry()[name]
    except KeyError as exc:
        raise KeyError(f"unknown role agent: {name}") from exc


def all_roles() -> tuple[RoleSpec, ...]:
    return ROLES


def starter_roles() -> tuple[RoleSpec, ...]:
    return tuple(role for role in ROLES if role.name in STARTER_SET)


def reviewer_for(role_name: str) -> str:
    """Every builder needs a separate reviewer (principle 6).

    The evaluation agent reviews model-producing work; the orchestrator reviews
    planning work; data work is reviewed by security/compliance once activated.
    """
    mapping = {
        "orchestrator": "ml_engineer",
        "data_engineer": "security_compliance",
        "data_scientist": "evaluation",
        "researcher": "ml_engineer",
        "ml_engineer": "evaluation",
        "infra_engineer": "ml_engineer",
        "ai_engineer": "evaluation",
        "software_engineer": "ml_engineer",
        "evaluation": "security_compliance",
        "security_compliance": "orchestrator",
    }
    reviewer = mapping.get(role_name)
    if reviewer is None:
        raise KeyError(f"no reviewer mapped for role: {role_name}")
    return reviewer


def enabled_roles(active: set[str] | None = None) -> tuple[RoleSpec, ...]:
    if active is None:
        return starter_roles()
    return tuple(role for role in ROLES if role.name in active)
