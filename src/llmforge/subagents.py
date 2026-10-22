"""Sub-agents: the third level of the hierarchy (PRD section 6.1).

The PRD allows exactly three levels: orchestrator, role agents, sub-agents.
Sub-agents are task-scoped, they inherit their parent's permissions, and they
cannot spawn further sub-agents, because each handoff loses information.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmforge.domain import Capability


@dataclass(frozen=True, slots=True)
class SubAgentSpec:
    name: str
    mission: str
    produces: str
    capabilities: frozenset[Capability] = frozenset()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mission": self.mission,
            "produces": self.produces,
            "capabilities": sorted(str(item) for item in self.capabilities),
        }


SUBAGENTS: dict[str, tuple[SubAgentSpec, ...]] = {
    "orchestrator": (
        SubAgentSpec(
            "planner_decomposer", "Split the goal into ordered tasks with owners.", "task_graph"
        ),
        SubAgentSpec(
            "dependency_and_schedule_tracker",
            "Keep the dependency graph and the schedule honest.",
            "dependency_report",
        ),
        SubAgentSpec("budget_monitor", "Track spend against caps and forecast.", "budget_report"),
        SubAgentSpec(
            "status_and_report_writer", "Write the status and risk report.", "status_report"
        ),
    ),
    "data_engineer": (
        SubAgentSpec("source_ingestion", "Read approved sources into records.", "ingested_records"),
        SubAgentSpec(
            "format_normalization",
            "Normalize formats and strip boilerplate.",
            "normalized_records",
        ),
        SubAgentSpec(
            "deduplication",
            "Remove exact and near duplicates with statistics.",
            "dedup_report",
        ),
        SubAgentSpec("pii_secret_scrubber", "Redact PII and secrets.", "scrub_report"),
        SubAgentSpec(
            "quality_and_language_filter",
            "Filter on quality and language.",
            "quality_report",
        ),
        SubAgentSpec(
            "tokenizer_trainer", "Train the tokenizer on the cleaned corpus.", "tokenizer"
        ),
        SubAgentSpec(
            "dataset_versioning_and_lineage",
            "Version the dataset and record its lineage.",
            "dataset_version",
        ),
        SubAgentSpec(
            "data_split_manager",
            "Split data and guard against leakage.",
            "split_report",
        ),
    ),
    "data_scientist": (
        SubAgentSpec("corpus_profiler", "Profile the corpus.", "corpus_profile"),
        SubAgentSpec("eval_set_builder", "Draft eval sets.", "draft_eval_sets"),
        SubAgentSpec("baseline_builder", "Build the baseline to beat.", "baseline"),
        SubAgentSpec("error_analyst", "Analyse errors by cluster.", "error_analysis"),
        SubAgentSpec("ablation_analyst", "Design ablations.", "ablation_plan"),
    ),
    "researcher": (
        SubAgentSpec("literature_scanner", "Find published work that applies.", "literature_notes"),
        SubAgentSpec(
            "architecture_and_hyperparameter_proposer",
            "Propose architectures and hyperparameters.",
            "architecture_proposal",
        ),
        SubAgentSpec("scaling_law_estimator", "Estimate compute and cost.", "compute_estimate"),
        SubAgentSpec("experiment_designer", "Design the experiment.", "experiment_plan"),
    ),
    "ml_engineer": (
        SubAgentSpec(
            "model_implementation", "Implement the model and training code.", "model_code"
        ),
        SubAgentSpec("training_config_writer", "Write the training config.", "training_config"),
        SubAgentSpec(
            "loss_and_instability_debugger",
            "Diagnose loss spikes and divergence.",
            "debug_report",
        ),
        SubAgentSpec(
            "checkpoint_and_resume_manager",
            "Own checkpointing and resume.",
            "checkpoint_state",
        ),
        SubAgentSpec(
            "reproducibility_checker",
            "Verify the run reproduces from its record.",
            "reproducibility_report",
        ),
    ),
    "infra_engineer": (
        SubAgentSpec(
            "environment_and_container_builder", "Build reproducible environments.", "environment"
        ),
        SubAgentSpec("job_scheduler", "Schedule jobs within quotas and caps.", "job_submission"),
        SubAgentSpec("gpu_utilization_monitor", "Track utilisation.", "utilization_report"),
        SubAgentSpec("cost_tracker", "Track cost by run and agent.", "cost_report"),
        SubAgentSpec(
            "storage_and_checkpoint_lifecycle",
            "Manage storage and checkpoint retention.",
            "storage_report",
        ),
        SubAgentSpec("alerting", "Raise alerts on the signals that matter.", "alert_report"),
    ),
    "ai_engineer": (
        SubAgentSpec(
            "task_data_builder", "Build task data from internal tickets and docs.", "task_pairs"
        ),
        SubAgentSpec("fine_tuning_runner", "Run fine-tuning.", "fine_tune_run"),
        SubAgentSpec(
            "embedding_and_retrieval_builder", "Build embeddings and the retrieval index.", "index"
        ),
        SubAgentSpec("inference_optimizer", "Tune batching and quantisation.", "inference_config"),
        SubAgentSpec(
            "prompt_and_template_author", "Author prompts and templates.", "prompt_templates"
        ),
    ),
    "software_engineer": (
        SubAgentSpec("api_service_builder", "Build the inference API.", "service_code"),
        SubAgentSpec("test_writer", "Write tests.", "test_suite"),
        SubAgentSpec("ci_cd_maintainer", "Keep the pipelines green.", "pipeline"),
        SubAgentSpec("code_reviewer", "Review changes.", "review_notes"),
        SubAgentSpec("documentation_writer", "Write the runbooks.", "documentation"),
    ),
    "evaluation": (
        SubAgentSpec("benchmark_runner", "Run the frozen benchmarks.", "benchmark_results"),
        SubAgentSpec(
            "regression_detector", "Detect regressions against the baseline.", "regression_report"
        ),
        SubAgentSpec(
            "contamination_checker", "Check train and eval contamination.", "contamination_report"
        ),
        SubAgentSpec("report_generator", "Generate the eval report.", "eval_report"),
    ),
    "security_compliance": (
        SubAgentSpec("access_auditor", "Audit access events.", "access_audit"),
        SubAgentSpec(
            "license_provenance_checker", "Check licences and provenance.", "license_register"
        ),
        SubAgentSpec("red_team_tester", "Run adversarial probes.", "red_team_report"),
        SubAgentSpec(
            "governance_writer", "Write model cards and risk assessments.", "governance_docs"
        ),
    ),
}


def subagents_for(role_name: str) -> tuple[SubAgentSpec, ...]:
    return SUBAGENTS.get(role_name, ())


def total_subagents() -> int:
    return sum(len(items) for items in SUBAGENTS.values())
