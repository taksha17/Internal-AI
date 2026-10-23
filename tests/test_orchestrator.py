"""Orchestrator behaviour: dependency order, review pairing, gates, loop detection."""

from __future__ import annotations

from internal_ai.domain import BudgetScope, GateType, TaskState
from internal_ai.planner import TaskDraft, TemplatePlanner
from internal_ai.roles import all_roles, get_role, reviewer_for


def test_every_role_has_a_distinct_reviewer() -> None:
    for role in all_roles():
        reviewer = reviewer_for(role.name)
        assert reviewer != role.name
        assert reviewer in {item.name for item in all_roles()}


def test_roles_are_least_privilege(platform) -> None:
    orchestrator = get_role("orchestrator")
    assert not orchestrator.can(
        platform.runtime.capabilities_for
        and __import__("internal_ai.domain", fromlist=["Capability"]).Capability.DATA_READ
    )
    assert not orchestrator.can(
        __import__("internal_ai.domain", fromlist=["Capability"]).Capability.DEPLOY
    )


def test_plan_respects_dependencies(platform) -> None:
    drafts = TemplatePlanner().plan("demo", {})
    platform.orchestrator.submit_plan(drafts)
    platform.orchestrator.approve_plan("program_lead")
    ready = {task.id for task in platform.orchestrator.ready_tasks()}
    assert "tsk_policy" in ready
    assert "tsk_tiny_run" not in ready


def test_gated_task_is_blocked_until_approved(platform) -> None:
    drafts = TemplatePlanner().plan("demo", {})
    tasks = platform.orchestrator.submit_plan(drafts)
    platform.orchestrator.approve_plan("program_lead")
    corpus = next(task for task in tasks if task.id == "tsk_corpus")
    assert corpus.gate_id
    platform.orchestrator.run(max_iterations=5)
    corpus = platform.store.get_task("tsk_corpus")
    assert corpus.state == TaskState.BLOCKED
    gate = platform.store.get_gate(corpus.gate_id)
    assert str(gate.gate_type) == str(GateType.DATASET_RELEASE)
    for approver in gate.required_approvers:
        platform.gates.decide(gate.id, approver, approved=True, approver_kind="human")
    released = platform.orchestrator.release_unblocked()
    assert "tsk_corpus" in {task.id for task in released}
    assert platform.store.get_task("tsk_corpus").state == TaskState.APPROVED


def test_every_task_gets_a_separate_reviewer(platform) -> None:
    drafts = TemplatePlanner().plan("demo", {})
    tasks = platform.orchestrator.submit_plan(drafts)
    for task in tasks:
        assert task.reviewer_role != task.owner_role


def test_budget_blocks_dispatch_when_exhausted(platform) -> None:
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 0.0)
    platform.budget.record_spend(1.0, agent="orchestrator")
    drafts = [
        TaskDraft(
            key="single",
            title="Do work",
            description="",
            kind=TaskDraft(
                key="single", title="x", description="", kind="ops", owner_role="orchestrator"
            ).kind,
            owner_role="orchestrator",
            acceptance_criteria=["done"],
        )
    ]
    platform.orchestrator.submit_plan(drafts)
    platform.orchestrator.approve_plan("program_lead")
    platform.orchestrator.run(max_iterations=3)
    task = platform.store.get_task("tsk_single")
    assert task.state == TaskState.BLOCKED
    assert "budget" in (task.blocked_reason or "")


def test_repeated_failure_escalates_to_a_human(platform) -> None:
    class FailingAdapter:
        name = "failing"

        def complete(self, prompt, system="", max_tokens=2048, temperature=0.0):
            from internal_ai.llm import Completion

            return Completion(text='{"status":"failed","summary":"boom"}', model="failing")

    platform.runtime.llm = FailingAdapter()
    drafts = [
        TaskDraft(
            key="flaky",
            title="Flaky task",
            description="",
            kind="ops",
            owner_role="orchestrator",
            acceptance_criteria=["done"],
        )
    ]
    platform.orchestrator.submit_plan(drafts)
    platform.orchestrator.approve_plan("program_lead")
    platform.orchestrator.run(max_iterations=10)
    task = platform.store.get_task("tsk_flaky")
    assert task.state == TaskState.ESCALATED
    assert platform.orchestrator.escalations


def test_status_report_tracks_spend_and_gates(platform) -> None:
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", 100.0)
    drafts = TemplatePlanner().plan("demo", {})
    platform.orchestrator.submit_plan(drafts)
    platform.orchestrator.approve_plan("program_lead")
    platform.orchestrator.run(max_iterations=3)
    report = platform.orchestrator.status_report()
    assert report.tasks_by_state
    assert report.budget_usd == 100.0
    assert isinstance(report.open_gates, list)
