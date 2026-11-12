"""The harness layer: what it refuses, how it repairs, and what it records."""

from __future__ import annotations

import hashlib
import time

import pytest

from internal_ai.domain import BudgetScope
from internal_ai.harness import (
    HARNESS_VERSION,
    MAX_HISTORY_TURNS,
    BackendChoice,
    BoundaryRefusal,
    BudgetRefused,
    Harness,
    HarnessDeadlineExceeded,
    HarnessRequest,
    ModelSignature,
    PromptTooLong,
    RequestRefused,
    choose_backend,
    condition,
    preamble_budget,
    run_with_repair,
)
from internal_ai.llm import Completion
from internal_ai.training.alpha.citations import AskHit
from internal_ai.training.alpha.pairs import CANDIDATE_LIMIT


class Rehearsing:
    """An adapter that answers with one scripted text per call."""

    name = "rehearsing"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Completion:
        self.prompts.append(prompt)
        text = self.texts[min(len(self.prompts) - 1, len(self.texts) - 1)]
        return Completion(text=text, model="rehearsing-3b")


class SlowAdapter(Rehearsing):
    """An adapter that takes longer than a short deadline allows."""

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Completion:
        time.sleep(0.05)
        return super().complete(prompt, system, max_tokens, temperature)


class Recorder:
    """A stand-in for the audit log: it records what it was told, and nothing else."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record(self, **event: object) -> None:
        self.events.append(event)


def test_unknown_prompt_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown prompt variant"):
        condition("where?", [("a.txt:1", "text")], ModelSignature("m", 8192), variant="chatty")


def test_candidate_list_longer_than_measured_is_refused() -> None:
    offered = [(f"a.txt:{index}", "text") for index in range(CANDIDATE_LIMIT + 1)]
    with pytest.raises(ValueError, match="measured with"):
        condition("where?", offered, ModelSignature("m", 8192))


def test_small_window_refuses_a_flagship_sized_context() -> None:
    context = " ".join(["word"] * 400)
    with pytest.raises(PromptTooLong, match="client context"):
        condition("where?", [], ModelSignature("7b", 4096), system=context)


def test_large_window_carries_the_same_context() -> None:
    context = " ".join(["word"] * 400)
    conditioned = condition("where?", [], ModelSignature("frontier", 128000), system=context)
    assert conditioned.preamble_tokens > preamble_budget(ModelSignature("7b", 4096))
    assert conditioned.text.startswith("System: word")


def test_a_prompt_that_does_not_fit_the_window_is_refused_not_trimmed() -> None:
    offered = [(f"a.txt:{index}", " ".join(["word"] * 40)) for index in range(CANDIDATE_LIMIT)]
    with pytest.raises(PromptTooLong, match="once the answer is reserved"):
        condition("where?", offered, ModelSignature("tiny", 600))


def test_repair_corrects_one_bad_answer() -> None:
    answers = iter(["The answer is a.txt:9.", "The answer is a.txt:3."])
    prompts: list[str] = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    def validate(text: str) -> str:
        return "" if "a.txt:3" in text else "line 9 belongs to b.txt"

    outcome = run_with_repair(generate, "Question: where?\nAnswer:", validate, max_attempts=2)
    assert outcome.exhausted is False
    assert outcome.used == 2
    assert outcome.text == "The answer is a.txt:3."
    assert "line 9 belongs to b.txt" in prompts[1]
    assert prompts[1].startswith("Question: where?\nAnswer:")


def test_repair_stops_at_the_cap() -> None:
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return "still wrong"

    outcome = run_with_repair(generate, "Q", lambda text: "nope", max_attempts=2)
    assert outcome.exhausted is True
    assert len(calls) == 2


@pytest.mark.parametrize("cap", [0, 4])
def test_attempt_budget_outside_the_cap_is_refused(cap: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        run_with_repair(lambda prompt: "text", "Q", lambda text: "", max_attempts=cap)


def test_external_backend_is_refused_for_records() -> None:
    with pytest.raises(BoundaryRefusal):
        choose_backend("anthropic")


def test_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        choose_backend("openrouter")


def test_scripted_backend_is_local() -> None:
    choice = choose_backend("scripted")
    assert choice.external is False
    assert choice.adapter.name == "scripted"


def test_harness_repairs_then_reports_what_answered() -> None:
    adapter = Rehearsing("The answer is a.txt:9.", "The answer is a.txt:3.")
    harness = Harness(
        BackendChoice("rehearsing", adapter, False),
        ModelSignature("qwen-3b", 8192),
    )
    answer = harness.answer(
        HarnessRequest(
            question="where?",
            hits=(AskHit("a.txt", 3), AskHit("b.txt", 9)),
            excerpts=("the thing", "another thing"),
        )
    )
    assert answer.passed is True
    assert answer.attempts == 2
    assert answer.exhausted is False
    assert answer.variant == "selection-v1"
    assert answer.backend == "rehearsing"
    assert answer.model == "rehearsing-3b"
    assert len(adapter.prompts) == 2


def test_harness_records_the_run_without_the_answer_text() -> None:
    audit = Recorder()
    adapter = Rehearsing("The answer is a.txt:3.")
    harness = Harness(
        BackendChoice("rehearsing", adapter, False),
        ModelSignature("qwen-3b", 8192),
    )
    harness.answer(
        HarnessRequest(question="where?", hits=(AskHit("a.txt", 3),), excerpts=("the thing",)),
        audit=audit,  # type: ignore[arg-type]
        actor="alice",
        actor_kind="human",
    )
    assert len(audit.events) == 1
    event = audit.events[0]
    assert event["action"] == "harness.answer"
    assert event["actor"] == "alice"
    details = event["details"]
    assert isinstance(details, dict)
    assert details["attempts"] == 1
    assert details["variant"] == "selection-v1"
    assert details["backend"] == "rehearsing"
    assert "text" not in details


def test_harness_reports_a_failure_it_could_not_repair() -> None:
    adapter = Rehearsing("The answer is a.txt:9.")
    harness = Harness(
        BackendChoice("rehearsing", adapter, False),
        ModelSignature("qwen-3b", 8192),
        max_attempts=2,
    )
    answer = harness.answer(
        HarnessRequest(
            question="where?",
            hits=(AskHit("a.txt", 3), AskHit("b.txt", 9)),
            excerpts=("the thing", "another thing"),
        )
    )
    assert answer.passed is False
    assert answer.exhausted is True
    assert answer.problems


def test_an_empty_question_is_refused_before_any_generation() -> None:
    adapter = Rehearsing("The answer is a.txt:3.")
    harness = Harness(BackendChoice("rehearsing", adapter, False), ModelSignature("qwen-3b", 8192))
    with pytest.raises(RequestRefused, match="cannot be empty"):
        harness.answer(HarnessRequest(question="   "))
    assert adapter.prompts == []


def test_a_document_is_not_a_question() -> None:
    adapter = Rehearsing("The answer is a.txt:3.")
    harness = Harness(BackendChoice("rehearsing", adapter, False), ModelSignature("qwen-3b", 8192))
    with pytest.raises(RequestRefused, match="document, not a question"):
        harness.answer(HarnessRequest(question="find the clause " * 500))
    assert adapter.prompts == []


def test_history_longer_than_the_harness_carries_is_refused() -> None:
    adapter = Rehearsing("The answer is a.txt:3.")
    harness = Harness(BackendChoice("rehearsing", adapter, False), ModelSignature("qwen-3b", 8192))
    history = tuple(("user", "hello") for _ in range(MAX_HISTORY_TURNS + 1))
    with pytest.raises(RequestRefused, match="turns of history"):
        harness.answer(HarnessRequest(question="where?", history=history))


def test_an_answer_past_its_deadline_is_reported_not_waited_for() -> None:
    adapter = SlowAdapter("The answer is a.txt:9.")
    harness = Harness(
        BackendChoice("rehearsing", adapter, False),
        ModelSignature("qwen-3b", 8192),
        max_attempts=2,
        deadline_s=0.01,
    )
    with pytest.raises(HarnessDeadlineExceeded, match="deadline"):
        harness.answer(
            HarnessRequest(
                question="where?",
                hits=(AskHit("a.txt", 3), AskHit("b.txt", 9)),
                excerpts=("the thing", "another thing"),
            )
        )
    assert len(adapter.prompts) == 1


def test_the_answer_carries_the_prompt_digest_and_its_latency() -> None:
    adapter = Rehearsing("The answer is a.txt:3.")
    harness = Harness(BackendChoice("rehearsing", adapter, False), ModelSignature("qwen-3b", 8192))
    answer = harness.answer(
        HarnessRequest(question="where?", hits=(AskHit("a.txt", 3),), excerpts=("the thing",))
    )
    assert answer.prompt_sha256 == hashlib.sha256(adapter.prompts[0].encode()).hexdigest()
    assert answer.latency_ms >= 0
    assert answer.as_dict()["prompt_sha256"] == answer.prompt_sha256


def test_a_harness_built_with_a_log_records_without_being_asked_to() -> None:
    audit = Recorder()
    adapter = Rehearsing("The answer is a.txt:3.")
    harness = Harness(
        BackendChoice("rehearsing", adapter, False),
        ModelSignature("qwen-3b", 8192),
        audit=audit,  # type: ignore[arg-type]
    )
    harness.answer(
        HarnessRequest(question="where?", hits=(AskHit("a.txt", 3),), excerpts=("the thing",))
    )
    assert [event["action"] for event in audit.events] == ["harness.answer"]
    details = audit.events[0]["details"]
    assert isinstance(details, dict)
    assert len(str(details["prompt_sha256"])) == 64


def test_describe_names_everything_a_report_has_to_print() -> None:
    harness = Harness(
        BackendChoice("rehearsing", Rehearsing("text"), False), ModelSignature("qwen-3b", 8192)
    )
    described = harness.describe()
    assert described["harness"] == HARNESS_VERSION
    assert described["variant"] == "selection-v1"
    assert described["variant_source"] == "pairs.prompt_for_candidates"
    assert described["external"] is False
    assert described["tools"] is None


def test_the_kill_switch_stops_the_harness_before_it_spends(platform) -> None:  # type: ignore[no-untyped-def]
    platform.budget.engage_kill_switch()
    harness = Harness(
        BackendChoice("scripted", platform.runtime.llm, False),
        ModelSignature("scripted", 8192),
        budget=platform.budget,
        audit=platform.audit,
    )
    with pytest.raises(BudgetRefused, match="kill switch"):
        harness.answer(HarnessRequest(question="where?"))
    assert platform.store.list_audit(action="harness.answer")


def test_what_the_harness_spends_is_in_the_ledger(platform) -> None:  # type: ignore[no-untyped-def]
    harness = Harness(
        BackendChoice("scripted", platform.runtime.llm, False),
        ModelSignature("scripted", 8192),
        budget=platform.budget,
    )
    answer = harness.answer(HarnessRequest(question="where?"))
    assert answer.cost_usd > 0
    assert platform.store.spend_by(BudgetScope.AGENT, "harness") == pytest.approx(answer.cost_usd)
