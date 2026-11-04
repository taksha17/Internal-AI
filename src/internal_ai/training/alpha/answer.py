"""An answer may quote Ask hits. A line number stays on the file it belongs to."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from internal_ai.llm import LLMAdapter
from internal_ai.training.alpha.citations import AskHit, score_answer
from internal_ai.training.alpha.memorisation import probe
from internal_ai.training.alpha.pairs import prompt_for, prompt_for_candidates, shown


@dataclass(frozen=True, slots=True)
class CitedAnswer:
    text: str
    passed: bool
    model: str
    mismatches: tuple[str, ...]
    prompt: str = ""


def selection_candidates(
    hits: list[AskHit], excerpts: Sequence[str] | None
) -> list[tuple[str, str]]:
    """The hits as the pairs builder shows them: ``(key, shown excerpt)``.

    The excerpts are truncated by the same cap the pairs use, so a candidate
    shown at serving time is byte-for-byte the candidate training saw.
    """
    candidates: list[tuple[str, str]] = []
    for index, hit in enumerate(hits):
        excerpt = ""
        if excerpts is not None and index < len(excerpts):
            excerpt = shown(excerpts[index])
        candidates.append((f"{hit.path}:{hit.line}", excerpt))
    return candidates


def with_context(prompt: str, system: str, history: Sequence[tuple[str, str]]) -> str:
    """Put the conversation above the selection prompt instead of inside it.

    The tail ``Question: … / Candidates: … / Answer:`` is the text the adapter was
    trained and scored in, so context a client sends joins it as a preamble: the
    part that decides the answer's shape stays byte-identical to training, and a
    system prompt or an earlier turn is read rather than silently dropped.
    """
    lines: list[str] = []
    if system.strip():
        lines.append(f"System: {system.strip()}")
    for role, text in history:
        if text.strip():
            lines.append(f"{role.strip().capitalize()}: {text.strip()}")
    if not lines:
        return prompt
    return "\n".join(lines) + "\n\n" + prompt


def cited_answer(
    adapter: LLMAdapter,
    question: str,
    hits: list[AskHit],
    excerpts: list[str] | None = None,
    training_texts: Sequence[str] | None = None,
    *,
    system: str = "",
    history: Sequence[tuple[str, str]] = (),
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> CitedAnswer:
    """Ask the on-prem runner, then drop a reply that miscites or recites its corpus.

    The prompt is the pairs builder's selection prompt, unchanged: the model
    was trained and scored inside that text, so serving asks in it too instead
    of in a wording it has never seen. With no hits at all the open-question
    prompt is the shared one.
    """
    if hits:
        prompt = prompt_for_candidates(question.strip(), selection_candidates(hits, excerpts))
    else:
        prompt = prompt_for(question.strip())
    prompt = with_context(prompt, system, history)
    completion = adapter.complete(prompt, max_tokens=max_tokens, temperature=temperature)
    report = score_answer(completion.text, hits)
    recall = probe(completion.text, training_texts or [])
    if report.passed and recall.passed:
        return CitedAnswer(completion.text.strip(), True, completion.model, (), prompt)
    problems = list(report.mismatches)
    problems.extend(f"repeats {span!r} from the training data" for span in recall.repeated[:3])
    safe = "\n".join(f"{hit.path}:{hit.line}" for hit in hits)
    return CitedAnswer(safe, False, completion.model, tuple(problems), prompt)
