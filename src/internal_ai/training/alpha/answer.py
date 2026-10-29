"""An answer may quote Ask hits. A line number stays on the file it belongs to."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from internal_ai.llm import LLMAdapter
from internal_ai.training.alpha.citations import AskHit, score_answer
from internal_ai.training.alpha.memorisation import probe


@dataclass(frozen=True, slots=True)
class CitedAnswer:
    text: str
    passed: bool
    model: str
    mismatches: tuple[str, ...]


def cited_answer(
    adapter: LLMAdapter,
    question: str,
    hits: list[AskHit],
    excerpts: list[str] | None = None,
    training_texts: Sequence[str] | None = None,
) -> CitedAnswer:
    """Ask the on-prem runner, then drop a reply that miscites or recites its corpus."""
    lines = []
    for index, hit in enumerate(hits):
        excerpt = ""
        if excerpts is not None and index < len(excerpts):
            excerpt = excerpts[index].strip()
        lines.append(f"{hit.path}:{hit.line} {excerpt}".rstrip())
    prompt = (
        "Answer only from the citations below. When you name a file and a line number, "
        "they must be one of these pairs. Do not move a line number onto a different file.\n"
        + "\n".join(lines)
        + f"\nQuestion: {question.strip()}"
    )
    completion = adapter.complete(
        prompt,
        system="You answer from the cited internal files only.",
        max_tokens=400,
        temperature=0.0,
    )
    report = score_answer(completion.text, hits)
    recall = probe(completion.text, training_texts or [])
    if report.passed and recall.passed:
        return CitedAnswer(completion.text.strip(), True, completion.model, ())
    problems = list(report.mismatches)
    problems.extend(f"repeats {span!r} from the training data" for span in recall.repeated[:3])
    safe = "\n".join(f"{hit.path}:{hit.line}" for hit in hits)
    return CitedAnswer(safe, False, completion.model, tuple(problems))
