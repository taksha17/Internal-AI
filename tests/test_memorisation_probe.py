"""An answer must not recite the corpus it was trained on.

A planted phrase is the test: it appears in a training record, and an answer
that repeats it word for word is refused even when every citation is correct.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.llm import Completion
from internal_ai.serving.service import Principal
from internal_ai.training.alpha.answer import cited_answer
from internal_ai.training.alpha.citations import AskHit
from internal_ai.training.alpha.memorisation import probe

PLANTED = "the mercury vault access code rotates on the ninth of every month at dawn"
PARAPHRASE = "The access code for the vault changes monthly, early in the day."


class Repeats:
    """An on-prem runner that answers by quoting its prompt back."""

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(
        self, prompt: str, system: str = "", max_tokens: int = 512, temperature: float = 0.0
    ):
        return Completion(text=self.text, model="fake")


def test_an_answer_that_repeats_a_span_is_reported_with_the_span() -> None:
    report = probe(f"According to the runbook, {PLANTED}.", [PLANTED])

    assert report.passed is False
    assert report.repeated == ("the mercury vault access code rotates on the",)


def test_a_rewritten_answer_passes() -> None:
    report = probe(PARAPHRASE, [PLANTED])

    assert report.passed is True
    assert report.repeated == ()


def test_a_short_overlap_is_not_a_recitation() -> None:
    report = probe("the vault code rotates on monday", [PLANTED])

    assert report.passed is True


def test_cited_answer_drops_a_reply_that_recites_its_training_data() -> None:
    hits = [AskHit("docs/runbook.md", 12)]
    recited = cited_answer(
        Repeats(f"See docs/runbook.md at line 12. {PLANTED}."),
        "when does the code rotate?",
        hits,
        [PLANTED],
        training_texts=[PLANTED],
    )

    assert recited.passed is False
    assert recited.text == "docs/runbook.md:12"
    assert "repeats" in recited.mismatches[0]


def test_cited_answer_keeps_a_rewritten_reply() -> None:
    hits = [AskHit("docs/runbook.md", 12)]
    answer = cited_answer(
        Repeats(f"See docs/runbook.md at line 12. {PARAPHRASE}"),
        "when does the code rotate?",
        hits,
        [PLANTED],
        training_texts=[PLANTED],
    )

    assert answer.passed is True
    assert PARAPHRASE in answer.text


def test_the_answer_endpoint_refuses_a_recited_reply_against_the_served_corpus(
    tmp_path: Path, monkeypatch
) -> None:
    from internal_ai.data.pipeline import DataPipeline, PipelineConfig
    from internal_ai.data.records import Source
    from internal_ai.serving import app as serving_app

    path = tmp_path / "runbook.csv"
    path.write_text(f"ticket,description\n1,{PLANTED}\n", encoding="utf-8")
    platform = Platform.create(root=tmp_path)
    built = DataPipeline(platform.store, platform.audit, tmp_path).run(
        [
            Source(
                source_id="src_runbook",
                path=path,
                acl=["engineering"],
                approved=True,
                owner="tester",
                text_field="description",
            )
        ],
        PipelineConfig(name="runbook"),
    )
    platform.store.add_model(
        ModelVersion(
            id="mdl_alpha",
            name="embedding-v1",
            version=1,
            content_hash="h",
            created_at="now",
            status="promoted",
            dataset_ids=[built.dataset.id],
        )
    )
    platform.close()

    monkeypatch.setenv("INTERNAL_AI_LOCAL_RUNNER", "http://127.0.0.1:9111")
    import internal_ai.llm as llm_module

    monkeypatch.setattr(llm_module, "LocalRunnerAdapter", lambda **kwargs: Repeats(PLANTED + "."))

    app = serving_app.create_serving_app(
        root=tmp_path,
        principals={
            "k": Principal(user_id="alice", groups=frozenset({"engineering"}), api_key="k")
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/answer",
            json={
                "question": "when does it rotate?",
                "hits": [{"path": "docs/runbook.md", "line": 12}],
            },
            headers={"X-Internal-Key": "k"},
        )

    payload = response.json()
    assert response.status_code == 200, response.text
    assert payload["citation_passed"] is False
    assert "repeats" in " ".join(payload["mismatches"])
