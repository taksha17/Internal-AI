"""The KYC agent fills fake forms from evidence, through audited tools."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from internal_ai.data.kyc import (
    DEFAULT_FORMS,
    DOC_FAMILIES,
    KYC_FIELDS,
    build_corpus,
    evidence_lines,
    generate_forms,
)
from internal_ai.domain import Capability, RoleSpec
from internal_ai.services.kyc_agent import KycAgent, agent_role, extract, rank_lines
from internal_ai.tools import ToolDenied, ToolRegistry
from internal_ai.training.alpha.pairs import kyc_fill_text, kyc_prompt


def wired(platform, tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(platform.audit, tmp_path / "work", tmp_path / "artifacts")


def test_same_seed_same_corpus() -> None:
    assert (
        build_corpus(count=40, seed=3).content_hash == build_corpus(count=40, seed=3).content_hash
    )


def test_another_seed_another_corpus() -> None:
    assert (
        build_corpus(count=40, seed=3).content_hash != build_corpus(count=40, seed=4).content_hash
    )


def test_every_value_is_marked_fake() -> None:
    forms = generate_forms(count=DEFAULT_FORMS, seed=7)
    assert len(forms) == DEFAULT_FORMS
    assert len({form.customer_id for form in forms}) == DEFAULT_FORMS // 4
    for form in forms:
        assert tuple(form.values) == KYC_FIELDS
        assert form.customer_id.startswith("CUST-TEST-")
        assert form.values["phone"].startswith("+1-555-01")
        assert form.values["id_number"].startswith("SAMPLE-ID-")
        assert form.values["email"].endswith("@example.test")


def test_evidence_carries_truth_plus_decoys() -> None:
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7), ("DECOY LINE",))
    assert len(lines) == 6
    assert "DECOY LINE" in lines
    assert any(form.values["phone"] in line for line in lines)
    assert any(form.values["id_number"] in line for line in lines)


def test_clean_evidence_fills_perfectly(platform, tmp_path: Path) -> None:
    forms = generate_forms(count=8, seed=11)
    agent = KycAgent(wired(platform, tmp_path), agent_role())
    evidence = {form.form_id: evidence_lines(form, random.Random(11)) for form in forms}
    report = agent.run(forms, evidence)
    assert report.field_accuracy == 1.0
    assert report.consistency == 1.0
    assert report.tool_calls == (len(forms) + 2) * len(KYC_FIELDS)


def test_memory_serves_a_customers_later_forms(platform, tmp_path: Path) -> None:
    forms = generate_forms(count=8, seed=11)
    agent = KycAgent(wired(platform, tmp_path), agent_role())
    evidence = {form.form_id: evidence_lines(form, random.Random(11)) for form in forms}
    agent.run(forms, evidence)
    assert agent.memory_hits == (len(forms) - 2) * len(KYC_FIELDS)


def test_a_role_without_write_may_not_fill(platform, tmp_path: Path) -> None:
    reader = RoleSpec(
        name="kyc-reader",
        mission="read only",
        sub_agents=(),
        inputs=(),
        outputs=(),
        capabilities=frozenset({Capability.TASK_READ}),
        gates=(),
        kpis=(),
    )
    agent = KycAgent(wired(platform, tmp_path), reader)
    form = generate_forms(count=4, seed=7)[0]
    with pytest.raises(ToolDenied):
        agent.fill(form, evidence_lines(form, random.Random(7)))


def test_training_text_and_trial_prompt_share_the_layout() -> None:
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7))
    text = kyc_fill_text(form.values, lines)
    assert all(value in text for value in form.values.values())
    prompt = kyc_prompt(lines, "phone")
    assert "phone" in prompt
    assert form.values["phone"] in prompt
    assert prompt != text


def test_the_prompt_can_name_the_person_the_form_is_for() -> None:
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7))
    anchored = kyc_prompt(lines, "phone", customer=form.values["full_name"])
    assert f"The person this form is for is {form.values['full_name']}." in anchored
    assert anchored.endswith("Question: What is phone?\nAnswer:")
    unanchored = kyc_prompt(lines, "phone")
    assert "The person this form is for" not in unanchored


def test_training_text_answers_before_it_retells() -> None:
    """The retelling used to sit before the answers and name every value,
    so the answers could be copied from it instead of from the papers."""
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7))
    text = kyc_fill_text(form.values, lines)
    assert all(value in text for value in form.values.values())
    assert text.rindex("Question:") < text.index("A clerk read those papers")


def test_every_question_after_the_name_carries_it() -> None:
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7))
    text = kyc_fill_text(form.values, lines)
    anchor = f"The person this form is for is {form.values['full_name']}."
    assert anchor in text
    assert text.index("Question: What is full_name?") < text.index(anchor)
    assert text.index(anchor) < text.index("Question: What is phone?")


def test_the_agent_anchors_each_field_on_the_name_it_filled(platform, tmp_path: Path) -> None:
    seen: list[str] = []
    form = generate_forms(count=4, seed=7)[0]

    def reader(prompt: str) -> str:
        seen.append(prompt)
        if "What is full_name?" in prompt:
            return f"account holder is {form.values['full_name']}"
        return ""

    agent = KycAgent(wired(platform, tmp_path), agent_role(), reader=reader)
    agent.fill(form, evidence_lines(form, random.Random(7)))
    assert seen, "the reader was never asked"
    first, rest = seen[0], seen[1:]
    assert "The person this form is for" not in first
    name = form.values["full_name"]
    assert rest, "only one field was asked"
    assert all(f"The person this form is for is {name}." in prompt for prompt in rest)


def test_the_field_order_is_not_the_same_for_every_record() -> None:
    """A fixed order lets a model answer by position instead of by question,
    and at trial time it is asked for one field at a time."""
    orders = set()
    for form in generate_forms(count=12, seed=7):
        lines = evidence_lines(form, random.Random(7))
        text = kyc_fill_text(form.values, lines)
        asked = [
            line.split("What is ")[1].split("?")[0]
            for line in text.splitlines()
            if line.startswith("Question: ")
        ]
        orders.add(tuple(asked))
    assert len(orders) > 1


def test_the_same_record_renders_the_same_text_twice() -> None:
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7))
    assert kyc_fill_text(form.values, lines) == kyc_fill_text(form.values, lines)


def test_an_agent_with_a_reader_asks_the_model_not_the_patterns(platform, tmp_path: Path) -> None:
    forms = generate_forms(count=8, seed=11)
    truth = {form.form_id: evidence_lines(form, random.Random(11)) for form in forms}
    agent = KycAgent(wired(platform, tmp_path), agent_role(), reader=lambda prompt: prompt)
    evidence = {form.form_id: truth[form.form_id] for form in forms}
    report = agent.run(forms, evidence)
    # This test is about the wiring, not about accuracy: the reader echoes the
    # prompt back, which is not a model answer, so most fields come back but
    # nothing here claims how well a real model would do.
    assert report.model_calls == 2 * len(KYC_FIELDS)
    assert report.field_accuracy >= 0.9


def test_every_document_wording_is_extractable() -> None:
    """A new phrasing without its pattern fails here, not in the trial score."""
    form = generate_forms(count=4, seed=7)[0]
    for family in DOC_FAMILIES:
        for template in family:
            line = template.format(**form.values)
            carried = [name for name in KYC_FIELDS if "{" + name + "}" in template]
            assert carried, template
            for name in carried:
                assert extract(name, [line]) == form.values[name], (template, name)


def test_ranking_prefers_the_cued_line() -> None:
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7))
    top = rank_lines("phone", lines, limit=1)
    assert form.values["phone"] in top[0]
    assert extract("id_number", rank_lines("id_number", lines)) == form.values["id_number"]
