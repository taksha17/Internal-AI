"""The KYC agent fills fake forms from evidence, through audited tools."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from internal_ai.data.kyc import (
    DEFAULT_FORMS,
    KYC_FIELDS,
    build_corpus,
    evidence_lines,
    generate_forms,
)
from internal_ai.domain import Capability, RoleSpec
from internal_ai.services.kyc_agent import KycAgent, agent_role, extract, rank_lines
from internal_ai.tools import ToolDenied, ToolRegistry


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


def test_ranking_prefers_the_cued_line() -> None:
    form = generate_forms(count=4, seed=7)[0]
    lines = evidence_lines(form, random.Random(7))
    top = rank_lines("phone", lines, limit=1)
    assert form.values["phone"] in top[0]
    assert extract("id_number", rank_lines("id_number", lines)) == form.values["id_number"]
