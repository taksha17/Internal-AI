"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmforge.context import Forge
from llmforge.data.samples import generate_corpus
from llmforge.llm import ScriptedAdapter


@pytest.fixture()
def forge(tmp_path: Path) -> Forge:
    instance = Forge.create(root=tmp_path, llm_kind="scripted")
    instance.runtime.llm = ScriptedAdapter(nominal_cost_usd=0.01)
    yield instance
    instance.close()


@pytest.fixture()
def corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    generate_corpus(path, records=60, seed=3)
    return path
