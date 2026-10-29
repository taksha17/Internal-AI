"""The project brief is the record the acceptance checklist asks a company to keep.

It is checked here because it is the one page that decides whether a run has a
target to miss, a metric to judge it by, and a person to ask when it fails.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from conftest import enroll, session_csrf, sign_in
from fastapi.testclient import TestClient

from internal_ai import brief
from internal_ai.api import create_app
from internal_ai.data.records import Record
from internal_ai.domain import AuditOutcome
from internal_ai.evals.advice import retrieval_advice
from internal_ai.evals.harness import EvalHarness

FLASH = re.compile(r'class="flash flash-(?P<kind>ok|error)"[^>]*>(?P<body>[^<]*)')


def full_brief(**overrides: str) -> dict[str, str]:
    data = {
        "target": "Answer support questions from resolved tickets, with a citation.",
        "quality_bar": "Better recall than the keyword search agents use today.",
        "languages": "English",
        "context_length": "512",
        "metric_name_1": "recall@10",
        "metric_direction_1": "at_least",
        "metric_threshold_1": "0.70",
        "budget_usd": "5000",
        "timeline": "Pilot by the end of Q1",
        "deployment": "On-prem, support VLAN",
        "latency_ms": "800",
        "data_restrictions": "No customer records outside the EU region.",
        "non_goals": "Not a general assistant; no answer without a cited source.",
    }
    for role in brief.OWNER_ROLES:
        data[f"owner_{role}"] = "Lena Lead"
    data.update(overrides)
    return data


def flashes(page: str) -> list[tuple[str, str]]:
    return [(match.group("kind"), match.group("body")) for match in FLASH.finditer(page)]


def signed_in_as(app, name: str) -> tuple[TestClient, str]:
    client = TestClient(app)
    assert sign_in(client, name).status_code == 303
    return client, session_csrf(client)


def test_a_complete_brief_reaches_the_overview_and_the_plan(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        enroll(platform, "Lena Lead", ["program_lead"])

        client, csrf = signed_in_as(app, "Lena Lead")
        page = client.post("/console/brief", data={**full_brief(), "csrf_token": csrf})
        assert page.status_code == 200, page.text
        assert ("ok", "brief saved") in flashes(page.text)

        saved = brief.load(platform.store)
        assert saved.complete, saved.missing
        assert saved.metrics[0].threshold == "0.70"

        overview = client.get("/console/home").text
        assert "Project brief" in overview
        assert "Answer support questions from resolved tickets" in overview
        assert "recall@10" in overview and "≥ 0.70" in overview
        assert "Still to record" not in overview

        plan = client.get("/console/plan").text
        assert "has no complete project brief" not in plan

        event = next(
            item for item in platform.audit.recent(limit=50) if item.action == "brief.updated"
        )
        assert event.details["complete"] is True
        assert event.details["missing"] == []


def test_a_brief_without_owners_names_them_where_the_work_starts(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        enroll(platform, "Lena Lead", ["program_lead"])

        client, csrf = signed_in_as(app, "Lena Lead")
        data = full_brief(**{f"owner_{role}": "" for role in brief.OWNER_ROLES})
        page = client.post("/console/brief", data={**data, "csrf_token": csrf})
        unowned = "owners for " + ", ".join(brief.OWNER_ROLES)
        assert ("ok", f"brief saved; still missing {unowned}") in flashes(page.text)

        saved = brief.load(platform.store)
        assert saved.missing == [unowned]

        overview = client.get("/console/home").text
        assert "Still to record" in overview
        assert "owners for ml_lead" in overview

        plan = client.get("/console/plan").text
        assert "has no complete project brief" in plan
        assert "owners for ml_lead" in plan


def test_writing_the_brief_is_refused_to_a_role_that_does_not_hold_it(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        enroll(platform, "Cora Compliance", ["compliance"])

        client, csrf = signed_in_as(app, "Cora Compliance")
        page = client.post("/console/brief", data={**full_brief(), "csrf_token": csrf})
        assert page.status_code == 200
        assert any(kind == "error" for kind, _body in flashes(page.text))
        assert not brief.load(platform.store).target

        refusal = next(
            item for item in platform.audit.recent(limit=50) if item.action == "console.denied"
        )
        assert refusal.outcome == AuditOutcome.DENIED
        assert "write brief" in refusal.details["reason"]

        read_only = client.get("/console/brief").text
        assert "needs program_lead or ml_lead" in read_only


class HashEmbedder:
    """Deterministic stand-in, so the brief comparison is what is under test."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(8, dtype=float)
            for index, char in enumerate(text):
                vector[index % 8] += (ord(char) % 13) / 13
            vectors.append(vector / (np.linalg.norm(vector) or 1.0))
        return np.array(vectors)


def record(key: str, text: str) -> Record:
    return Record(
        id=key,
        record_key=key,
        text=text,
        source_id="src",
        acl=["engineering"],
        sensitivity="internal",
    )


def run_eval(platform, target: brief.Metric | None):
    harness = EvalHarness(platform.store, platform.audit)
    corpus = [record("a", "how do I reset my vpn on the laptop")]
    eval_set = harness.create_eval_set("brief", [{"query": "reset vpn", "relevant_ids": ["a"]}])
    project = brief.ProjectBrief()
    if target is not None:
        project.metrics = [target]
    return harness.evaluate_retrieval(
        HashEmbedder(),
        eval_set,
        corpus,
        training_records=[record("b", "quarterly invoice reconciliation runbook")],
        brief=project,
    )


def test_a_threshold_is_read_the_way_people_write_it() -> None:
    assert brief.parse_threshold("0.70") == 0.70
    assert brief.parse_threshold("≥ 0.7") == 0.7
    assert brief.parse_threshold("at least 0.85") == 0.85
    assert brief.parse_threshold("70%", 0.42) == 0.70
    assert brief.parse_threshold("0,65") == 0.65
    assert brief.parse_threshold("better than search") is None


def test_a_run_that_misses_the_brief_is_told_so(platform) -> None:
    # 1.01 is above any metric's ceiling, so the miss does not depend on how the
    # stand-in embedder happens to rank a one-document corpus.
    outcome = run_eval(
        platform, brief.Metric(name="recall@10", threshold="1.01", direction="at_least")
    )
    check = outcome.checks["brief_targets"]
    assert check["passed"] is False
    assert check["missed"] == ["recall@10"]
    assert check["targets"][0]["measured"] is not None
    assert outcome.passed is False
    assert "short of the brief" in outcome.notes
    assert "recall@10 at least 1.01" in outcome.notes

    advice = retrieval_advice(
        {"passed": False, "metrics": outcome.metrics, "checks": outcome.checks},
        records=1,
        tokens=10,
    )
    assert any("brief sets a bar this run did not reach" in line for line in advice)


def test_a_run_that_meets_the_brief_says_nothing_about_it(platform) -> None:
    outcome = run_eval(
        platform, brief.Metric(name="recall@10", threshold="0.0", direction="at_least")
    )
    check = outcome.checks["brief_targets"]
    assert check["passed"] is True
    assert check["note"] == "every metric the project brief records is met"
    assert "short of the brief" not in outcome.notes


def test_a_metric_the_evaluation_never_measured_cannot_pass_the_brief(platform) -> None:
    outcome = run_eval(
        platform, brief.Metric(name="accuracy", threshold="0.9", direction="at_least")
    )
    check = outcome.checks["brief_targets"]
    assert check["passed"] is False
    assert check["unmeasured"] == ["accuracy"]
    assert "does not measure accuracy" in outcome.notes
    assert outcome.passed is False


def test_no_brief_metric_means_no_verdict_at_all(platform) -> None:
    outcome = run_eval(platform, None)
    assert "brief_targets" not in outcome.checks


def test_a_classifier_is_judged_by_the_same_recorded_bar(platform, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from internal_ai.data.labels import build_labeled_dataset
    from internal_ai.data.pipeline import DataPipeline, PipelineConfig
    from internal_ai.data.records import Source
    from internal_ai.data.tokenizer import train_tokenizer
    from internal_ai.training.classifier import ClassifierConfig, ClassifierTrainer
    from internal_ai.training.model import TinyTransformer, TransformerConfig
    from internal_ai.workflows import evaluate_classifier_model

    texts = [
        "invoice reconciliation failed for the emea account today",
        "shipment tracking stalled for the amer warehouse queue",
    ] * 6
    labels = ["billing", "fulfilment"] * 6
    table = "ticket,description,label\n" + "".join(
        f"{index},{text},{label}\n"
        for index, (text, label) in enumerate(zip(texts, labels, strict=True))
    )
    path = tmp_path / "tickets.csv"
    path.write_text(table, encoding="utf-8")
    built = DataPipeline(platform.store, platform.audit, tmp_path).run(
        [
            Source(
                source_id="src_cls",
                path=path,
                acl=["engineering"],
                approved=True,
                owner="tester",
                text_field="description",
                label_field="label",
            )
        ],
        PipelineConfig(name="cls"),
        label_field="label",
    )
    tokenizer = train_tokenizer(texts, vocab_size=200)
    records = [
        Record(
            id=row["id"],
            record_key=row["record_key"],
            text=row["payload"],
            source_id=row["source_id"],
            acl=row["acl"],
            sensitivity=row["sensitivity"],
            meta=row["meta"],
        )
        for row in platform.store.dataset_records(built.dataset.id)
    ]
    labelled = build_labeled_dataset(records)
    assert {item["label"] for item in labelled} == {"billing", "fulfilment"}
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=64,
    )
    encoder_path = tmp_path / "encoder.pt"
    torch.save(
        {
            "model_state": TinyTransformer(config).state_dict(),
            "model_config": config.to_dict(),
        },
        encoder_path,
    )
    trained = ClassifierTrainer(
        platform.store, platform.audit, platform.budget, tmp_path / "artifacts", root=tmp_path
    ).train(
        ClassifierConfig(epochs=5),
        encoder_path,
        tokenizer,
        labelled,
        [item["label"] for item in labelled],
    )

    lead = enroll(platform, "Lena Lead", ["program_lead"])
    brief.save(
        platform.store,
        platform.audit,
        lead,
        brief.ProjectBrief(
            metrics=[brief.Metric(name="accuracy", threshold="1.01", direction="at_least")]
        ),
    )
    missed = evaluate_classifier_model(
        platform, Path(trained.model_path), tokenizer, built.dataset.id
    )
    assert missed["brief_targets"]["missed"] == ["accuracy"]
    assert "short of the brief" in missed["notes"]
    assert missed["passed"] is False

    brief.save(
        platform.store,
        platform.audit,
        lead,
        brief.ProjectBrief(
            metrics=[brief.Metric(name="accuracy", threshold="0.0", direction="at_least")]
        ),
    )
    met = evaluate_classifier_model(platform, Path(trained.model_path), tokenizer, built.dataset.id)
    assert met["brief_targets"]["passed"] is True


def test_the_model_card_shows_the_gap_for_each_recent_evaluation(tmp_path: Path) -> None:
    from internal_ai.domain import EvalReport, ModelVersion

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app):
        platform = app.state.platform
        lead = enroll(platform, "Lena Lead", ["program_lead"])
        brief.save(
            platform.store,
            platform.audit,
            lead,
            brief.ProjectBrief(
                metrics=[brief.Metric(name="recall@10", threshold="0.70", direction="at_least")]
            ),
        )
        platform.store.add_model(
            ModelVersion(
                id="mdl_progress",
                name="tickets-embed",
                version=1,
                content_hash="h",
                created_at="now",
            )
        )
        for report_id, stamp, recall in (
            ("rep_old", "2026-10-01T09:00:00+00:00", 0.412),
            ("rep_new", "2026-10-02T09:00:00+00:00", 0.655),
        ):
            platform.store.add_eval_report(
                EvalReport(
                    id=report_id,
                    model_id="mdl_progress",
                    eval_set_id="evs_tickets",
                    created_at=stamp,
                    metrics={"recall_at_10": recall},
                    passed=False,
                )
            )

        client, _csrf = signed_in_as(app, "Lena Lead")
        page = client.get("/console/models").text

        assert "how far each was from the bar it was judged by" in page
        assert "recall@10 0.412 of 0.70 — 0.288 short" in page
        assert "recall@10 0.655 of 0.70 — 0.045 short" in page
        assert page.index("0.412 of 0.70") < page.index("0.655 of 0.70")


def test_a_met_bar_reads_as_headroom_and_an_unmeasured_one_says_why() -> None:
    class Report:
        created_at = "2026-10-03T09:00:00+00:00"
        passed = True
        metrics: ClassVar[dict[str, float]] = {"recall_at_10": 0.81, "mrr": 0.5}

    project = brief.ProjectBrief(
        metrics=[
            brief.Metric(name="recall@10", threshold="0.70", direction="at_least"),
            brief.Metric(name="ndcg@10", threshold="0.60", direction="at_least"),
        ]
    )
    rows = brief.progress_rows(project, [Report()])

    assert rows[0]["met"] is False
    assert "recall@10 0.810 of 0.70 — 0.110 to spare" in rows[0]["text"]
    assert "this evaluation does not measure ndcg@10" in rows[0]["text"]


def test_an_edited_threshold_does_not_rewrite_how_short_a_past_run_was() -> None:
    class Report:
        created_at = "2026-10-05T09:00:00+00:00"
        passed = False
        metrics: ClassVar[dict[str, float]] = {"recall_at_10": 0.655}
        checks: ClassVar[dict[str, object]] = {
            "brief_targets": {
                "passed": False,
                "targets": [
                    {
                        "name": "recall@10",
                        "threshold": "0.70",
                        "direction": "at_least",
                        "measured": 0.655,
                        "gap": -0.045,
                        "met": False,
                        "note": "",
                    }
                ],
            }
        }

    project = brief.ProjectBrief(
        metrics=[brief.Metric(name="recall@10", threshold="0.90", direction="at_least")]
    )
    row = brief.progress_rows(project, [Report()])[0]

    assert row["basis"] == "recorded"
    assert "recall@10 0.655 of 0.70 — 0.045 short" in row["text"]
    assert "the brief now asks 0.90" in row["text"]
    assert "0.90 — 0.245 short" not in row["text"]


def test_a_report_from_before_the_bar_existed_is_read_against_today() -> None:
    class Report:
        created_at = "2026-10-05T09:00:00+00:00"
        passed = False
        metrics: ClassVar[dict[str, float]] = {"recall_at_10": 0.655}
        checks: ClassVar[dict[str, object]] = {}

    project = brief.ProjectBrief(
        metrics=[brief.Metric(name="recall@10", threshold="0.70", direction="at_least")]
    )
    row = brief.progress_rows(project, [Report()])[0]

    assert row["basis"] == "current"
    assert "recall@10 0.655 of 0.70 — 0.045 short" in row["text"]
