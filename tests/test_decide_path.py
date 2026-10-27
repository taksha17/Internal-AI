"""The decide path: the classifier learns the answers the data already carries.

A row read from a table with an answer column keeps that answer. Where there is
no answer column the shipped keyword rules stay the fallback and the baseline,
so nothing that existed before changes meaning.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.api import create_app
from internal_ai.data.labels import build_labeled_dataset
from internal_ai.data.pipeline import DataPipeline, PipelineConfig
from internal_ai.data.records import Record, Source, choose_columns, ingest_source
from internal_ai.ids import content_hash
from internal_ai.services import build
from internal_ai.services.permissions import PermissionDenied

VPN = "my vpn will not connect to the office network from home again today"
INVOICE = "the invoice total is wrong and needs a credit note raised today"

TABLE = f"ticket,description,label\n1,{VPN},network\n2,{INVOICE},billing\n"


def on_disk(tmp_path: Path, name: str, body: str) -> Path:
    """A data file on disk, as an export would arrive."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_the_columns_are_chosen_by_name_then_by_how_much_they_hold() -> None:
    assert choose_columns(".csv", TABLE) == ("description", "label")
    assert choose_columns(".jsonl", '{"body": "hello there", "status": "open"}\n') == (
        "body",
        "status",
    )


def test_a_row_keeps_the_answer_its_table_carried(tmp_path: Path) -> None:
    source = Source(
        source_id="src_one",
        path=on_disk(tmp_path, "tickets.csv", TABLE),
        acl=["engineering"],
        approved=True,
        owner="tester",
        text_field="description",
        label_field="label",
    )

    records = ingest_source(source)

    assert [record.meta.get("label") for record in records] == ["network", "billing"]
    assert [record.text for record in records] == [VPN, INVOICE]


def test_a_row_without_an_answer_column_carries_no_label(tmp_path: Path) -> None:
    source = Source(
        source_id="src_two",
        path=on_disk(
            tmp_path, "tickets.csv", "ticket,description\n1,my vpn will not connect at all\n"
        ),
        acl=["engineering"],
        approved=True,
        owner="tester",
        text_field="description",
    )

    records = ingest_source(source)

    assert records and all("label" not in record.meta for record in records)


def test_labelling_prefers_the_data_over_the_shipped_rules() -> None:
    records = [
        Record(
            id="r1",
            record_key="1",
            text="shipment tracking was delayed for account acct-9",
            source_id="src",
            meta={"label": "logistics"},
        ),
        Record(
            id="r2",
            record_key="2",
            text="shipment tracking was delayed for account acct-8",
            source_id="src",
        ),
    ]

    labelled = build_labeled_dataset(records)

    assert labelled[0]["label"] == "logistics"
    assert labelled[1]["label"] == "fulfilment"


def test_a_labelled_build_records_where_the_answers_came_from(platform, tmp_path: Path) -> None:
    pipeline = DataPipeline(platform.store, platform.audit, tmp_path)
    source = Source(
        source_id="src_labelled",
        path=on_disk(tmp_path, "tickets.csv", TABLE),
        acl=["engineering"],
        approved=True,
        owner="tester",
        text_field="description",
        label_field="label",
    )

    result = pipeline.run([source], PipelineConfig(name="decide-test"), label_field="label")

    assert result.dataset.lineage["labels"] == {
        "column": "label",
        "count": 2,
        "distribution": {"billing": 1, "network": 1},
    }
    assert sorted(record.meta["label"] for record in result.records) == ["billing", "network"]
    assert build.label_summary(platform, result.dataset.id) == {
        "rows": 2,
        "from_data": 2,
        "from_rules": 0,
    }


def test_an_unlabelled_build_keeps_the_hash_it_always_had(platform, tmp_path: Path) -> None:
    rows = (
        "the vpn client keeps dropping the tunnel on the home network again\n\n"
        "the invoice total is wrong and needs a credit note raised today please\n"
    )
    config = PipelineConfig(name="no-labels")
    pipeline = DataPipeline(platform.store, platform.audit, tmp_path)
    source = Source(
        source_id="src_plain",
        path=on_disk(tmp_path, "notes.txt", rows),
        acl=["engineering"],
        approved=True,
        owner="tester",
    )

    result = pipeline.run([source], config)

    assert "labels" not in result.dataset.lineage
    assert result.dataset.content_hash == content_hash(
        {
            "records": sorted(record.fingerprint() for record in result.records),
            "config": config.hash,
        }
    )


def test_training_a_classifier_is_denied_to_a_role_that_must_not_have_it(platform) -> None:
    viewer = enroll(platform, "Vera Viewer", ["program_lead"])

    with pytest.raises(PermissionDenied):
        build.train_classifier(platform, viewer, None, "ds_any", "mdl_any")


def test_training_a_classifier_needs_a_dataset_that_exists(platform) -> None:
    builder = enroll(platform, "Mo Builder", ["ml_lead"])

    with pytest.raises(KeyError):
        build.train_classifier(platform, builder, None, "ds_missing", "mdl_missing")


def test_the_classifier_form_is_hidden_from_roles_without_train_model(tmp_path: Path) -> None:
    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        lead = enroll(app.state.platform, "Lena Lead", ["program_lead"])
        assert sign_in(client, lead.name).status_code == 303
        assert "Train a classifier" not in client.get("/console/models").text

        build_client = TestClient(app)
        builder = enroll(app.state.platform, "Mo Builder", ["ml_lead"])
        assert sign_in(build_client, builder.name).status_code == 303
        assert "Train a classifier" in build_client.get("/console/models").text
