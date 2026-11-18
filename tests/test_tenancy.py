"""Tenancy is a boundary, not a grant, and each tenant has its own room.

The rules under test: a record carries the tenant of the source it was read
from, a row cannot claim another one, retrieval never ranks another tenant's
rows whatever class or group the caller holds (the superuser included), a
dataset of another tenant is refused rather than answered as empty, and a
tenant's quota is spent by that tenant alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import csrf_from, enroll, sign_in
from fastapi.testclient import TestClient

from internal_ai.budget import BudgetManager
from internal_ai.data.pipeline import DataPipeline, PipelineConfig
from internal_ai.data.records import Source, ingest_source
from internal_ai.data.shards import read_shard, write_shard
from internal_ai.domain import BudgetScope, ModelVersion
from internal_ai.ids import now_iso
from internal_ai.services.attributes import ReadPolicy
from internal_ai.services.governance import set_budget_limit
from internal_ai.services.permissions import PermissionDenied
from internal_ai.services.tenancy import DEFAULT_TENANT, TenantError, check, normalise
from internal_ai.serving.app import create_serving_app
from internal_ai.serving.service import (
    Principal,
    RetrievalService,
    TenantQuota,
    parse_api_keys,
    parse_tenant_quotas,
)

pytestmark = pytest.mark.filterwarnings("ignore:Using `httpx` with `starlette.testclient`")

TEXT = "the ledger feed failed overnight and the runbook says to restart it"


def source(tmp_path: Path, tenant: str, row_tenant: str = "") -> Source:
    path = tmp_path / f"corpus-{tenant}.jsonl"
    row: dict[str, Any] = {"id": "doc-1", "text": TEXT}
    if row_tenant:
        row["tenant"] = row_tenant
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return Source(
        source_id=f"src_{tenant}",
        path=path,
        sensitivity="internal",
        acl=["engineering"],
        approved=True,
        tenant=tenant,
    )


def build(platform: Any, tmp_path: Path, tenant: str = "acme", row_tenant: str = "") -> str:
    pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root, platform.blobs)
    result = pipeline.run(
        [source(tmp_path, tenant, row_tenant)],
        PipelineConfig(name="handbook", version=1),
        actor="data_engineer",
    )
    return result.dataset.id


def test_a_record_carries_the_tenant_of_its_source_not_the_row(tmp_path, platform) -> None:
    records = ingest_source(source(tmp_path, "acme", row_tenant="globex"))
    assert [record.tenant for record in records] == ["acme"]
    rows = [record.to_row() for record in records]
    assert [row["tenant"] for row in rows] == ["acme"]


def test_a_dataset_and_its_rows_hold_the_tenant(tmp_path, platform) -> None:
    dataset_id = build(platform, tmp_path)
    dataset = platform.store.get_dataset(dataset_id)
    assert dataset is not None and dataset.tenant == "acme"
    assert {row["tenant"] for row in platform.store.dataset_records(dataset_id)} == {"acme"}


def test_the_shard_carries_the_tenant(tmp_path, platform) -> None:
    dataset_id = build(platform, tmp_path)
    rows = platform.store.dataset_records(dataset_id)
    shard = write_shard(platform.blobs, rows, dataset_id=dataset_id)
    assert [row["tenant"] for row in read_shard(platform.blobs, shard.key)] == ["acme"]


def test_retrieval_never_ranks_another_tenants_rows(tmp_path, platform) -> None:
    dataset_id = build(platform, tmp_path)
    retrieval = RetrievalService(platform.store, platform.audit, platform.blobs)
    same = Principal(user_id="p_acme", groups=frozenset({"engineering"}), tenant="acme")
    other = Principal(user_id="p_globex", groups=frozenset({"engineering"}), tenant="globex")
    assert [hit.document_id for hit in retrieval.keyword_search(dataset_id, "ledger", same)] == [
        "doc-1"
    ]
    assert retrieval.keyword_search(dataset_id, "ledger", other) == []
    # The refusal is recorded with the tenant, so a reviewer can see which boundary
    # a search ran against rather than guessing from an empty result.
    events = platform.store.list_audit(action="serving.search")
    assert events[0].details["tenant"] == "globex"
    assert events[0].tenant == "globex"
    assert events[0].details["visible_documents"] == 0
    assert events[0].details["total_documents"] == 1


def test_a_grant_and_the_superuser_group_do_not_cross_a_tenant(tmp_path, platform) -> None:
    dataset_id = build(platform, tmp_path)
    row = platform.store.dataset_records(dataset_id)[0]
    permissive = ReadPolicy.parse("internal=anyone")
    admin = Principal(
        user_id="p_admin",
        groups=frozenset({"platform-admins"}),
        tenant="globex",
        read_policy=permissive,
    )
    assert not admin.can_read_row(row)
    own = Principal(
        user_id="p_admin",
        groups=frozenset({"platform-admins"}),
        tenant="acme",
        read_policy=permissive,
    )
    assert own.can_read_row(row)
    # Nothing at all is readable across the boundary, whatever the class says.
    strict = ReadPolicy.strict()
    engineering = frozenset({"engineering"})
    assert not Principal(user_id="p_x", groups=engineering, tenant="globex").can_read_row(row)
    assert Principal(
        user_id="p_x", groups=engineering, tenant="acme", read_policy=strict
    ).can_read_row(row)


def test_a_dataset_of_another_tenant_is_refused_not_answered_empty(tmp_path) -> None:
    from internal_ai.context import Platform

    platform = Platform.create(root=tmp_path, llm_kind="scripted")
    try:
        dataset_id = build(platform, tmp_path)
        platform.store.add_model(
            ModelVersion(
                id="mdl_tenant",
                name="embedding-v1",
                version=1,
                content_hash="h",
                created_at=now_iso(),
                dataset_ids=[dataset_id],
                status="promoted",
            )
        )
    finally:
        platform.close()
    keys = {
        "k-globex": Principal(
            user_id="p_globex", groups=frozenset({"engineering"}), tenant="globex"
        )
    }
    app = create_serving_app(root=tmp_path, principals=keys, tenant_quotas={})
    with TestClient(app) as client:
        refused = client.post(
            "/v1/search", json={"query": "ledger"}, headers={"X-Internal-Key": "k-globex"}
        )
        assert refused.status_code == 403, refused.text
        detail = refused.json()["detail"]
        assert detail["tenant"] == "acme"
        assert "belongs to tenant acme" in detail["error"]
        assert "boundary, not a grant" in detail["error"]


def test_a_tenant_quota_caps_one_tenant_and_leaves_another_alone() -> None:
    quotas = TenantQuota({"acme": 2})
    assert quotas.limit("ACME") == 2
    assert [quotas.allow("acme", now=1.0) for _ in range(3)] == [True, True, False]
    assert quotas.remaining("acme", now=1.0) == 0
    assert quotas.allow("globex", now=1.0) is True
    assert quotas.limit("globex") is None
    assert quotas.remaining("globex") is None


def test_a_malformed_quota_list_refuses_instead_of_uncapping() -> None:
    assert parse_tenant_quotas("acme:30,globex:10") == {"acme": 30, "globex": 10}
    assert parse_tenant_quotas("") == {}
    with pytest.raises(ValueError, match="tenant:requests-per-minute"):
        parse_tenant_quotas("acme")
    with pytest.raises(ValueError, match="the floor is 1"):
        parse_tenant_quotas("acme:0")
    with pytest.raises(TenantError):
        parse_tenant_quotas("Acme Inc:5")


def test_a_serving_key_names_its_tenant_and_defaults_to_this_install() -> None:
    keys = parse_api_keys("k1:u1:engineering|support:acme,k2:u2:engineering")
    assert keys["k1"].tenant == "acme"
    assert keys["k1"].groups == frozenset({"engineering", "support"})
    assert keys["k2"].tenant == DEFAULT_TENANT
    payload = json.dumps({"k3": {"user_id": "u3", "groups": [], "tenant": "Globex"}})
    assert parse_api_keys(payload)["k3"].tenant == "globex"


def test_a_tenant_cap_is_spent_by_that_tenant_only(platform) -> None:
    acme = BudgetManager(platform.store, tenant="acme")
    globex = BudgetManager(platform.store, tenant="globex")
    acme.set_limit(BudgetScope.TENANT, "acme", 1.0)
    assert acme.reserve(0.6, "ops", job_id="job-a").allowed
    refused = acme.reserve(0.6, "ops")
    assert not refused.allowed
    assert "tenant:acme" in refused.reason
    # The other tenant holds no cap, so its own room is untouched by acme's spend.
    assert globex.reserve(0.6, "ops").allowed
    assert {item.key for item in acme.all_status()} == {"acme"}


def test_the_wrong_role_cannot_set_a_tenant_cap(platform, enroll_person) -> None:
    expert = enroll_person(platform, "Eve Expert", ["domain_expert"])
    with pytest.raises(PermissionDenied):
        set_budget_limit(platform, expert, BudgetScope.TENANT, "acme", 10.0)
    assert platform.budget.status(BudgetScope.TENANT, "acme") is None
    finance = enroll_person(platform, "Fay Finance", ["finance"])
    result = set_budget_limit(platform, finance, BudgetScope.TENANT, "acme", 10.0)
    assert result["limit_usd"] == 10.0


def test_an_unusable_tenant_name_is_refused() -> None:
    assert check("") == DEFAULT_TENANT
    assert normalise(" Acme ") == "acme"
    with pytest.raises(TenantError, match="not a tenant name"):
        check("Acme Inc")


def test_the_console_shows_the_tenant_and_refuses_a_stranger_to_cap_it(tmp_path: Path) -> None:
    from internal_ai.api import create_app

    app = create_app(root=tmp_path, llm_kind="scripted", sync_jobs=True)
    with TestClient(app) as client:
        platform = app.state.platform
        platform.store.conn.execute(
            "INSERT INTO data_sources (id, name, path, kind, acl, sensitivity, scan, purpose,"
            " requested_by, created_at, tenant) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "src_acme",
                "acme-handbook",
                "/tmp/acme",
                "file",
                '["engineering"]',
                "internal",
                "{}",
                "search",
                "ppl_x",
                now_iso(),
                "acme",
            ),
        )
        expert = enroll(platform, "Eve Expert", ["domain_expert"])
        lead = enroll(platform, "Pat Lead", ["program_lead"])
        sign_in(client, expert.name)
        page = client.get("/console/sources")
        assert "acme-handbook" in page.text
        assert "acme" in page.text

        csrf = csrf_from(client.get("/console/home").text)
        refused = client.post(
            "/console/admin/budget",
            data={
                "csrf_token": csrf,
                "scope": "tenant",
                "key": "acme",
                "limit_usd": "5",
            },
        )
        assert "set budget needs finance or program_lead" in refused.text
        assert platform.budget.status(BudgetScope.TENANT, "acme") is None

        other = TestClient(app)
        sign_in(other, lead.name)
        budget_page = other.get("/console/admin/budget")
        assert 'value="tenant"' in budget_page.text
