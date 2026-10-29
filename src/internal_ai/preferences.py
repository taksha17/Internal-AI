"""Which of two answers a person prefers. Not training data until a release.

A preference is an opinion a person recorded, not a model's judgement. It is
exported unreleased like feedback, and a later preference update waits for the
release gate and an attached backend; this module does not run one.
"""

from __future__ import annotations

from typing import Any

from internal_ai.compute.advise import BACKEND_KEY
from internal_ai.context import Platform
from internal_ai.data.pipeline import _git_commit
from internal_ai.domain import AuditOutcome, DatasetVersion
from internal_ai.ids import content_hash, new_id, now_iso
from internal_ai.services.actor import Actor

PREFERENCE_ROLE = "domain_expert"


def file_preference(
    platform: Platform,
    actor: Actor,
    question: str,
    answer_a: str,
    answer_b: str,
    preferred: str,
    dataset_id: str = "",
    authored_by: str = "",
    source_id: str = "",
) -> dict[str, Any]:
    """Store two answers and the better one, as a person holding domain_expert."""
    if not actor.holds(PREFERENCE_ROLE):
        platform.audit.record(
            actor=actor.id,
            actor_kind=actor.kind,
            action="preference.rejected",
            subject=dataset_id,
            outcome=AuditOutcome.DENIED,
            details={"reason": f"filing a preference requires {PREFERENCE_ROLE}"},
        )
        raise PermissionError(f"filing a preference requires {PREFERENCE_ROLE}")
    author = authored_by.strip() or actor.id
    if author == "model" or source_id.strip().startswith("alpha"):
        platform.audit.record(
            actor=actor.id,
            actor_kind=actor.kind,
            action="preference.rejected",
            subject=source_id,
            outcome=AuditOutcome.DENIED,
            details={"authored_by": author, "source_id": source_id},
        )
        raise ValueError("a model's answer is not a preference a person filed")
    if preferred not in {"a", "b"}:
        raise ValueError("preferred is a or b")
    if not (question.strip() and answer_a.strip() and answer_b.strip()):
        raise ValueError("a preference needs a question and both answers")
    if answer_a.strip() == answer_b.strip():
        raise ValueError("the two answers are the same, so there is nothing to prefer")
    row = {
        "id": new_id("prf"),
        "question": question.strip(),
        "answer_a": answer_a.strip(),
        "answer_b": answer_b.strip(),
        "preferred": preferred,
        "authored_by": author,
        "dataset_id": dataset_id,
        "filed_at": now_iso(),
    }
    platform.store.conn.execute(
        "INSERT INTO preference (id, question, answer_a, answer_b, preferred, authored_by,"
        " dataset_id, filed_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            row["id"],
            row["question"],
            row["answer_a"],
            row["answer_b"],
            row["preferred"],
            row["authored_by"],
            row["dataset_id"],
            row["filed_at"],
        ),
    )
    platform.audit.record(
        actor=actor.id,
        actor_kind=actor.kind,
        action="preference.filed",
        subject=row["id"],
        details={"dataset_id": dataset_id, "preferred": preferred, "authored_by": author},
    )
    return row


def list_preferences(platform: Platform) -> list[dict[str, Any]]:
    rows = platform.store.conn.execute("SELECT * FROM preference ORDER BY filed_at").fetchall()
    return [dict(row) for row in rows]


def export_preferences(platform: Platform, actor: Actor) -> DatasetVersion:
    """A new unreleased dataset. Release still needs its gate; nothing trains here."""
    if not actor.holds(PREFERENCE_ROLE):
        raise PermissionError(f"exporting preferences requires {PREFERENCE_ROLE}")
    rows = list_preferences(platform)
    if not rows:
        raise ValueError("no preferences have been filed")
    records = []
    for row in rows:
        chosen = row["answer_a"] if row["preferred"] == "a" else row["answer_b"]
        rejected = row["answer_b"] if row["preferred"] == "a" else row["answer_a"]
        records.append(
            {
                "record_key": row["id"],
                "payload": f"{row['question']}\nchosen: {chosen}\nrejected: {rejected}",
                "source_id": "preferences",
                "acl": ["engineering"],
                "meta": {"kind": "preference", "preference_id": row["id"]},
            }
        )
    existing = platform.store.list_datasets(name="preferences")
    version = max((item.version for item in existing), default=0) + 1
    dataset_id = new_id("ds")
    content = content_hash({"rows": [item["record_key"] for item in records]})
    dataset = DatasetVersion(
        id=dataset_id,
        name="preferences",
        version=version,
        content_hash=content,
        record_count=len(records),
        token_count=sum(max(1, len(item["payload"]) // 4) for item in records),
        created_at=now_iso(),
        lineage={
            "kind": "preference",
            "sources": [
                {
                    "source_id": "preferences",
                    "path": "preferences",
                    "content_hash": content,
                    "owner": actor.id,
                    "approved": True,
                }
            ],
            "config_hash": content_hash({"kind": "preference", "rows": len(records)}),
            "code_commit": _git_commit(platform.settings.root),
            "preference_ids": [row["id"] for row in rows],
        },
        stats={"kind": "preference", "training_data": False, "token_count_kind": "measured"},
        released=False,
    )
    platform.store.add_dataset(dataset)
    platform.store.add_dataset_records(dataset_id, records)
    platform.audit.record(
        actor=actor.id,
        actor_kind="human",
        action="preference.exported",
        subject=dataset_id,
        details={"rows": len(records), "released": False, "training_data": False},
    )
    return platform.store.get_dataset(dataset_id) or dataset


def train_preference(platform: Platform, actor: Actor, dataset_id: str) -> dict[str, Any]:
    """Refuse a preference update until its set is released and a backend is attached."""
    dataset = platform.store.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if str(dataset.stats.get("kind") or dataset.lineage.get("kind")) != "preference":
        raise ValueError("a preference update trains on an exported preference set")
    if not dataset.released:
        raise PermissionError("release the preference set before training on it")
    if dataset.stats.get("training_data") is not True:
        raise PermissionError("this set is not marked training data")
    if not platform.store.kv_get(BACKEND_KEY, ""):
        raise PermissionError("attach a compute backend before a preference update")
    raise NotImplementedError(
        "a preference update is not implemented; this command only checks the set and the backend"
    )
