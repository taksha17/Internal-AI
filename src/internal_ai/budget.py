"""Budget enforcement, caps and kill switches (PRD section 15, G6, FR-13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from internal_ai.domain import BudgetLimit, BudgetScope, SpendRecord
from internal_ai.ids import new_id, now_iso
from internal_ai.store import Store

KILL_SWITCH_KEY = "kill_switch"


def week_start(reference: datetime | None = None) -> datetime:
    ref = reference or datetime.now(UTC)
    start = ref - timedelta(days=ref.weekday())
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass(slots=True)
class BudgetStatus:
    scope: str
    key: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    utilization: float
    exceeded: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "key": self.key,
            "limit_usd": round(self.limit_usd, 4),
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "utilization": round(self.utilization, 4),
            "exceeded": self.exceeded,
        }


@dataclass(slots=True)
class BudgetDecision:
    allowed: bool
    reason: str
    status: BudgetStatus | None = None


class BudgetManager:
    """Applies caps across global, project, agent, job and weekly scopes."""

    def __init__(self, store: Store, project: str = "default") -> None:
        self.store = store
        self.project = project

    def set_limit(
        self,
        scope: BudgetScope,
        key: str,
        limit_usd: float,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> BudgetLimit:
        start = period_start or now_iso()
        return self.store.set_budget_limit(
            BudgetLimit(
                scope=scope,
                key=key,
                limit_usd=limit_usd,
                period_start=start,
                period_end=period_end,
            )
        )

    def status(self, scope: BudgetScope, key: str) -> BudgetStatus | None:
        limit = next(
            (
                item
                for item in self.store.list_budget_limits()
                if item.scope == scope and item.key == key
            ),
            None,
        )
        if limit is None:
            return None
        spent = self.store.spend_by(scope, key)
        remaining = limit.limit_usd - spent
        utilization = spent / limit.limit_usd if limit.limit_usd else 0.0
        return BudgetStatus(
            scope=str(scope),
            key=key,
            limit_usd=limit.limit_usd,
            spent_usd=spent,
            remaining_usd=remaining,
            utilization=utilization,
            exceeded=spent > limit.limit_usd,
        )

    def all_status(self) -> list[BudgetStatus]:
        return [
            status
            for limit in self.store.list_budget_limits()
            if (status := self.status(BudgetScope(limit.scope), limit.key)) is not None
        ]

    def check(
        self, agent: str, task_id: str | None = None, job_id: str | None = None
    ) -> BudgetDecision:
        if self.kill_switch_engaged():
            return BudgetDecision(False, "global kill switch is engaged")
        scopes = [
            (BudgetScope.GLOBAL, "global"),
            (BudgetScope.PROJECT, self.project),
            (BudgetScope.AGENT, agent),
            (BudgetScope.WEEKLY, week_start().isoformat()),
        ]
        if job_id:
            scopes.append((BudgetScope.JOB, job_id))
        for scope, key in scopes:
            status = self.status(scope, key)
            if status is None:
                continue
            if status.exceeded:
                return BudgetDecision(
                    False, f"{scope}:{key} budget exhausted ({status.spent_usd:.2f} USD)", status
                )
        return BudgetDecision(True, "within budget")

    def record_spend(
        self,
        amount_usd: float,
        agent: str,
        task_id: str | None = None,
        job_id: str | None = None,
        notes: str = "",
    ) -> list[SpendRecord]:
        if amount_usd <= 0:
            return []
        stamp = now_iso()
        week = week_start().isoformat()
        entries = [
            SpendRecord(
                id=new_id("spd"),
                scope=BudgetScope.GLOBAL,
                key="global",
                amount_usd=amount_usd,
                agent=agent,
                task_id=task_id,
                job_id=job_id,
                created_at=stamp,
                notes=notes,
            ),
            SpendRecord(
                id=new_id("spd"),
                scope=BudgetScope.PROJECT,
                key=self.project,
                amount_usd=amount_usd,
                agent=agent,
                task_id=task_id,
                job_id=job_id,
                created_at=stamp,
                notes=notes,
            ),
            SpendRecord(
                id=new_id("spd"),
                scope=BudgetScope.AGENT,
                key=agent,
                amount_usd=amount_usd,
                agent=agent,
                task_id=task_id,
                job_id=job_id,
                created_at=stamp,
                notes=notes,
            ),
            SpendRecord(
                id=new_id("spd"),
                scope=BudgetScope.WEEKLY,
                key=week,
                amount_usd=amount_usd,
                agent=agent,
                task_id=task_id,
                job_id=job_id,
                created_at=stamp,
                notes=notes,
            ),
        ]
        if job_id:
            entries.append(
                SpendRecord(
                    id=new_id("spd"),
                    scope=BudgetScope.JOB,
                    key=job_id,
                    amount_usd=amount_usd,
                    agent=agent,
                    task_id=task_id,
                    job_id=job_id,
                    created_at=stamp,
                    notes=notes,
                )
            )
        for entry in entries:
            self.store.record_spend(entry)
        return entries

    def kill_switch_engaged(self) -> bool:
        return self.store.kv_get(KILL_SWITCH_KEY, "0") == "1"

    def engage_kill_switch(self) -> None:
        self.store.kv_set(KILL_SWITCH_KEY, "1")

    def release_kill_switch(self) -> None:
        self.store.kv_set(KILL_SWITCH_KEY, "0")
