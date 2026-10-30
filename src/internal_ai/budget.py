"""Budget enforcement, caps and kill switches (PRD section 15, G6, FR-13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from internal_ai.db import transaction
from internal_ai.domain import BudgetLimit, BudgetScope, SpendRecord
from internal_ai.ids import new_id, now_iso
from internal_ai.store import Store

KILL_SWITCH_KEY = "kill_switch"

# A cap is compared against sums of floats, so spend that lands exactly on the
# limit must not be refused because of the last bit of the mantissa.
_TOLERANCE = 1e-9

# A hold that is never settled, because the caller died, returns its room after
# this. It is much longer than any call the platform makes.
RESERVATION_TTL_SECONDS = 900


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
    reservation_id: str | None = None


class BudgetManager:
    """Applies caps across global, project, agent, job and weekly scopes.

    A cap covers a call that has been paid for and a call that is in flight.
    ``reserve`` takes the hold inside one write transaction, so two callers
    racing for the last of a budget cannot both be told yes; ``settle`` turns
    the hold into spend and ``release`` gives it back when the call never ran.
    """

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
        limit = self._limit(scope, key)
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
        over = self._over_cap(0.0, agent, job_id)
        if over is None:
            return BudgetDecision(True, "within budget")
        scope, key, status = over
        spent = self._committed(scope, key)
        return BudgetDecision(False, f"{scope}:{key} budget exhausted ({spent:.2f} USD)", status)

    def reserve(
        self,
        amount_usd: float,
        agent: str,
        task_id: str | None = None,
        job_id: str | None = None,
        ttl_seconds: int = RESERVATION_TTL_SECONDS,
    ) -> BudgetDecision:
        """Hold room for one call before its cost is known.

        The check and the hold happen in a single `BEGIN IMMEDIATE`
        transaction, which is what stops a burst of callers from all passing
        the same check and then spending past the cap together.
        """
        if amount_usd <= 0:
            return BudgetDecision(True, "nothing to reserve")
        if self.kill_switch_engaged():
            return BudgetDecision(False, "global kill switch is engaged")
        stamp = now_iso()
        expires_at = (datetime.now(UTC) + timedelta(seconds=ttl_seconds)).isoformat()
        with transaction(self.store.conn):
            self.store.expire_spend_reservations(stamp)
            over = self._over_cap(amount_usd, agent, job_id)
            if over is not None:
                scope, key, status = over
                committed = self._committed(scope, key)
                return BudgetDecision(
                    False,
                    f"{scope}:{key} has {status.limit_usd - committed:.2f} USD room "
                    f"and {amount_usd:.2f} USD was requested",
                    status,
                )
            group_id = new_id("rsv")
            for scope, key in self._scopes(agent, job_id):
                self.store.add_spend_reservation(
                    reservation_id=new_id("rsv"),
                    group_id=group_id,
                    scope=scope,
                    key=key,
                    amount_usd=amount_usd,
                    agent=agent,
                    task_id=task_id,
                    job_id=job_id,
                    created_at=stamp,
                    expires_at=expires_at,
                )
        return BudgetDecision(True, f"reserved {amount_usd:.4f} USD", reservation_id=group_id)

    def settle(self, reservation_id: str, notes: str = "") -> list[SpendRecord]:
        """Turn a hold into spend. A hold can only be settled once."""
        with transaction(self.store.conn):
            rows = self.store.reservation_rows(reservation_id)
            if not rows:
                return []
            stamp = now_iso()
            entries = [
                SpendRecord(
                    id=new_id("spd"),
                    scope=BudgetScope(row["scope"]),
                    key=row["key"],
                    amount_usd=float(row["amount_usd"]),
                    agent=row["agent"],
                    task_id=row["task_id"],
                    job_id=row["job_id"],
                    created_at=stamp,
                    notes=notes,
                )
                for row in rows
            ]
            for entry in entries:
                self.store.record_spend(entry)
            self.store.delete_spend_reservations(reservation_id)
        return entries

    def release(self, reservation_id: str) -> None:
        """Give a hold back when the call it was taken for never ran."""
        self.store.delete_spend_reservations(reservation_id)

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
        entries = [
            SpendRecord(
                id=new_id("spd"),
                scope=scope,
                key=key,
                amount_usd=amount_usd,
                agent=agent,
                task_id=task_id,
                job_id=job_id,
                created_at=stamp,
                notes=notes,
            )
            for scope, key in self._scopes(agent, job_id)
        ]
        for entry in entries:
            self.store.record_spend(entry)
        return entries

    def kill_switch_engaged(self) -> bool:
        return self.store.kv_get(KILL_SWITCH_KEY, "0") == "1"

    def engage_kill_switch(self) -> None:
        self.store.kv_set(KILL_SWITCH_KEY, "1")

    def release_kill_switch(self) -> None:
        self.store.kv_set(KILL_SWITCH_KEY, "0")

    def _scopes(self, agent: str, job_id: str | None) -> list[tuple[BudgetScope, str]]:
        scopes = [
            (BudgetScope.GLOBAL, "global"),
            (BudgetScope.PROJECT, self.project),
            (BudgetScope.AGENT, agent),
            (BudgetScope.WEEKLY, week_start().isoformat()),
        ]
        if job_id:
            scopes.append((BudgetScope.JOB, job_id))
        return scopes

    def _limit(self, scope: BudgetScope, key: str) -> BudgetLimit | None:
        return next(
            (
                item
                for item in self.store.list_budget_limits()
                if item.scope == scope and item.key == key
            ),
            None,
        )

    def _committed(self, scope: BudgetScope, key: str) -> float:
        spent = self.store.spend_by(scope, key)
        held = self.store.reserved_spend(scope, key, now_iso())
        return spent + held

    def _over_cap(
        self, amount_usd: float, agent: str, job_id: str | None
    ) -> tuple[BudgetScope, str, BudgetStatus] | None:
        for scope, key in self._scopes(agent, job_id):
            limit = self._limit(scope, key)
            if limit is None:
                continue
            if self._committed(scope, key) + amount_usd > limit.limit_usd + _TOLERANCE:
                status = self.status(scope, key)
                if status is not None:
                    return scope, key, status
        return None
