"""A cap holds when callers race for the same money (change 0.2)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from internal_ai.domain import BudgetScope

CAP = 0.60
EACH = 0.05


def test_a_burst_cannot_spend_past_the_cap(platform) -> None:
    platform.budget.set_limit(BudgetScope.GLOBAL, "global", CAP)

    def run(_: int) -> bool:
        decision = platform.budget.reserve(EACH, agent="data_engineer")
        if not decision.allowed:
            return False
        assert decision.reservation_id is not None
        platform.budget.settle(decision.reservation_id, notes="burst")
        return True

    with ThreadPoolExecutor(max_workers=24) as pool:
        allowed = list(pool.map(run, range(24)))

    spent = platform.store.spend_by(BudgetScope.GLOBAL, "global")
    assert sum(allowed) == round(CAP / EACH)
    assert spent <= CAP + 1e-9


def test_a_released_hold_returns_its_room(platform) -> None:
    platform.budget.set_limit(BudgetScope.AGENT, "data_engineer", 1.0)
    held = platform.budget.reserve(0.80, agent="data_engineer")
    assert held.allowed and held.reservation_id is not None
    assert not platform.budget.reserve(0.40, agent="data_engineer").allowed

    platform.budget.release(held.reservation_id)

    assert platform.budget.reserve(0.40, agent="data_engineer").allowed
    assert platform.budget.check("data_engineer").allowed


def test_a_hold_settles_once_and_becomes_spend(platform) -> None:
    platform.budget.set_limit(BudgetScope.AGENT, "data_engineer", 1.0)
    held = platform.budget.reserve(0.30, agent="data_engineer")
    assert held.reservation_id is not None

    assert len(platform.budget.settle(held.reservation_id)) == 4

    assert platform.budget.settle(held.reservation_id) == []
    assert platform.store.spend_by(BudgetScope.AGENT, "data_engineer") == 0.30


def test_an_expired_hold_stops_counting(platform) -> None:
    platform.budget.set_limit(BudgetScope.AGENT, "data_engineer", 1.0)
    held = platform.budget.reserve(0.90, agent="data_engineer", ttl_seconds=-1)
    assert held.allowed
    assert platform.budget.reserve(0.90, agent="data_engineer").allowed
