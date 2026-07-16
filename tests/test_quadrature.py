"""Public contract tests for deterministic event-aligned direct integration."""

from __future__ import annotations

import pytest
from finoptions import (
    CashDividend,
    ConstantVolatility,
    CpuExecutor,
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EventAlignedQuadrature,
    EventSchedule,
    InputError,
    LogDiscountCurve,
    price_event_aligned_call,
)


def _one_dividend_schedule() -> EventSchedule:
    return EventSchedule(
        times=(0.0, 0.5, 1.0),
        dividends=(CashDividend(node=1, amount=7.0),),
        exercise_nodes=(1,),
        expiry_node=2,
    )


def _price(schedule: EventSchedule, method: EventAlignedQuadrature) -> float:
    return price_event_aligned_call(
        initial_spot=100.0,
        strike=100.0,
        schedule=schedule,
        curve=LogDiscountCurve(times=schedule.times, log_discounts=(0.0, -0.03, -0.06)),
        dividend_yield_curve=CumulativeDividendYieldCurve.zero(times=schedule.times),
        volatility=ConstantVolatility(annualized_decimal=0.30),
        dividend_policy=DividendPolicy.LIQUIDATOR,
        executor=CpuExecutor(threads=4),
        method=method,
    ).price


def test_published_one_dividend_call_uses_the_exact_public_event_contract() -> None:
    """Match the published liquidator benchmark under its exact event ordering."""
    method = EventAlignedQuadrature(
        upper_stock=600.0,
        stock_points=20_001,
        normal_order=1_024,
    )
    observed = _price(_one_dividend_schedule(), method)
    assert observed == pytest.approx(11.656_448_947_710_173, abs=2.0e-4)


def test_quadrature_method_rejects_unpinned_rules_and_insufficient_boundaries() -> None:
    """Reject undeclared numerical substitutions before constructing a method."""
    with pytest.raises(InputError, match="normal_order must be one of"):
        EventAlignedQuadrature(upper_stock=600.0, stock_points=20_001, normal_order=255)
    method = EventAlignedQuadrature(
        upper_stock=100.0,
        stock_points=20_001,
        normal_order=64,
    )
    with pytest.raises(InputError, match="must be strictly above"):
        _price(_one_dividend_schedule(), method)


def test_quadrature_rejects_curve_grid_mismatch_without_repair() -> None:
    """Reject curve/schedule grid mismatch instead of aligning it implicitly."""
    schedule = _one_dividend_schedule()
    method = EventAlignedQuadrature(
        upper_stock=600.0,
        stock_points=1_001,
        normal_order=64,
    )
    with pytest.raises(InputError, match="different time grids"):
        price_event_aligned_call(
            initial_spot=100.0,
            strike=100.0,
            schedule=schedule,
            curve=LogDiscountCurve(times=(0.0, 0.4, 1.0), log_discounts=(0.0, -0.024, -0.06)),
            dividend_yield_curve=CumulativeDividendYieldCurve.zero(times=schedule.times),
            volatility=ConstantVolatility(annualized_decimal=0.30),
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=1),
            method=method,
        )
