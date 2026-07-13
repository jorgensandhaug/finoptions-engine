"""Public independent-policy LSM integration tests."""

from __future__ import annotations

from typing import cast

import pytest
from finoptions import (
    CashDividend,
    ConstantVolatility,
    CpuExecutor,
    DividendPolicy,
    EventSchedule,
    ExerciseTieRule,
    InputError,
    LogDiscountCurve,
    Lsm,
    LsmError,
    OriginDecisionDiagnostics,
    RandomStream,
    RegressionFitDiagnostics,
    price_lsm_call,
)

EVALUATION_PATHS = 32_768
BASIS_COLUMNS = 5
ORIGIN_EXERCISE_VALUE = 100.0


def one_dividend_inputs() -> tuple[EventSchedule, LogDiscountCurve, ConstantVolatility]:
    """Return the published one-dividend geometry with one justified exercise node."""
    schedule = EventSchedule(
        times=(0.0, 0.5, 1.0),
        dividends=(CashDividend(node=1, amount=7.0),),
        exercise_nodes=(1,),
        expiry_node=2,
    )
    curve = LogDiscountCurve(
        times=schedule.times,
        log_discounts=(0.0, -0.03, -0.06),
    )
    return schedule, curve, ConstantVolatility(0.30)


def method(*, paths: int = EVALUATION_PATHS) -> Lsm:
    """Build one explicit fit/evaluation identity."""
    return Lsm(
        training_paths=paths,
        evaluation_paths=paths,
        training_stream=RandomStream(seed=10_000, domain=1),
        evaluation_stream=RandomStream(seed=20_000, domain=2),
        polynomial_degree=4,
        rank_tolerance=1.0e-12,
        condition_limit=1.0e8,
        tie_rule=ExerciseTieRule.STRICTLY_GREATER,
    )


def test_public_lsm_is_thread_invariant_and_returns_complete_diagnostics() -> None:
    """Worker scheduling does not alter the fixed policy, payoff mean, or diagnostics."""
    schedule, curve, volatility = one_dividend_inputs()
    lsm_method = method()
    common = {
        "initial_spot": 100.0,
        "strike": 100.0,
        "schedule": schedule,
        "curve": curve,
        "volatility": volatility,
        "dividend_policy": DividendPolicy.LIQUIDATOR,
        "method": lsm_method,
    }
    serial = price_lsm_call(**common, executor=CpuExecutor(threads=1))
    parallel = price_lsm_call(**common, executor=CpuExecutor(threads=4))

    assert serial == parallel
    assert serial.paths == EVALUATION_PATHS
    assert serial.observation_nodes == (1, 2)
    assert sum(serial.exercise_counts) == serial.paths
    assert serial.sample_standard_deviation is not None
    assert serial.standard_error is not None
    assert len(serial.fit_diagnostics) == 1
    regression = serial.fit_diagnostics[0]
    assert isinstance(regression, RegressionFitDiagnostics)
    assert regression.node == 1
    assert regression.observations == regression.in_the_money_paths
    assert regression.columns == regression.rank == BASIS_COLUMNS
    assert regression.condition_number < lsm_method.condition_limit


def test_origin_exercise_is_reported_as_one_common_decision() -> None:
    """A deterministic valuation decision does not enter a degenerate cross-sectional fit."""
    schedule = EventSchedule(
        times=(0.0, 0.5, 1.0),
        dividends=(CashDividend(node=1, amount=200.0),),
        exercise_nodes=(0,),
        expiry_node=2,
    )
    curve = LogDiscountCurve(times=schedule.times, log_discounts=(0.0, 0.0, 0.0))
    result = price_lsm_call(
        initial_spot=200.0,
        strike=100.0,
        schedule=schedule,
        curve=curve,
        volatility=ConstantVolatility(0.30),
        dividend_policy=DividendPolicy.LIQUIDATOR,
        executor=CpuExecutor(threads=2),
        method=method(paths=1_000),
    )
    assert result.price == ORIGIN_EXERCISE_VALUE
    assert result.exercise_counts == (1_000, 0)
    origin = result.fit_diagnostics[0]
    assert isinstance(origin, OriginDecisionDiagnostics)
    assert origin.exercise


def test_lsm_configuration_and_regression_fail_instead_of_substituting() -> None:
    """Invalid identity and unsupported regression populations raise before returning a price."""
    invalid_paths: object = True
    with pytest.raises(InputError, match="training_paths must be a plain int"):
        method(paths=cast("int", invalid_paths))
    with pytest.raises(InputError, match="different domains"):
        Lsm(
            training_paths=100,
            evaluation_paths=100,
            training_stream=RandomStream(seed=1, domain=7),
            evaluation_stream=RandomStream(seed=2, domain=7),
        )
    with pytest.raises(InputError, match="polynomial_degree"):
        Lsm(
            training_paths=100,
            evaluation_paths=100,
            training_stream=RandomStream(seed=1, domain=1),
            evaluation_stream=RandomStream(seed=1, domain=2),
            polynomial_degree=5,
        )
    with pytest.raises(InputError, match="tie_rule must be ExerciseTieRule"):
        Lsm(
            training_paths=100,
            evaluation_paths=100,
            training_stream=RandomStream(seed=1, domain=1),
            evaluation_stream=RandomStream(seed=1, domain=2),
            tie_rule=cast("ExerciseTieRule", "strictly_greater"),
        )

    schedule, curve, volatility = one_dividend_inputs()
    with pytest.raises(LsmError, match="requires at least 5"):
        price_lsm_call(
            initial_spot=100.0,
            strike=100.0,
            schedule=schedule,
            curve=curve,
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=1),
            method=method(paths=4),
        )
