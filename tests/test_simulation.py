"""Native GBM path interface tests against independent NumPy calculations."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from finoptions import (
    CashDividend,
    ConstantVolatility,
    CpuExecutor,
    DividendPolicy,
    EventSchedule,
    InputError,
    LogDiscountCurve,
    SimulationError,
    simulate_gbm,
)

PATHS = 2


def inputs(
    *,
    dividend: CashDividend | None = None,
    exercises: tuple[int, ...] = (1, 2),
) -> tuple[EventSchedule, LogDiscountCurve, ConstantVolatility]:
    """Build one irregular, event-aligned synthetic model."""
    times = (0.0, 0.25, 0.5, 1.0)
    schedule = EventSchedule(
        times=times,
        dividends=() if dividend is None else (dividend,),
        exercise_nodes=exercises,
        expiry_node=3,
    )
    curve = LogDiscountCurve(
        times=times,
        log_discounts=(0.0, -0.01, -0.02, -0.05),
    )
    return schedule, curve, ConstantVolatility(0.20)


def test_matches_independent_adjacent_transition_and_time_major_layout() -> None:
    """The Python boundary preserves exact normal indexing and financial coefficients."""
    schedule, curve, volatility = inputs()
    normals = np.array(
        [[0.0, 1.0], [0.5, -0.5], [-1.0, 0.25]],
        dtype=np.float64,
    )
    result = simulate_gbm(
        initial_spot=100.0,
        schedule=schedule,
        curve=curve,
        volatility=volatility,
        dividend_policy=DividendPolicy.LIQUIDATOR,
        executor=CpuExecutor(threads=1),
        normals=normals,
    )

    expected = np.empty((3, 2), dtype=np.float64)
    current = np.full(2, 100.0, dtype=np.float64)
    output_row = 0
    for interval, (left, right) in enumerate(pairwise(schedule.times)):
        variance = volatility.annualized_decimal**2 * (right - left)
        log_drift = (
            -(curve.log_discounts[interval + 1] - curve.log_discounts[interval]) - 0.5 * variance
        )
        current *= np.exp(log_drift + np.sqrt(variance) * normals[interval])
        node = interval + 1
        if node in result.observation_nodes:
            expected[output_row] = current
            output_row += 1

    assert result.observation_nodes == (1, 2, 3)
    assert result.path_count == PATHS
    np.testing.assert_allclose(result.states, expected, rtol=2e-15, atol=0.0)
    assert not result.states.flags.writeable
    assert np.shares_memory(result.values_at(1), result.states)


def test_observes_pre_dividend_exercise_state_and_uses_post_dividend_state_afterward() -> None:
    """A same-node dividend is applied only after the exercise observation is stored."""
    schedule, curve, volatility = inputs(
        dividend=CashDividend(node=1, amount=200.0),
        exercises=(1,),
    )
    result = simulate_gbm(
        initial_spot=100.0,
        schedule=schedule,
        curve=curve,
        volatility=volatility,
        dividend_policy=DividendPolicy.LIQUIDATOR,
        executor=CpuExecutor(threads=1),
        normals=np.zeros((3, 1), dtype=np.float64),
    )
    assert result.values_at(1)[0] > 0.0
    assert result.values_at(3)[0] == 0.0


def test_rejects_implicit_array_conversion_and_nonfinite_values() -> None:
    """Dtype/layout conversion and invalid shocks fail instead of copying or recovering."""
    schedule, curve, volatility = inputs()
    common = {
        "initial_spot": 100.0,
        "schedule": schedule,
        "curve": curve,
        "volatility": volatility,
        "dividend_policy": DividendPolicy.LIQUIDATOR,
        "executor": CpuExecutor(threads=1),
    }
    with pytest.raises(InputError, match="dtype float64"):
        simulate_gbm(**common, normals=np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(InputError, match="C-contiguous"):
        simulate_gbm(**common, normals=np.zeros((2, 3), dtype=np.float64).T)

    invalid = np.zeros((3, 2), dtype=np.float64)
    invalid[2, 1] = np.nan
    with pytest.raises(InputError, match="interval 2, path 1"):
        simulate_gbm(**common, normals=invalid)


def test_distinguishes_invalid_input_from_numerical_transition_failure() -> None:
    """Shape errors are input failures while finite-input overflow is an execution failure."""
    schedule, curve, volatility = inputs()
    with pytest.raises(InputError, match="interval-row"):
        simulate_gbm(
            initial_spot=100.0,
            schedule=schedule,
            curve=curve,
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=1),
            normals=np.zeros((2, 2), dtype=np.float64),
        )

    overflow = np.zeros((3, 1), dtype=np.float64)
    overflow[0, 0] = 1e308
    with pytest.raises(SimulationError, match="path 0"):
        simulate_gbm(
            initial_spot=100.0,
            schedule=schedule,
            curve=curve,
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=1),
            normals=overflow,
        )


def test_explicit_parallel_executor_is_bitwise_identical_to_serial() -> None:
    """Thread scheduling does not alter independent path identity or arithmetic."""
    schedule, curve, volatility = inputs()
    normals = np.ascontiguousarray(
        np.random.default_rng(7).standard_normal((3, 10_000)),
        dtype=np.float64,
    )
    common = {
        "initial_spot": 100.0,
        "schedule": schedule,
        "curve": curve,
        "volatility": volatility,
        "dividend_policy": DividendPolicy.LIQUIDATOR,
        "normals": normals,
    }
    serial = simulate_gbm(**common, executor=CpuExecutor(threads=1))
    parallel = simulate_gbm(**common, executor=CpuExecutor(threads=2))
    assert np.array_equal(serial.states, parallel.states)


def test_executor_and_parallel_failure_reporting_are_strict_and_deterministic() -> None:
    """Executor construction fails early and parallel overflow reports the first path."""
    with pytest.raises(InputError, match="positive plain int"):
        CpuExecutor(threads=0)
    with pytest.raises(InputError, match="positive plain int"):
        CpuExecutor(threads=True)

    schedule, curve, volatility = inputs()
    overflow = np.zeros((3, 4), dtype=np.float64)
    overflow[0, 3] = 1e308
    overflow[0, 1] = 1e308
    with pytest.raises(SimulationError, match="path 1"):
        simulate_gbm(
            initial_spot=100.0,
            schedule=schedule,
            curve=curve,
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=2),
            normals=overflow,
        )
