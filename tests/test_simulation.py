"""Native GBM path interface tests against independent NumPy calculations."""

from __future__ import annotations

import sys
from itertools import pairwise
from math import erfc, exp, sqrt
from typing import cast

import numpy as np
import pytest
from finoptions import (
    CashDividend,
    ConstantVolatility,
    CpuExecutor,
    CpuMathBackend,
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EventSchedule,
    ExecutionError,
    GbmPaths,
    InputError,
    LogDiscountCurve,
    RandomStream,
    SimulationError,
    simulate_gbm,
    simulate_gbm_seeded,
)

PATHS = 2
TAIL_THRESHOLD = 3.0


def zero_yield(schedule: EventSchedule) -> CumulativeDividendYieldCurve:
    """Build the explicit zero continuous proportional-dividend-yield input."""
    return CumulativeDividendYieldCurve.zero(times=schedule.times)


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
        dividend_yield_curve=zero_yield(schedule),
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
        dividend_yield_curve=zero_yield(schedule),
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
        "dividend_yield_curve": zero_yield(schedule),
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
            dividend_yield_curve=zero_yield(schedule),
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
            dividend_yield_curve=zero_yield(schedule),
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
        "dividend_yield_curve": zero_yield(schedule),
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
            dividend_yield_curve=zero_yield(schedule),
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=2),
            normals=overflow,
        )


def test_cpu_math_backend_is_explicit_strict_and_platform_checked() -> None:
    """The numerical backend is typed, inspectable, and never silently substituted."""
    scalar = CpuExecutor(threads=1, math_backend=CpuMathBackend.RUST_STD_F64)
    assert scalar.math_backend is CpuMathBackend.RUST_STD_F64
    with pytest.raises(InputError, match="must be CpuMathBackend"):
        CpuExecutor(threads=1, math_backend=cast("CpuMathBackend", "rust_std_f64"))

    if sys.platform != "darwin":
        with pytest.raises(ExecutionError, match="unavailable"):
            CpuExecutor(
                threads=1,
                math_backend=CpuMathBackend.APPLE_ACCELERATE_VFORCE,
            )
        return

    schedule, curve, volatility = inputs()
    normals = np.ascontiguousarray(
        np.random.default_rng(19).standard_normal((3, 257)),
        dtype=np.float64,
    )
    external_common = {
        "initial_spot": 100.0,
        "schedule": schedule,
        "curve": curve,
        "dividend_yield_curve": zero_yield(schedule),
        "volatility": volatility,
        "dividend_policy": DividendPolicy.LIQUIDATOR,
        "normals": normals,
    }
    external_scalar = simulate_gbm(**external_common, executor=scalar)
    external_apple_serial = simulate_gbm(
        **external_common,
        executor=CpuExecutor(
            threads=1,
            math_backend=CpuMathBackend.APPLE_ACCELERATE_VFORCE,
        ),
    )
    external_apple_parallel = simulate_gbm(
        **external_common,
        executor=CpuExecutor(
            threads=4,
            math_backend=CpuMathBackend.APPLE_ACCELERATE_VFORCE,
        ),
    )
    assert np.array_equal(external_apple_serial.states, external_apple_parallel.states)
    np.testing.assert_allclose(
        external_apple_serial.states,
        external_scalar.states,
        rtol=4e-15,
        atol=0.0,
    )

    common = {
        "initial_spot": 100.0,
        "schedule": schedule,
        "curve": curve,
        "dividend_yield_curve": zero_yield(schedule),
        "volatility": volatility,
        "dividend_policy": DividendPolicy.LIQUIDATOR,
        "paths": 257,
        "random_stream": RandomStream(seed=7, domain=11),
    }
    serial = simulate_gbm_seeded(
        **common,
        executor=CpuExecutor(
            threads=1,
            math_backend=CpuMathBackend.APPLE_ACCELERATE_VFORCE,
        ),
    )
    parallel = simulate_gbm_seeded(
        **common,
        executor=CpuExecutor(
            threads=4,
            math_backend=CpuMathBackend.APPLE_ACCELERATE_VFORCE,
        ),
    )
    assert np.array_equal(serial.states, parallel.states)


def test_seeded_paths_preserve_stream_identity_across_workers_and_batch_sizes() -> None:
    """Seed/domain/path identity is independent of worker scheduling and larger path batches."""
    schedule, curve, volatility = inputs()

    def run(*, executor: CpuExecutor, paths: int, random_stream: RandomStream) -> GbmPaths:
        return simulate_gbm_seeded(
            initial_spot=100.0,
            schedule=schedule,
            curve=curve,
            dividend_yield_curve=zero_yield(schedule),
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=executor,
            paths=paths,
            random_stream=random_stream,
        )

    stream = RandomStream(seed=7, domain=11)
    serial = run(executor=CpuExecutor(threads=1), paths=10_000, random_stream=stream)
    parallel = run(executor=CpuExecutor(threads=4), paths=10_000, random_stream=stream)
    larger = run(executor=CpuExecutor(threads=4), paths=10_013, random_stream=stream)

    assert np.array_equal(serial.states, parallel.states)
    assert np.array_equal(serial.states, larger.states[:, :10_000])
    assert not serial.states.flags.writeable
    different_domain = run(
        executor=CpuExecutor(threads=4),
        paths=10_000,
        random_stream=RandomStream(seed=7, domain=12),
    )
    assert not np.array_equal(serial.states, different_domain.states)


def test_seeded_public_contract_rejects_implicit_or_ambiguous_inputs() -> None:
    """Random namespaces and path counts reject booleans, signs, ranges, and duck typing."""
    with pytest.raises(InputError, match="seed must be a plain int"):
        RandomStream(seed=True, domain=0)
    with pytest.raises(InputError, match="seed must be a plain int"):
        RandomStream(seed=-1, domain=0)
    with pytest.raises(InputError, match="domain must be a plain int"):
        RandomStream(seed=0, domain=2**32)

    schedule, curve, volatility = inputs()

    def run(*, paths: int, random_stream: RandomStream) -> GbmPaths:
        return simulate_gbm_seeded(
            initial_spot=100.0,
            schedule=schedule,
            curve=curve,
            dividend_yield_curve=zero_yield(schedule),
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=1),
            paths=paths,
            random_stream=random_stream,
        )

    with pytest.raises(InputError, match="paths must be a plain int"):
        run(paths=True, random_stream=RandomStream(seed=0, domain=0))
    with pytest.raises(InputError, match="paths must be a plain int"):
        run(paths=0, random_stream=RandomStream(seed=0, domain=0))
    with pytest.raises(InputError, match="random_stream must be RandomStream"):
        run(paths=1, random_stream=cast("RandomStream", (0, 0)))


def test_seeded_normal_transform_and_risk_neutral_terminal_moment() -> None:
    """A fixed large stream satisfies predeclared normal and exact GBM moment diagnostics."""
    paths = 262_144
    schedule = EventSchedule(
        times=(0.0, 1.0),
        dividends=(),
        exercise_nodes=(),
        expiry_node=1,
    )
    curve = LogDiscountCurve(times=(0.0, 1.0), log_discounts=(0.0, -0.05))
    dividend_yield_curve = CumulativeDividendYieldCurve.constant(
        times=schedule.times,
        annualized_decimal=0.02,
    )
    volatility = ConstantVolatility(0.20)
    result = simulate_gbm_seeded(
        initial_spot=100.0,
        schedule=schedule,
        curve=curve,
        dividend_yield_curve=dividend_yield_curve,
        volatility=volatility,
        dividend_policy=DividendPolicy.LIQUIDATOR,
        executor=CpuExecutor(threads=4),
        paths=paths,
        random_stream=RandomStream(seed=0x01D6328F7A4C915B, domain=0x17290042),
    )
    terminal = result.states[0]
    normals = (np.log(terminal / 100.0) - (0.05 - 0.02 - 0.5 * 0.20**2)) / 0.20

    mean_tolerance = 5.0 / sqrt(paths)
    variance_tolerance = 5.0 * sqrt(2.0 / (paths - 1))
    assert abs(float(np.mean(normals))) <= mean_tolerance
    assert abs(float(np.var(normals, ddof=1)) - 1.0) <= variance_tolerance

    tail_probability = erfc(TAIL_THRESHOLD / sqrt(2.0))
    tail_count = int(np.count_nonzero(np.abs(normals) > TAIL_THRESHOLD))
    expected_tail_count = paths * tail_probability
    tail_standard_deviation = sqrt(paths * tail_probability * (1.0 - tail_probability))
    assert abs(tail_count - expected_tail_count) <= 6.0 * tail_standard_deviation

    expected_terminal_mean = 100.0 * exp(0.05 - 0.02)
    terminal_variance = 100.0**2 * exp(2.0 * (0.05 - 0.02)) * (exp(0.20**2) - 1.0)
    terminal_standard_error = sqrt(terminal_variance / paths)
    assert abs(float(np.mean(terminal)) - expected_terminal_mean) <= 5.0 * terminal_standard_error


def test_seeded_parallel_transition_failure_reports_lowest_path() -> None:
    """Finite-input overflow remains a typed execution error with deterministic path context."""
    schedule, curve, volatility = inputs()
    with pytest.raises(SimulationError, match="path 0"):
        simulate_gbm_seeded(
            initial_spot=float.fromhex("0x1.fffffffffffffp+1023"),
            schedule=schedule,
            curve=curve,
            dividend_yield_curve=zero_yield(schedule),
            volatility=volatility,
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=CpuExecutor(threads=4),
            paths=1_000,
            random_stream=RandomStream(seed=7, domain=11),
        )
