"""Public contract-parallel LSM integration and fail-early tests."""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest
from finoptions import (
    BatchDividends,
    CashDividendBatch,
    ConstantVolatility,
    ContinuousDividendYieldBatch,
    CpuExecutor,
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EscrowedCashDividendBatch,
    EscrowedVolatilityConvention,
    EventSchedule,
    ExerciseTieRule,
    InputError,
    LogDiscountCurve,
    Lsm,
    LsmBatchResult,
    LsmStrikeGroups,
    OriginExercise,
    RaggedCashDividends,
    RandomStream,
    RegressionFitDiagnostics,
    UniformExerciseMesh,
    price_lsm_call,
    price_lsm_call_batch,
    price_lsm_call_grouped_batch,
)

_PATHS = 4_096
_INTERVALS = 8
_ROWS = 2
_ORIGIN_INTRINSIC = 100.0

Float64Vector = npt.NDArray[np.float64]


class BatchInputs(TypedDict):
    """Statically precise keyword columns shared by batch integration tests."""

    initial_spots: Float64Vector
    strikes: Float64Vector
    maturities: Float64Vector
    rates: Float64Vector
    volatilities: Float64Vector
    dividends: BatchDividends
    mesh: UniformExerciseMesh
    method: Lsm


def method(
    *,
    training_domain: int = 100,
    evaluation_domain: int = 10_000,
    polynomial_degree: int = 4,
) -> Lsm:
    """Build one scalar or row-addressed independent-policy identity."""
    return Lsm(
        training_paths=_PATHS,
        evaluation_paths=_PATHS,
        training_stream=RandomStream(seed=7, domain=training_domain),
        evaluation_stream=RandomStream(seed=11, domain=evaluation_domain),
        polynomial_degree=polynomial_degree,
        rank_tolerance=1.0e-12,
        condition_limit=1.0e8,
        tie_rule=ExerciseTieRule.STRICTLY_GREATER,
    )


def mesh(*, origin: OriginExercise = OriginExercise.EXCLUDED) -> UniformExerciseMesh:
    """Build the explicit uniform Bermudan exercise geometry."""
    return UniformExerciseMesh(base_intervals=_INTERVALS, origin_exercise=origin)


def continuous_inputs() -> BatchInputs:
    """Return two constant-parameter rows without implicit dtype conversion."""
    return {
        "initial_spots": np.array([100.0, 120.0], dtype=np.float64),
        "strikes": np.array([100.0, 110.0], dtype=np.float64),
        "maturities": np.array([1.0, 0.75], dtype=np.float64),
        "rates": np.array([0.05, 0.03], dtype=np.float64),
        "volatilities": np.array([0.25, 0.30], dtype=np.float64),
        "dividends": ContinuousDividendYieldBatch(
            annualized_decimals=np.array([0.02, 0.01], dtype=np.float64)
        ),
        "mesh": mesh(),
        "method": method(),
    }


def assert_batch_equal(left: LsmBatchResult, right: LsmBatchResult) -> None:
    """Require bitwise identity for every native result column."""
    assert np.array_equal(left.prices, right.prices)
    assert np.array_equal(left.sample_standard_deviations, right.sample_standard_deviations)
    assert np.array_equal(left.standard_errors, right.standard_errors)
    assert np.array_equal(left.origin_decisions_fitted, right.origin_decisions_fitted)
    assert np.array_equal(left.origin_exercised, right.origin_exercised)
    assert np.array_equal(left.regression_nodes, right.regression_nodes)
    assert np.array_equal(left.no_training_support_nodes, right.no_training_support_nodes)
    assert np.array_equal(
        left.minimum_in_the_money_training_paths,
        right.minimum_in_the_money_training_paths,
    )
    assert np.array_equal(
        left.maximum_regression_condition_numbers,
        right.maximum_regression_condition_numbers,
    )
    assert np.array_equal(left.early_exercise_paths, right.early_exercise_paths)
    assert np.array_equal(left.expiry_paths, right.expiry_paths)


def assert_batch_row_equal(
    grouped: LsmBatchResult,
    grouped_row: int,
    independent: LsmBatchResult,
) -> None:
    """Require one grouped strike to equal a one-row run on its exact group population."""
    assert independent.rows == 1
    assert grouped.prices[grouped_row] == independent.prices[0]
    assert (
        grouped.sample_standard_deviations[grouped_row] == independent.sample_standard_deviations[0]
    )
    assert grouped.standard_errors[grouped_row] == independent.standard_errors[0]
    assert grouped.origin_decisions_fitted[grouped_row] == independent.origin_decisions_fitted[0]
    assert grouped.origin_exercised[grouped_row] == independent.origin_exercised[0]
    assert grouped.regression_nodes[grouped_row] == independent.regression_nodes[0]
    assert (
        grouped.no_training_support_nodes[grouped_row] == independent.no_training_support_nodes[0]
    )
    assert (
        grouped.minimum_in_the_money_training_paths[grouped_row]
        == independent.minimum_in_the_money_training_paths[0]
    )
    assert (
        grouped.maximum_regression_condition_numbers[grouped_row]
        == independent.maximum_regression_condition_numbers[0]
    )
    assert grouped.early_exercise_paths[grouped_row] == independent.early_exercise_paths[0]
    assert grouped.expiry_paths[grouped_row] == independent.expiry_paths[0]


def test_one_contract_groups_match_row_batches_for_every_dividend_boundary_bitwise() -> None:
    """Python grouped dispatch preserves the accepted row algorithm on identical domains."""
    inputs = continuous_inputs()
    schedules = RaggedCashDividends(
        offsets=np.array([0, 2, 3], dtype=np.int64),
        times=np.array([0.2, 0.6, 0.4], dtype=np.float64),
        amounts=np.array([1.0, 1.0, 2.0], dtype=np.float64),
    )
    for dividends in (
        inputs["dividends"],
        CashDividendBatch(schedules=schedules, policy=DividendPolicy.SURVIVOR),
        EscrowedCashDividendBatch(
            schedules=schedules,
            volatility_convention=EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN,
        ),
    ):
        row = price_lsm_call_batch(
            initial_spots=inputs["initial_spots"],
            strikes=inputs["strikes"],
            maturities=inputs["maturities"],
            rates=inputs["rates"],
            volatilities=inputs["volatilities"],
            dividends=dividends,
            mesh=inputs["mesh"],
            executor=CpuExecutor(threads=1),
            method=inputs["method"],
        )
        grouped = price_lsm_call_grouped_batch(
            groups=LsmStrikeGroups(
                group_offsets=np.array([0, 1, 2], dtype=np.uint32),
                initial_spots=inputs["initial_spots"],
                strikes=inputs["strikes"],
                maturities=inputs["maturities"],
                rates=inputs["rates"],
                volatilities=inputs["volatilities"],
                dividends=dividends,
            ),
            mesh=inputs["mesh"],
            executor=CpuExecutor(threads=1),
            method=inputs["method"],
        )
        assert_batch_equal(grouped, row)


def test_every_shared_group_strike_matches_its_independent_same_population_run() -> None:
    """Path reuse changes cross-strike coupling, never a strike's declared LSM estimator."""
    spot = np.array([100.0], dtype=np.float64)
    strikes = np.array([85.0, 100.0, 115.0], dtype=np.float64)
    maturity = np.array([1.0], dtype=np.float64)
    rate = np.array([0.03], dtype=np.float64)
    volatility = np.array([0.25], dtype=np.float64)
    schedules = RaggedCashDividends(
        offsets=np.array([0, 2], dtype=np.int64),
        times=np.array([0.25, 0.75], dtype=np.float64),
        amounts=np.array([1.0, 1.25], dtype=np.float64),
    )
    for dividends in (
        ContinuousDividendYieldBatch(annualized_decimals=np.array([0.02], dtype=np.float64)),
        CashDividendBatch(schedules=schedules, policy=DividendPolicy.LIQUIDATOR),
        EscrowedCashDividendBatch(
            schedules=schedules,
            volatility_convention=EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN,
        ),
    ):
        grouped = price_lsm_call_grouped_batch(
            groups=LsmStrikeGroups(
                group_offsets=np.array([0, 3], dtype=np.uint32),
                initial_spots=spot,
                strikes=strikes,
                maturities=maturity,
                rates=rate,
                volatilities=volatility,
                dividends=dividends,
            ),
            mesh=mesh(),
            executor=CpuExecutor(threads=1),
            method=method(training_domain=733, evaluation_domain=91_733),
        )
        for row, strike in enumerate(strikes):
            independent = price_lsm_call_batch(
                initial_spots=spot,
                strikes=np.array([strike], dtype=np.float64),
                maturities=maturity,
                rates=rate,
                volatilities=volatility,
                dividends=dividends,
                mesh=mesh(),
                executor=CpuExecutor(threads=1),
                method=method(training_domain=733, evaluation_domain=91_733),
            )
            assert_batch_row_equal(grouped, row, independent)


def test_grouped_common_random_numbers_are_thread_invariant_and_strike_specific() -> None:
    """Duplicate strikes agree bitwise while a different strike retains its own fitted policy."""
    groups = LsmStrikeGroups(
        group_offsets=np.array([0, 3, 5], dtype=np.uint32),
        initial_spots=np.array([100.0, 120.0], dtype=np.float64),
        strikes=np.array([95.0, 100.0, 100.0, 105.0, 105.0], dtype=np.float64),
        maturities=np.array([1.0, 0.75], dtype=np.float64),
        rates=np.array([0.05, 0.03], dtype=np.float64),
        volatilities=np.array([0.25, 0.30], dtype=np.float64),
        dividends=ContinuousDividendYieldBatch(
            annualized_decimals=np.array([0.02, 0.01], dtype=np.float64)
        ),
    )
    serial = price_lsm_call_grouped_batch(
        groups=groups,
        mesh=mesh(),
        executor=CpuExecutor(threads=1),
        method=method(),
    )
    parallel = price_lsm_call_grouped_batch(
        groups=groups,
        mesh=mesh(),
        executor=CpuExecutor(threads=4),
        method=method(),
    )
    assert_batch_equal(parallel, serial)
    assert serial.prices[0] != serial.prices[1]
    for left, right in ((1, 2), (3, 4)):
        for values in (
            serial.prices,
            serial.sample_standard_deviations,
            serial.standard_errors,
            serial.regression_nodes,
            serial.no_training_support_nodes,
            serial.minimum_in_the_money_training_paths,
            serial.maximum_regression_condition_numbers,
            serial.early_exercise_paths,
            serial.expiry_paths,
        ):
            assert values[left] == values[right]


def test_grouped_offsets_and_compact_axes_fail_before_simulation_without_repair() -> None:
    """The public and native boundaries reject dtype, empty groups, and terminal mismatch."""
    with pytest.raises(InputError, match="group_offsets must have dtype uint32"):
        LsmStrikeGroups(
            group_offsets=cast(
                "npt.NDArray[np.uint32]",
                np.array([0, 1], dtype=np.int64),
            ),
            initial_spots=np.array([100.0], dtype=np.float64),
            strikes=np.array([100.0], dtype=np.float64),
            maturities=np.array([1.0], dtype=np.float64),
            rates=np.array([0.05], dtype=np.float64),
            volatilities=np.array([0.25], dtype=np.float64),
            dividends=ContinuousDividendYieldBatch(
                annualized_decimals=np.array([0.02], dtype=np.float64)
            ),
        )

    def invalid_groups(offsets: list[int]) -> LsmStrikeGroups:
        return LsmStrikeGroups(
            group_offsets=np.array(offsets, dtype=np.uint32),
            initial_spots=np.array([100.0, 120.0], dtype=np.float64),
            strikes=np.array([95.0, 100.0, 110.0], dtype=np.float64),
            maturities=np.array([1.0, 0.75], dtype=np.float64),
            rates=np.array([0.05, 0.03], dtype=np.float64),
            volatilities=np.array([0.25, 0.30], dtype=np.float64),
            dividends=ContinuousDividendYieldBatch(
                annualized_decimals=np.array([0.02, 0.01], dtype=np.float64)
            ),
        )

    with pytest.raises(InputError, match="range 0 must be nonempty"):
        price_lsm_call_grouped_batch(
            groups=invalid_groups([0, 0, 3]),
            mesh=mesh(),
            executor=CpuExecutor(threads=1),
            method=method(),
        )
    with pytest.raises(InputError, match="terminal offset is 2; expected 3"):
        price_lsm_call_grouped_batch(
            groups=invalid_groups([0, 1, 2]),
            mesh=mesh(),
            executor=CpuExecutor(threads=1),
            method=method(),
        )


def test_continuous_batch_is_thread_invariant_and_matches_scalar_row_bitwise() -> None:
    """Outer contract parallelism preserves the accepted scalar method exactly per row."""
    inputs = continuous_inputs()
    serial = price_lsm_call_batch(**inputs, executor=CpuExecutor(threads=1))
    parallel = price_lsm_call_batch(**inputs, executor=CpuExecutor(threads=4))
    assert_batch_equal(serial, parallel)

    times = tuple(float(node) / _INTERVALS for node in range(_INTERVALS + 1))
    schedule = EventSchedule(
        times=times,
        dividends=(),
        exercise_nodes=tuple(range(1, _INTERVALS)),
        expiry_node=_INTERVALS,
    )
    scalar = price_lsm_call(
        initial_spot=100.0,
        strike=100.0,
        schedule=schedule,
        curve=LogDiscountCurve(
            times=times,
            log_discounts=tuple(
                0.0 if node == 0 else -0.05 * time for node, time in enumerate(times)
            ),
        ),
        dividend_yield_curve=CumulativeDividendYieldCurve.constant(
            times=times,
            annualized_decimal=0.02,
        ),
        volatility=ConstantVolatility(0.25),
        dividend_policy=DividendPolicy.LIQUIDATOR,
        executor=CpuExecutor(threads=1),
        method=method(),
    )
    regressions = tuple(
        diagnostic
        for diagnostic in scalar.fit_diagnostics
        if isinstance(diagnostic, RegressionFitDiagnostics)
    )
    assert serial.prices[0] == scalar.price
    assert serial.sample_standard_deviations[0] == scalar.sample_standard_deviation
    assert serial.standard_errors[0] == scalar.standard_error
    assert serial.regression_nodes[0] == len(regressions)
    assert serial.minimum_in_the_money_training_paths[0] == min(
        diagnostic.in_the_money_paths for diagnostic in regressions
    )
    assert serial.maximum_regression_condition_numbers[0] == max(
        diagnostic.condition_number for diagnostic in regressions
    )
    assert serial.early_exercise_paths[0] == sum(scalar.exercise_counts[:-1])
    assert serial.expiry_paths[0] == scalar.exercise_counts[-1]


def test_cash_batch_inserts_exact_events_and_is_thread_invariant() -> None:
    """Ragged cash rows use exact event insertion and pre-jump exercise on every worker count."""
    dividends = CashDividendBatch(
        schedules=RaggedCashDividends(
            offsets=np.array([0, 2, 3], dtype=np.int64),
            times=np.array([0.2, 0.6, 0.4], dtype=np.float64),
            amounts=np.array([1.0, 1.0, 2.0], dtype=np.float64),
        ),
        policy=DividendPolicy.SURVIVOR,
    )
    common: BatchInputs = {
        "initial_spots": np.array([100.0, 120.0], dtype=np.float64),
        "strikes": np.array([100.0, 110.0], dtype=np.float64),
        "maturities": np.array([1.0, 0.75], dtype=np.float64),
        "rates": np.array([0.05, 0.03], dtype=np.float64),
        "volatilities": np.array([0.25, 0.30], dtype=np.float64),
        "dividends": dividends,
        "mesh": mesh(),
        "method": method(),
    }
    serial = price_lsm_call_batch(**common, executor=CpuExecutor(threads=1))
    parallel = price_lsm_call_batch(**common, executor=CpuExecutor(threads=4))
    assert_batch_equal(serial, parallel)
    assert serial.rows == _ROWS
    assert np.array_equal(
        serial.early_exercise_paths.astype(np.uint64) + serial.expiry_paths.astype(np.uint64),
        np.full(_ROWS, _PATHS, dtype=np.uint64),
    )
    for values in (
        serial.prices,
        serial.sample_standard_deviations,
        serial.standard_errors,
        serial.origin_decisions_fitted,
        serial.origin_exercised,
        serial.regression_nodes,
        serial.no_training_support_nodes,
        serial.minimum_in_the_money_training_paths,
        serial.maximum_regression_condition_numbers,
        serial.early_exercise_paths,
        serial.expiry_paths,
    ):
        assert not values.flags.writeable


def test_escrowed_batch_is_thread_invariant_and_volatility_semantics_are_explicit() -> None:
    """Origin spot matching is exactly the declared prepaid-forward volatility transformation."""
    schedules = RaggedCashDividends(
        offsets=np.array([0, 1], dtype=np.int64),
        times=np.array([0.5], dtype=np.float64),
        amounts=np.array([10.0], dtype=np.float64),
    )
    initial_spots = np.array([100.0], dtype=np.float64)
    strikes = np.array([100.0], dtype=np.float64)
    maturities = np.array([1.0], dtype=np.float64)
    rates = np.array([0.0], dtype=np.float64)
    spot_volatility = np.array([0.20], dtype=np.float64)
    prepaid_forward_volatility = np.array([0.20 * 100.0 / 90.0], dtype=np.float64)
    common = {
        "initial_spots": initial_spots,
        "strikes": strikes,
        "maturities": maturities,
        "rates": rates,
        "mesh": mesh(),
        "method": method(),
    }
    spot_matched = price_lsm_call_batch(
        **common,
        volatilities=spot_volatility,
        dividends=EscrowedCashDividendBatch(
            schedules=schedules,
            volatility_convention=EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN,
        ),
        executor=CpuExecutor(threads=1),
    )
    direct = price_lsm_call_batch(
        **common,
        volatilities=prepaid_forward_volatility,
        dividends=EscrowedCashDividendBatch(
            schedules=schedules,
            volatility_convention=EscrowedVolatilityConvention.PREPAID_FORWARD,
        ),
        executor=CpuExecutor(threads=4),
    )
    assert_batch_equal(spot_matched, direct)


def test_escrowed_batch_fails_before_simulation_when_cash_exhausts_spot() -> None:
    """The approximation never repairs an invalid nonpositive risky prepaid forward."""
    with pytest.raises(InputError, match=r"prepaid forward must be strictly positive.*row 0"):
        price_lsm_call_batch(
            initial_spots=np.array([100.0], dtype=np.float64),
            strikes=np.array([100.0], dtype=np.float64),
            maturities=np.array([1.0], dtype=np.float64),
            rates=np.array([0.0], dtype=np.float64),
            volatilities=np.array([0.20], dtype=np.float64),
            dividends=EscrowedCashDividendBatch(
                schedules=RaggedCashDividends(
                    offsets=np.array([0, 1], dtype=np.int64),
                    times=np.array([0.5], dtype=np.float64),
                    amounts=np.array([100.0], dtype=np.float64),
                ),
                volatility_convention=EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN,
            ),
            mesh=mesh(),
            method=method(),
            executor=CpuExecutor(threads=2),
        )


def test_included_origin_is_one_common_pre_event_exercise_decision() -> None:
    """Origin exercise is explicit, common to all holders, and precedes every future cash event."""
    result = price_lsm_call_batch(
        initial_spots=np.array([200.0], dtype=np.float64),
        strikes=np.array([100.0], dtype=np.float64),
        maturities=np.array([1.0], dtype=np.float64),
        rates=np.array([-1.0], dtype=np.float64),
        volatilities=np.array([0.30], dtype=np.float64),
        dividends=CashDividendBatch(
            schedules=RaggedCashDividends(
                offsets=np.array([0, 1], dtype=np.int64),
                times=np.array([0.125], dtype=np.float64),
                amounts=np.array([200.0], dtype=np.float64),
            ),
            policy=DividendPolicy.LIQUIDATOR,
        ),
        mesh=mesh(origin=OriginExercise.INCLUDED),
        executor=CpuExecutor(threads=2),
        method=method(polynomial_degree=0),
    )
    assert result.prices[0] == _ORIGIN_INTRINSIC
    assert result.sample_standard_deviations[0] == 0.0
    assert result.standard_errors[0] == 0.0
    assert result.origin_decisions_fitted[0]
    assert result.origin_exercised[0]
    assert result.early_exercise_paths[0] == _PATHS
    assert result.expiry_paths[0] == 0


def test_array_layout_random_intervals_and_cash_boundaries_fail_early() -> None:
    """No dtype copy, domain repair, dividend aggregation, or expiry event is inferred."""
    inputs = continuous_inputs()
    invalid_dtype = inputs.copy()
    invalid_dtype["initial_spots"] = cast(
        "Float64Vector",
        np.array([100.0, 120.0], dtype=np.float32),
    )
    with pytest.raises(InputError, match="initial_spots must have dtype float64"):
        price_lsm_call_batch(
            **invalid_dtype,
            executor=CpuExecutor(threads=1),
        )
    invalid_layout = inputs.copy()
    invalid_layout["initial_spots"] = np.arange(4.0, dtype=np.float64)[::2]
    with pytest.raises(InputError, match="initial_spots must be C-contiguous"):
        price_lsm_call_batch(
            **invalid_layout,
            executor=CpuExecutor(threads=1),
        )
    colliding_streams = inputs.copy()
    colliding_streams["method"] = method(training_domain=100, evaluation_domain=101)
    with pytest.raises(InputError, match="overlap evaluation domains"):
        price_lsm_call_batch(
            **colliding_streams,
            executor=CpuExecutor(threads=1),
        )

    one_row: BatchInputs = {
        "initial_spots": np.array([100.0], dtype=np.float64),
        "strikes": np.array([100.0], dtype=np.float64),
        "maturities": np.array([1.0], dtype=np.float64),
        "rates": np.array([0.05], dtype=np.float64),
        "volatilities": np.array([0.25], dtype=np.float64),
        "dividends": ContinuousDividendYieldBatch(
            annualized_decimals=np.array([0.0], dtype=np.float64)
        ),
        "mesh": mesh(),
        "method": method(),
    }
    duplicate = CashDividendBatch(
        schedules=RaggedCashDividends(
            offsets=np.array([0, 2], dtype=np.int64),
            times=np.array([0.25, 0.25], dtype=np.float64),
            amounts=np.array([1.0, 2.0], dtype=np.float64),
        ),
        policy=DividendPolicy.LIQUIDATOR,
    )
    duplicate_inputs = one_row.copy()
    duplicate_inputs["dividends"] = duplicate
    with pytest.raises(InputError, match="repeats exact event time"):
        price_lsm_call_batch(**duplicate_inputs, executor=CpuExecutor(threads=1))

    at_expiry = CashDividendBatch(
        schedules=RaggedCashDividends(
            offsets=np.array([0, 1], dtype=np.int64),
            times=np.array([1.0], dtype=np.float64),
            amounts=np.array([1.0], dtype=np.float64),
        ),
        policy=DividendPolicy.LIQUIDATOR,
    )
    expiry_inputs = one_row.copy()
    expiry_inputs["dividends"] = at_expiry
    with pytest.raises(InputError, match="strictly before maturity"):
        price_lsm_call_batch(**expiry_inputs, executor=CpuExecutor(threads=1))


def test_mutated_numpy_values_are_revalidated_by_the_native_financial_boundary() -> None:
    """Post-construction mutation cannot bypass row-level financial validation."""
    inputs = continuous_inputs()
    dividends = inputs["dividends"]
    assert isinstance(dividends, ContinuousDividendYieldBatch)
    dividends.annualized_decimals[1] = -0.01
    with pytest.raises(InputError, match=r"continuous_dividend_yields.*row 1"):
        price_lsm_call_batch(**inputs, executor=CpuExecutor(threads=1))

    cash_amounts = np.array([1.0], dtype=np.float64)
    cash_inputs = inputs.copy()
    cash_inputs["dividends"] = CashDividendBatch(
        schedules=RaggedCashDividends(
            offsets=np.array([0, 1, 1], dtype=np.int64),
            times=np.array([0.25], dtype=np.float64),
            amounts=cash_amounts,
        ),
        policy=DividendPolicy.LIQUIDATOR,
    )
    cash_amounts[0] = -1.0
    with pytest.raises(InputError, match=r"cash-dividend amount at row 0, event 0"):
        price_lsm_call_batch(**cash_inputs, executor=CpuExecutor(threads=2))
