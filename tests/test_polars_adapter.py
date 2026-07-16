"""Polars lazy projection, explicit materialization, and native-pricing boundary tests."""

from __future__ import annotations

import gc
from typing import cast

import numpy as np
import polars as pl
import pytest
from finoptions import (
    CashDividendBatch,
    CpuExecutor,
    DividendPolicy,
    ExerciseTieRule,
    InsufficientExerciseSupportPolicy,
    Lsm,
    LsmBatchResult,
    MaturityBoundary,
    OriginExercise,
    RandomStream,
    UniformExerciseMesh,
    price_lsm_call_batch,
    price_lsm_call_grouped_batch,
)
from finoptions.polars import (
    BufferPolicy,
    CashDividendColumns,
    ContinuousYieldColumn,
    LsmCallColumns,
    LsmStrikeGroupColumns,
    PolarsAdapterTypeError,
    PolarsBufferError,
    PolarsContentError,
    PolarsSchemaError,
    lsm_result_frame,
    prepare_lsm_call_batch,
    prepare_lsm_strike_groups,
    price_lsm_call_frame,
    price_lsm_strike_group_frames,
    select_lsm_call_inputs,
    select_lsm_strike_group_inputs,
)

_COLUMNS = LsmCallColumns(
    initial_spot="spot",
    strike="strike",
    maturity="maturity",
    rate="rate",
    volatility="volatility",
)
_YIELD = ContinuousYieldColumn("yield")
_GROUP_COLUMNS = LsmStrikeGroupColumns(
    initial_spot="spot",
    maturity="maturity",
    rate="rate",
    volatility="volatility",
)
_ROWS = 2
_GROUPS = 2
_GROUP_ROWS = 3
_FIRST_ROW_ID = 9


def _continuous_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": [9, 4],
            "spot": [100.0, 120.0],
            "strike": [100.0, 110.0],
            "maturity": [1.0, 0.75],
            "rate": [0.05, 0.03],
            "volatility": [0.25, 0.30],
            "yield": [0.02, 0.01],
        },
        schema_overrides={
            "spot": pl.Float64,
            "strike": pl.Float64,
            "maturity": pl.Float64,
            "rate": pl.Float64,
            "volatility": pl.Float64,
            "yield": pl.Float64,
        },
    )


def _group_state_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "state_id": [8, 3],
            "spot": [120.0, 100.0],
            "maturity": [0.75, 1.0],
            "rate": [0.03, 0.05],
            "volatility": [0.30, 0.25],
            "yield": [0.01, 0.02],
        },
        schema_overrides={
            "spot": pl.Float64,
            "maturity": pl.Float64,
            "rate": pl.Float64,
            "volatility": pl.Float64,
            "yield": pl.Float64,
        },
    )


def _group_strike_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"contract_id": [5, 2, 9], "strike": [110.0, 125.0, 100.0]},
        schema_overrides={"strike": pl.Float64},
    )


def _group_offsets() -> pl.Series:
    return pl.Series("group_offsets", [0, 2, 3], dtype=pl.UInt32)


def _method() -> Lsm:
    return Lsm(
        training_paths=512,
        evaluation_paths=512,
        training_stream=RandomStream(seed=7, domain=100),
        evaluation_stream=RandomStream(seed=11, domain=10_000),
        insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
        polynomial_degree=4,
        rank_tolerance=1.0e-12,
        condition_limit=1.0e8,
        tie_rule=ExerciseTieRule.STRICTLY_GREATER,
    )


def _mesh() -> UniformExerciseMesh:
    return UniformExerciseMesh(base_intervals=8, origin_exercise=OriginExercise.EXCLUDED)


def test_lazy_selection_projects_exact_columns_without_collecting_or_casting() -> None:
    """Projection preserves laziness, exact dtypes, declared order, and no extra columns."""
    selected = select_lsm_call_inputs(
        _continuous_frame().lazy(),
        columns=_COLUMNS,
        dividends=_YIELD,
    )

    assert type(selected) is pl.LazyFrame
    assert selected.collect_schema() == pl.Schema(
        {
            "spot": pl.Float64,
            "strike": pl.Float64,
            "maturity": pl.Float64,
            "rate": pl.Float64,
            "volatility": pl.Float64,
            "yield": pl.Float64,
        }
    )
    assert selected.collect().columns == [
        "spot",
        "strike",
        "maturity",
        "rate",
        "volatility",
        "yield",
    ]


def test_lazy_selection_rejects_wrong_dtype_instead_of_casting() -> None:
    """The lazy adapter does not silently coerce an integer strike column."""
    invalid = _continuous_frame().with_columns(pl.col("strike").cast(pl.Int64)).lazy()

    with pytest.raises(PolarsSchemaError, match=r"strike.*Float64.*Int64"):
        select_lsm_call_inputs(invalid, columns=_COLUMNS, dividends=_YIELD)


def test_grouped_lazy_selection_preserves_two_axes_without_grouping_or_sorting() -> None:
    """The compact state and strike plans retain their independent declared row orders."""
    states, strikes = select_lsm_strike_group_inputs(
        _group_state_frame().lazy(),
        _group_strike_frame().lazy(),
        state_columns=_GROUP_COLUMNS,
        strike_column="strike",
        dividends=_YIELD,
    )

    assert states.collect_schema() == pl.Schema(
        {
            "spot": pl.Float64,
            "maturity": pl.Float64,
            "rate": pl.Float64,
            "volatility": pl.Float64,
            "yield": pl.Float64,
        }
    )
    assert strikes.collect_schema() == pl.Schema({"strike": pl.Float64})
    assert states.collect().get_column("spot").to_list() == [120.0, 100.0]
    assert strikes.collect().get_column("strike").to_list() == [110.0, 125.0, 100.0]


def test_eager_boundary_rejects_lazy_frame_explicitly() -> None:
    """Stateful pricing requires the caller to own the collection boundary."""
    with pytest.raises(PolarsAdapterTypeError, match=r"materialized polars.DataFrame"):
        prepare_lsm_call_batch(
            cast("pl.DataFrame", _continuous_frame().lazy()),
            columns=_COLUMNS,
            dividends=_YIELD,
        )


def test_continuous_frame_pricing_matches_direct_native_batch_bitwise() -> None:
    """The adapter changes neither native inputs nor any result diagnostic."""
    frame = _continuous_frame()
    prepared = prepare_lsm_call_batch(frame, columns=_COLUMNS, dividends=_YIELD)
    executor = CpuExecutor(threads=1)
    method = _method()
    mesh = _mesh()

    adapted = prepared.price(mesh=mesh, executor=executor, method=method)
    direct = price_lsm_call_batch(
        initial_spots=prepared.initial_spots,
        strikes=prepared.strikes,
        maturities=prepared.maturities,
        rates=prepared.rates,
        volatilities=prepared.volatilities,
        dividends=prepared.dividends,
        mesh=mesh,
        executor=executor,
        method=method,
    )

    for column in adapted.__dataclass_fields__:
        assert np.array_equal(getattr(adapted, column), getattr(direct, column))


def test_grouped_frame_pricing_matches_direct_native_grouped_batch_bitwise() -> None:
    """The two-axis adapter changes no group, strike, random, or result identity."""
    prepared = prepare_lsm_strike_groups(
        _group_state_frame(),
        _group_strike_frame(),
        group_offsets=_group_offsets(),
        state_columns=_GROUP_COLUMNS,
        strike_column="strike",
        dividends=_YIELD,
    )
    executor = CpuExecutor(threads=1)
    method = _method()
    mesh = _mesh()

    adapted = prepared.price(mesh=mesh, executor=executor, method=method)
    direct = price_lsm_call_grouped_batch(
        groups=prepared.native,
        mesh=mesh,
        executor=executor,
        method=method,
    )

    assert prepared.group_count == _GROUPS
    assert prepared.rows == _GROUP_ROWS
    np.testing.assert_array_equal(
        prepared.native.group_offsets,
        np.array([0, 2, 3], dtype=np.uint32),
    )
    np.testing.assert_array_equal(prepared.native.initial_spots, np.array([120.0, 100.0]))
    np.testing.assert_array_equal(prepared.native.strikes, np.array([110.0, 125.0, 100.0]))
    for column in adapted.__dataclass_fields__:
        assert np.array_equal(getattr(adapted, column), getattr(direct, column))


def test_grouped_frame_requires_exact_explicit_uint32_boundaries() -> None:
    """The adapter rejects casts, missing boundaries, and empty groups instead of repairing them."""
    with pytest.raises(PolarsSchemaError, match=r"group_offsets.*UInt32.*UInt64"):
        prepare_lsm_strike_groups(
            _group_state_frame(),
            _group_strike_frame(),
            group_offsets=pl.Series([0, 2, 3], dtype=pl.UInt64),
            state_columns=_GROUP_COLUMNS,
            strike_column="strike",
            dividends=_YIELD,
        )
    with pytest.raises(PolarsContentError, match="group_offset_count_mismatch"):
        prepare_lsm_strike_groups(
            _group_state_frame(),
            _group_strike_frame(),
            group_offsets=pl.Series([0, 3], dtype=pl.UInt32),
            state_columns=_GROUP_COLUMNS,
            strike_column="strike",
            dividends=_YIELD,
        )
    with pytest.raises(PolarsContentError, match="empty_or_reversed_groups"):
        prepare_lsm_strike_groups(
            _group_state_frame(),
            _group_strike_frame(),
            group_offsets=pl.Series([0, 0, 3], dtype=pl.UInt32),
            state_columns=_GROUP_COLUMNS,
            strike_column="strike",
            dividends=_YIELD,
        )


def test_grouped_offsets_require_explicit_rechunk_policy() -> None:
    """Fragmented compact boundaries cannot allocate unless the caller permits consolidation."""
    offsets = pl.concat(
        (
            pl.Series("group_offsets", [0, 2], dtype=pl.UInt32),
            pl.Series("group_offsets", [3], dtype=pl.UInt32),
        ),
        rechunk=False,
    )
    with pytest.raises(PolarsBufferError, match=r"group_offsets.*explicit rechunk"):
        prepare_lsm_strike_groups(
            _group_state_frame(),
            _group_strike_frame(),
            group_offsets=offsets,
            state_columns=_GROUP_COLUMNS,
            strike_column="strike",
            dividends=_YIELD,
        )
    prepared = prepare_lsm_strike_groups(
        _group_state_frame(),
        _group_strike_frame(),
        group_offsets=offsets,
        state_columns=_GROUP_COLUMNS,
        strike_column="strike",
        dividends=_YIELD,
        buffer_policy=BufferPolicy.ALLOW_RECHUNK,
    )

    np.testing.assert_array_equal(
        prepared.native.group_offsets,
        np.array([0, 2, 3], dtype=np.uint32),
    )


def test_prepared_grouped_request_retains_borrowed_polars_storage() -> None:
    """Deleting caller frames cannot invalidate arrays retained by the prepared native facade."""
    states = _group_state_frame()
    strikes = _group_strike_frame()
    offsets = _group_offsets()
    prepared = prepare_lsm_strike_groups(
        states,
        strikes,
        group_offsets=offsets,
        state_columns=_GROUP_COLUMNS,
        strike_column="strike",
        dividends=_YIELD,
    )

    del states, strikes, offsets
    gc.collect()

    np.testing.assert_array_equal(prepared.native.initial_spots, np.array([120.0, 100.0]))
    np.testing.assert_array_equal(prepared.native.strikes, np.array([110.0, 125.0, 100.0]))


def test_convenience_pricing_returns_every_row_aligned_diagnostic() -> None:
    """The concise frame API returns every native diagnostic in stable row order."""
    result = price_lsm_call_frame(
        _continuous_frame(),
        columns=_COLUMNS,
        dividends=_YIELD,
        mesh=_mesh(),
        executor=CpuExecutor(threads=1),
        method=_method(),
    )

    assert result.shape == (2, 12)
    assert result.columns == [
        "prices",
        "sample_standard_deviations",
        "standard_errors",
        "origin_decisions_fitted",
        "origin_exercised",
        "regression_nodes",
        "no_training_support_nodes",
        "insufficient_exercise_support_nodes",
        "minimum_in_the_money_training_paths",
        "maximum_regression_condition_numbers",
        "early_exercise_paths",
        "expiry_paths",
    ]


def test_grouped_convenience_pricing_returns_strike_aligned_diagnostics() -> None:
    """The grouped convenience API emits one complete result row per declared strike row."""
    result = price_lsm_strike_group_frames(
        _group_state_frame(),
        _group_strike_frame(),
        group_offsets=_group_offsets(),
        state_columns=_GROUP_COLUMNS,
        strike_column="strike",
        dividends=_YIELD,
        mesh=_mesh(),
        executor=CpuExecutor(threads=1),
        method=_method(),
    )

    assert result.shape == (_GROUP_ROWS, 12)


def test_cash_lists_preserve_empty_rows_offsets_and_event_order() -> None:
    """Exact list schedules retain empty rows, offsets, times, amounts, and order."""
    frame = _continuous_frame().with_columns(
        cash_times=pl.Series([[0.25, 0.75], []], dtype=pl.List(pl.Float64)),
        cash_amounts=pl.Series([[1.0, 1.5], []], dtype=pl.List(pl.Float64)),
    )
    columns = CashDividendColumns(
        times="cash_times",
        amounts="cash_amounts",
        policy=DividendPolicy.LIQUIDATOR,
        maturity_boundary=MaturityBoundary.STRICTLY_BEFORE,
    )

    prepared = prepare_lsm_call_batch(frame, columns=_COLUMNS, dividends=columns)

    assert type(prepared.dividends) is CashDividendBatch
    np.testing.assert_array_equal(
        prepared.dividends.schedules.offsets,
        np.array([0, 2, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        prepared.dividends.schedules.times,
        np.array([0.25, 0.75], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        prepared.dividends.schedules.amounts,
        np.array([1.0, 1.5], dtype=np.float64),
    )


def test_fragmented_storage_requires_explicit_rechunk_policy() -> None:
    """Only the caller-selected policy may consolidate fragmented Polars storage."""
    source = _continuous_frame()
    fragmented = pl.concat((source.slice(0, 1), source.slice(1)), rechunk=False)

    with pytest.raises(PolarsBufferError, match="explicit rechunk"):
        prepare_lsm_call_batch(fragmented, columns=_COLUMNS, dividends=_YIELD)

    prepared = prepare_lsm_call_batch(
        fragmented,
        columns=_COLUMNS,
        dividends=_YIELD,
        buffer_policy=BufferPolicy.ALLOW_RECHUNK,
    )
    assert prepared.rows == _ROWS


def test_null_scalar_fails_before_native_execution() -> None:
    """Nullable financial columns fail at the adapter instead of becoming NaN values."""
    invalid = _continuous_frame().with_columns(
        pl.when(pl.col("row_id") == _FIRST_ROW_ID)
        .then(None)
        .otherwise(pl.col("rate"))
        .alias("rate")
    )

    with pytest.raises(PolarsContentError, match="null_rate"):
        prepare_lsm_call_batch(invalid, columns=_COLUMNS, dividends=_YIELD)


def test_result_frame_rejects_duck_typed_results() -> None:
    """Result conversion admits only the exact immutable native result type."""
    with pytest.raises(PolarsAdapterTypeError, match="LsmBatchResult"):
        lsm_result_frame(cast("LsmBatchResult", object()))
