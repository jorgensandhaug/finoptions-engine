"""Polars lazy projection, explicit materialization, and native-pricing boundary tests."""

from __future__ import annotations

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
)
from finoptions.polars import (
    BufferPolicy,
    CashDividendColumns,
    ContinuousYieldColumn,
    LsmCallColumns,
    PolarsAdapterTypeError,
    PolarsBufferError,
    PolarsContentError,
    PolarsSchemaError,
    lsm_result_frame,
    prepare_lsm_call_batch,
    price_lsm_call_frame,
    select_lsm_call_inputs,
)

_COLUMNS = LsmCallColumns(
    initial_spot="spot",
    strike="strike",
    maturity="maturity",
    rate="rate",
    volatility="volatility",
)
_YIELD = ContinuousYieldColumn("yield")
_ROWS = 2
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
