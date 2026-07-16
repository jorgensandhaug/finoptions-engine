"""Explicit Polars materialization adapters for stateful native pricing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from .batch_lsm import (
    BatchDividends,
    CashDividendBatch,
    ContinuousDividendYieldBatch,
    EscrowedCashDividendBatch,
    EscrowedVolatilityConvention,
    LsmBatchResult,
    price_lsm_call_batch,
)
from .dividends import MaturityBoundary, RaggedCashDividends
from .model import DividendPolicy

if TYPE_CHECKING:
    from .batch_lsm import ExerciseMesh
    from .lsm import Lsm
    from .simulation import CpuExecutor

Float64Vector = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]
PolarsDataType = type[pl.DataType] | pl.DataType

_FLOAT64: Final = np.dtype(np.float64)
_RESULT_COLUMNS: Final = (
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
)


class BufferPolicy(StrEnum):
    """Explicit handling of fragmented Polars primitive and list buffers."""

    REQUIRE_ZERO_COPY = "require_zero_copy"
    ALLOW_RECHUNK = "allow_rechunk"


@dataclass(slots=True)
class PolarsAdapterTypeError(TypeError):
    """A Polars boundary received an unsupported runtime object."""

    field: str
    observed: str
    expected: str

    def __str__(self) -> str:
        """Describe the rejected runtime contract without coercing it."""
        return f"{self.field} must be {self.expected}; observed {self.observed}"


@dataclass(slots=True)
class PolarsSchemaError(ValueError):
    """A materialized or lazy frame does not have the exact pricing schema."""

    missing: tuple[str, ...]
    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe missing and incorrectly typed columns without exposing values."""
        return (
            "invalid Polars pricing schema: "
            f"missing={self.missing!r}, mismatches={self.mismatches!r}"
        )


@dataclass(slots=True)
class PolarsBufferError(ValueError):
    """A Polars column cannot satisfy the declared zero-copy or rechunk policy."""

    column: str
    policy: BufferPolicy
    reason: str

    def __str__(self) -> str:
        """Describe the failed storage contract without exposing values."""
        return f"column {self.column!r} violates {self.policy.value}: {self.reason}"


@dataclass(slots=True)
class PolarsContentError(ValueError):
    """Complete frame content cannot form one native pricing request."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing financial rows."""
        return f"invalid Polars pricing content: {self.violations!r}"


def _require_column_name(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise PolarsAdapterTypeError(
            field=field,
            observed=repr(value),
            expected="a nonempty plain string",
        )
    return cast("str", value)


@dataclass(frozen=True, slots=True)
class LsmCallColumns:
    """Exact source-column names for one row-addressed LSM call batch."""

    initial_spot: str
    strike: str
    maturity: str
    rate: str
    volatility: str

    def __post_init__(self) -> None:
        """Require five distinct, explicit, nonempty source columns."""
        names = tuple(
            _require_column_name(value, field=field)
            for field, value in (
                ("initial_spot", self.initial_spot),
                ("strike", self.strike),
                ("maturity", self.maturity),
                ("rate", self.rate),
                ("volatility", self.volatility),
            )
        )
        if len(set(names)) != len(names):
            raise PolarsAdapterTypeError(
                field="LSM scalar columns",
                observed=repr(names),
                expected="five distinct column names",
            )

    def ordered(self) -> tuple[str, str, str, str, str]:
        """Return the stable native argument order."""
        return (self.initial_spot, self.strike, self.maturity, self.rate, self.volatility)


@dataclass(frozen=True, slots=True)
class ContinuousYieldColumn:
    """One Float64 continuously compounded annual dividend-yield column."""

    annualized_decimals: str

    def __post_init__(self) -> None:
        """Require an explicit source column."""
        _require_column_name(self.annualized_decimals, field="annualized_decimals")


@dataclass(frozen=True, slots=True)
class CashDividendColumns:
    """Paired List(Float64) cash-event columns with explicit jump semantics."""

    times: str
    amounts: str
    policy: DividendPolicy
    maturity_boundary: MaturityBoundary

    def __post_init__(self) -> None:
        """Require distinct event columns and exact financial conventions."""
        times = _require_column_name(self.times, field="times")
        amounts = _require_column_name(self.amounts, field="amounts")
        if times == amounts:
            raise PolarsAdapterTypeError(
                field="cash-dividend columns",
                observed=repr((times, amounts)),
                expected="two distinct column names",
            )
        if type(self.policy) is not DividendPolicy:
            raise PolarsAdapterTypeError(
                field="policy",
                observed=type(self.policy).__name__,
                expected="DividendPolicy",
            )
        if type(self.maturity_boundary) is not MaturityBoundary:
            raise PolarsAdapterTypeError(
                field="maturity_boundary",
                observed=type(self.maturity_boundary).__name__,
                expected="MaturityBoundary",
            )


@dataclass(frozen=True, slots=True)
class EscrowedCashDividendColumns:
    """Paired cash-event columns for the declared escrowed approximation."""

    times: str
    amounts: str
    volatility_convention: EscrowedVolatilityConvention
    maturity_boundary: MaturityBoundary

    def __post_init__(self) -> None:
        """Require distinct event columns and exact approximation conventions."""
        times = _require_column_name(self.times, field="times")
        amounts = _require_column_name(self.amounts, field="amounts")
        if times == amounts:
            raise PolarsAdapterTypeError(
                field="escrowed cash-dividend columns",
                observed=repr((times, amounts)),
                expected="two distinct column names",
            )
        if type(self.volatility_convention) is not EscrowedVolatilityConvention:
            raise PolarsAdapterTypeError(
                field="volatility_convention",
                observed=type(self.volatility_convention).__name__,
                expected="EscrowedVolatilityConvention",
            )
        if type(self.maturity_boundary) is not MaturityBoundary:
            raise PolarsAdapterTypeError(
                field="maturity_boundary",
                observed=type(self.maturity_boundary).__name__,
                expected="MaturityBoundary",
            )


type DividendColumns = ContinuousYieldColumn | CashDividendColumns | EscrowedCashDividendColumns


@dataclass(frozen=True, slots=True, eq=False)
class PreparedLsmCallBatch:
    """Complete contiguous native request borrowed from one materialized Polars frame."""

    initial_spots: Float64Vector
    strikes: Float64Vector
    maturities: Float64Vector
    rates: Float64Vector
    volatilities: Float64Vector
    dividends: BatchDividends

    @property
    def rows(self) -> int:
        """Return the preserved source-row count."""
        return int(self.initial_spots.size)

    def price(
        self,
        *,
        mesh: ExerciseMesh,
        executor: CpuExecutor,
        method: Lsm,
    ) -> LsmBatchResult:
        """Execute the native estimator without changing the prepared request."""
        return price_lsm_call_batch(
            initial_spots=self.initial_spots,
            strikes=self.strikes,
            maturities=self.maturities,
            rates=self.rates,
            volatilities=self.volatilities,
            dividends=self.dividends,
            mesh=mesh,
            executor=executor,
            method=method,
        )


def select_lsm_call_inputs(
    frame: pl.LazyFrame,
    *,
    columns: LsmCallColumns,
    dividends: DividendColumns,
) -> pl.LazyFrame:
    """Project exact pricing columns while keeping the query lazy.

    This resolves only the logical schema. It never collects data, casts a value, sorts rows,
    infers groups, or inserts a pricing UDF into the lazy plan.
    """
    _validate_lazy_request(frame, columns, dividends)
    expected = _expected_schema(columns, dividends)
    _validate_schema(frame.collect_schema(), expected)
    return frame.select(*expected)


def prepare_lsm_call_batch(
    frame: pl.DataFrame,
    *,
    columns: LsmCallColumns,
    dividends: DividendColumns,
    buffer_policy: BufferPolicy = BufferPolicy.REQUIRE_ZERO_COPY,
) -> PreparedLsmCallBatch:
    """Borrow one fully materialized frame as a strict row-addressed LSM request.

    A `LazyFrame` is deliberately rejected: callers must make collection and its resource policy
    visible. Float64 scalar columns and event children remain zero-copy unless `ALLOW_RECHUNK`
    explicitly permits consolidation. The adapter never casts, sorts, repairs, or drops a row.
    """
    _validate_eager_request(frame, columns, dividends, buffer_policy)
    if frame.height == 0:
        raise PolarsContentError(violations=(("empty_frame", 1),))
    expected = _expected_schema(columns, dividends)
    _validate_schema(frame.schema, expected)
    scalar_vectors = tuple(
        _float64_vector(frame.get_column(name), column=name, policy=buffer_policy)
        for name in columns.ordered()
    )
    prepared_dividends = _prepare_dividends(frame, dividends, buffer_policy)
    return PreparedLsmCallBatch(
        initial_spots=scalar_vectors[0],
        strikes=scalar_vectors[1],
        maturities=scalar_vectors[2],
        rates=scalar_vectors[3],
        volatilities=scalar_vectors[4],
        dividends=prepared_dividends,
    )


def price_lsm_call_frame(  # noqa: PLR0913 - every execution contract remains explicit
    frame: pl.DataFrame,
    *,
    columns: LsmCallColumns,
    dividends: DividendColumns,
    mesh: ExerciseMesh,
    executor: CpuExecutor,
    method: Lsm,
    buffer_policy: BufferPolicy = BufferPolicy.REQUIRE_ZERO_COPY,
) -> pl.DataFrame:
    """Price one materialized frame and return a row-aligned result table."""
    prepared = prepare_lsm_call_batch(
        frame,
        columns=columns,
        dividends=dividends,
        buffer_policy=buffer_policy,
    )
    return lsm_result_frame(prepared.price(mesh=mesh, executor=executor, method=method))


def lsm_result_frame(result: LsmBatchResult) -> pl.DataFrame:
    """Expose every native diagnostic as an immutable-order Polars result table."""
    if type(result) is not LsmBatchResult:
        raise PolarsAdapterTypeError(
            field="result",
            observed=type(result).__name__,
            expected="LsmBatchResult",
        )
    return pl.DataFrame(
        {column: getattr(result, column) for column in _RESULT_COLUMNS},
    )


def _validate_lazy_request(
    frame: object,
    columns: object,
    dividends: object,
) -> None:
    if type(frame) is not pl.LazyFrame:
        raise PolarsAdapterTypeError(
            field="frame",
            observed=type(frame).__name__,
            expected="polars.LazyFrame",
        )
    _validate_specs(columns, dividends)


def _validate_eager_request(
    frame: object,
    columns: object,
    dividends: object,
    buffer_policy: object,
) -> None:
    if type(frame) is not pl.DataFrame:
        raise PolarsAdapterTypeError(
            field="frame",
            observed=type(frame).__name__,
            expected="a materialized polars.DataFrame",
        )
    _validate_specs(columns, dividends)
    if type(buffer_policy) is not BufferPolicy:
        raise PolarsAdapterTypeError(
            field="buffer_policy",
            observed=type(buffer_policy).__name__,
            expected="BufferPolicy",
        )


def _validate_specs(columns: object, dividends: object) -> None:
    if type(columns) is not LsmCallColumns:
        raise PolarsAdapterTypeError(
            field="columns",
            observed=type(columns).__name__,
            expected="LsmCallColumns",
        )
    if type(dividends) not in (
        ContinuousYieldColumn,
        CashDividendColumns,
        EscrowedCashDividendColumns,
    ):
        raise PolarsAdapterTypeError(
            field="dividends",
            observed=type(dividends).__name__,
            expected=("ContinuousYieldColumn, CashDividendColumns, or EscrowedCashDividendColumns"),
        )
    typed_dividends = cast("DividendColumns", dividends)
    overlap = set(columns.ordered()) & set(_dividend_column_names(typed_dividends))
    if overlap:
        raise PolarsAdapterTypeError(
            field="pricing columns",
            observed=repr(tuple(sorted(overlap))),
            expected="distinct scalar and dividend columns",
        )


def _expected_schema(
    columns: LsmCallColumns,
    dividends: DividendColumns,
) -> dict[str, PolarsDataType]:
    expected: dict[str, PolarsDataType] = dict.fromkeys(columns.ordered(), pl.Float64)
    if isinstance(dividends, ContinuousYieldColumn):
        expected[dividends.annualized_decimals] = pl.Float64
    else:
        expected[dividends.times] = pl.List(pl.Float64)
        expected[dividends.amounts] = pl.List(pl.Float64)
    return expected


def _dividend_column_names(dividends: DividendColumns) -> tuple[str, ...]:
    if isinstance(dividends, ContinuousYieldColumn):
        return (dividends.annualized_decimals,)
    return (dividends.times, dividends.amounts)


def _validate_schema(
    observed: pl.Schema,
    expected: dict[str, PolarsDataType],
) -> None:
    missing = tuple(name for name in expected if name not in observed)
    mismatches = tuple(
        (name, str(dtype), str(observed[name]))
        for name, dtype in expected.items()
        if name in observed and observed[name] != dtype
    )
    if missing or mismatches:
        raise PolarsSchemaError(missing=missing, mismatches=mismatches)


def _float64_vector(
    series: pl.Series,
    *,
    column: str,
    policy: BufferPolicy,
) -> Float64Vector:
    if series.null_count() != 0:
        raise PolarsContentError(violations=((f"null_{column}", series.null_count()),))
    candidate = _single_chunk(series, column=column, policy=policy)
    try:
        converted = candidate.to_numpy(writable=False, allow_copy=False)
    except RuntimeError as error:
        raise PolarsBufferError(
            column=column,
            policy=policy,
            reason="Float64 storage cannot be borrowed by NumPy after the declared policy",
        ) from error
    result = cast("Float64Vector", converted)
    if (
        type(result) is not np.ndarray
        or result.dtype != _FLOAT64
        or result.ndim != 1
        or not result.flags.c_contiguous
    ):
        raise PolarsAdapterTypeError(
            field=column,
            observed=(
                f"{type(result).__name__}, {getattr(result, 'dtype', None)}, "
                f"{getattr(result, 'shape', None)}"
            ),
            expected="a C-contiguous Float64 vector",
        )
    return result


def _single_chunk(series: pl.Series, *, column: str, policy: BufferPolicy) -> pl.Series:
    if series.n_chunks() == 1:
        return series
    if policy is BufferPolicy.REQUIRE_ZERO_COPY:
        raise PolarsBufferError(
            column=column,
            policy=policy,
            reason="fragmented storage requires an explicit rechunk",
        )
    return series.rechunk()


def _prepare_dividends(
    frame: pl.DataFrame,
    dividends: DividendColumns,
    policy: BufferPolicy,
) -> BatchDividends:
    if isinstance(dividends, ContinuousYieldColumn):
        return ContinuousDividendYieldBatch(
            annualized_decimals=_float64_vector(
                frame.get_column(dividends.annualized_decimals),
                column=dividends.annualized_decimals,
                policy=policy,
            ),
        )
    schedules = _ragged_cash_dividends(
        frame,
        times_column=dividends.times,
        amounts_column=dividends.amounts,
        policy=policy,
    )
    if isinstance(dividends, CashDividendColumns):
        return CashDividendBatch(
            schedules=schedules,
            policy=dividends.policy,
            maturity_boundary=dividends.maturity_boundary,
        )
    return EscrowedCashDividendBatch(
        schedules=schedules,
        volatility_convention=dividends.volatility_convention,
        maturity_boundary=dividends.maturity_boundary,
    )


def _ragged_cash_dividends(
    frame: pl.DataFrame,
    *,
    times_column: str,
    amounts_column: str,
    policy: BufferPolicy,
) -> RaggedCashDividends:
    times = _single_chunk(frame.get_column(times_column), column=times_column, policy=policy)
    amounts = _single_chunk(frame.get_column(amounts_column), column=amounts_column, policy=policy)
    violations: list[tuple[str, int]] = []
    if times.null_count():
        violations.append(("null_cash_time_lists", times.null_count()))
    if amounts.null_count():
        violations.append(("null_cash_amount_lists", amounts.null_count()))
    time_lengths = times.list.len()
    amount_lengths = amounts.list.len()
    length_mismatches = cast("int", (time_lengths != amount_lengths).sum() or 0)
    if length_mismatches:
        violations.append(("cash_event_length_mismatches", length_mismatches))
    exploded_times = times.explode(empty_as_null=False, keep_nulls=True)
    exploded_amounts = amounts.explode(empty_as_null=False, keep_nulls=True)
    if exploded_times.null_count():
        violations.append(("null_cash_times", exploded_times.null_count()))
    if exploded_amounts.null_count():
        violations.append(("null_cash_amounts", exploded_amounts.null_count()))
    if violations:
        raise PolarsContentError(violations=tuple(violations))

    lengths = cast(
        "Int64Vector",
        time_lengths.cast(pl.Int64).to_numpy(writable=False, allow_copy=True),
    )
    offsets = np.empty(frame.height + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    event_count = int(offsets[-1])
    time_values = _float64_vector(exploded_times, column=times_column, policy=policy)
    amount_values = _float64_vector(exploded_amounts, column=amounts_column, policy=policy)
    if time_values.size != event_count or amount_values.size != event_count:
        raise PolarsContentError(violations=(("cash_event_count_mismatch", 1),))
    return RaggedCashDividends(offsets=offsets, times=time_values, amounts=amount_values)


__all__ = [
    "BufferPolicy",
    "CashDividendColumns",
    "ContinuousYieldColumn",
    "DividendColumns",
    "EscrowedCashDividendColumns",
    "LsmCallColumns",
    "PolarsAdapterTypeError",
    "PolarsBufferError",
    "PolarsContentError",
    "PolarsSchemaError",
    "PreparedLsmCallBatch",
    "lsm_result_frame",
    "prepare_lsm_call_batch",
    "price_lsm_call_frame",
    "select_lsm_call_inputs",
]
