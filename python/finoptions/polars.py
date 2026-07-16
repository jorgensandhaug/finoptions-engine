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
    LsmStrikeGroups,
    price_lsm_call_batch,
    price_lsm_call_grouped_batch,
)
from .dividends import MaturityBoundary, RaggedCashDividends
from .model import DividendPolicy

if TYPE_CHECKING:
    from .batch_lsm import ExerciseMesh
    from .lsm import Lsm
    from .simulation import CpuExecutor

Float64Vector = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]
UInt32Vector = npt.NDArray[np.uint32]
PolarsDataType = type[pl.DataType] | pl.DataType

_FLOAT64: Final = np.dtype(np.float64)
_UINT32: Final = np.dtype(np.uint32)
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
class LsmStrikeGroupColumns:
    """Exact state-column names for explicitly bounded strike groups."""

    initial_spot: str
    maturity: str
    rate: str
    volatility: str

    def __post_init__(self) -> None:
        """Require four distinct, explicit, nonempty state columns."""
        names = tuple(
            _require_column_name(value, field=field)
            for field, value in (
                ("initial_spot", self.initial_spot),
                ("maturity", self.maturity),
                ("rate", self.rate),
                ("volatility", self.volatility),
            )
        )
        if len(set(names)) != len(names):
            raise PolarsAdapterTypeError(
                field="LSM strike-group state columns",
                observed=repr(names),
                expected="four distinct column names",
            )

    def ordered(self) -> tuple[str, str, str, str]:
        """Return the stable native state argument order."""
        return (self.initial_spot, self.maturity, self.rate, self.volatility)


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


@dataclass(frozen=True, slots=True, eq=False)
class PreparedLsmStrikeGroups:
    """Complete compact grouped request borrowed from two materialized Polars axes."""

    native: LsmStrikeGroups

    def __post_init__(self) -> None:
        """Reject construction around anything except the exact native facade."""
        if type(self.native) is not LsmStrikeGroups:
            raise PolarsAdapterTypeError(
                field="native",
                observed=type(self.native).__name__,
                expected="LsmStrikeGroups",
            )

    @property
    def group_count(self) -> int:
        """Return the independently simulated state count."""
        return int(self.native.initial_spots.size)

    @property
    def rows(self) -> int:
        """Return the preserved strike-row count."""
        return int(self.native.strikes.size)

    def price(
        self,
        *,
        mesh: ExerciseMesh,
        executor: CpuExecutor,
        method: Lsm,
    ) -> LsmBatchResult:
        """Execute the grouped native estimator without changing either source axis."""
        return price_lsm_call_grouped_batch(
            groups=self.native,
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


def select_lsm_strike_group_inputs(
    state_frame: pl.LazyFrame,
    strike_frame: pl.LazyFrame,
    *,
    state_columns: LsmStrikeGroupColumns,
    strike_column: str,
    dividends: DividendColumns,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Project the two exact grouped-pricing axes while keeping both queries lazy.

    The caller remains responsible for constructing explicit group offsets and collecting both
    axes. This function never joins, groups, sorts, casts, collects, or infers state equality.
    """
    _validate_group_lazy_request(
        state_frame,
        strike_frame,
        state_columns,
        strike_column,
        dividends,
    )
    state_schema = _expected_group_state_schema(state_columns, dividends)
    strike_schema: dict[str, PolarsDataType] = {
        _require_column_name(strike_column, field="strike_column"): pl.Float64
    }
    _validate_schema(state_frame.collect_schema(), state_schema)
    _validate_schema(strike_frame.collect_schema(), strike_schema)
    return state_frame.select(*state_schema), strike_frame.select(*strike_schema)


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


def prepare_lsm_strike_groups(  # noqa: PLR0913 - every compact axis is explicit
    state_frame: pl.DataFrame,
    strike_frame: pl.DataFrame,
    *,
    group_offsets: pl.Series,
    state_columns: LsmStrikeGroupColumns,
    strike_column: str,
    dividends: DividendColumns,
    buffer_policy: BufferPolicy = BufferPolicy.REQUIRE_ZERO_COPY,
) -> PreparedLsmStrikeGroups:
    """Borrow explicit compact state, strike, and offset axes as one grouped LSM request.

    `state_frame` has one row per independently simulated market state. `strike_frame` has one row
    per output contract. `group_offsets` are exact contiguous boundaries into the strike axis and
    must contain one more value than the state-row count. No equality, grouping, sorting, casting,
    repair, or row removal is performed.
    """
    strike_name = _validate_group_eager_request(
        state_frame,
        strike_frame,
        group_offsets,
        state_columns,
        strike_column,
        dividends,
        buffer_policy,
    )
    violations: list[tuple[str, int]] = []
    if state_frame.height == 0:
        violations.append(("empty_state_frame", 1))
    if strike_frame.height == 0:
        violations.append(("empty_strike_frame", 1))
    if violations:
        raise PolarsContentError(violations=tuple(violations))

    state_schema = _expected_group_state_schema(state_columns, dividends)
    strike_schema: dict[str, PolarsDataType] = {strike_name: pl.Float64}
    _validate_schema(state_frame.schema, state_schema)
    _validate_schema(strike_frame.schema, strike_schema)
    offsets = _uint32_vector(group_offsets, column="group_offsets", policy=buffer_policy)
    _validate_group_offsets(
        offsets,
        expected_groups=state_frame.height,
        expected_rows=strike_frame.height,
    )
    state_vectors = tuple(
        _float64_vector(state_frame.get_column(name), column=name, policy=buffer_policy)
        for name in state_columns.ordered()
    )
    strikes = _float64_vector(
        strike_frame.get_column(strike_name),
        column=strike_name,
        policy=buffer_policy,
    )
    prepared_dividends = _prepare_dividends(state_frame, dividends, buffer_policy)
    return PreparedLsmStrikeGroups(
        native=LsmStrikeGroups(
            group_offsets=offsets,
            initial_spots=state_vectors[0],
            strikes=strikes,
            maturities=state_vectors[1],
            rates=state_vectors[2],
            volatilities=state_vectors[3],
            dividends=prepared_dividends,
        )
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


def price_lsm_strike_group_frames(  # noqa: PLR0913 - every axis and convention is explicit
    state_frame: pl.DataFrame,
    strike_frame: pl.DataFrame,
    *,
    group_offsets: pl.Series,
    state_columns: LsmStrikeGroupColumns,
    strike_column: str,
    dividends: DividendColumns,
    mesh: ExerciseMesh,
    executor: CpuExecutor,
    method: Lsm,
    buffer_policy: BufferPolicy = BufferPolicy.REQUIRE_ZERO_COPY,
) -> pl.DataFrame:
    """Price explicit compact strike groups and return row-aligned native diagnostics."""
    prepared = prepare_lsm_strike_groups(
        state_frame,
        strike_frame,
        group_offsets=group_offsets,
        state_columns=state_columns,
        strike_column=strike_column,
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


def _validate_group_lazy_request(
    state_frame: object,
    strike_frame: object,
    state_columns: object,
    strike_column: object,
    dividends: object,
) -> None:
    if type(state_frame) is not pl.LazyFrame:
        raise PolarsAdapterTypeError(
            field="state_frame",
            observed=type(state_frame).__name__,
            expected="polars.LazyFrame",
        )
    if type(strike_frame) is not pl.LazyFrame:
        raise PolarsAdapterTypeError(
            field="strike_frame",
            observed=type(strike_frame).__name__,
            expected="polars.LazyFrame",
        )
    _validate_group_specs(state_columns, strike_column, dividends)


def _validate_group_eager_request(  # noqa: PLR0913 - validates one explicit public contract
    state_frame: object,
    strike_frame: object,
    group_offsets: object,
    state_columns: object,
    strike_column: object,
    dividends: object,
    buffer_policy: object,
) -> str:
    if type(state_frame) is not pl.DataFrame:
        raise PolarsAdapterTypeError(
            field="state_frame",
            observed=type(state_frame).__name__,
            expected="a materialized polars.DataFrame",
        )
    if type(strike_frame) is not pl.DataFrame:
        raise PolarsAdapterTypeError(
            field="strike_frame",
            observed=type(strike_frame).__name__,
            expected="a materialized polars.DataFrame",
        )
    if type(group_offsets) is not pl.Series:
        raise PolarsAdapterTypeError(
            field="group_offsets",
            observed=type(group_offsets).__name__,
            expected="a materialized polars.Series",
        )
    strike_name = _validate_group_specs(state_columns, strike_column, dividends)
    if type(buffer_policy) is not BufferPolicy:
        raise PolarsAdapterTypeError(
            field="buffer_policy",
            observed=type(buffer_policy).__name__,
            expected="BufferPolicy",
        )
    return strike_name


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


def _validate_group_specs(
    state_columns: object,
    strike_column: object,
    dividends: object,
) -> str:
    if type(state_columns) is not LsmStrikeGroupColumns:
        raise PolarsAdapterTypeError(
            field="state_columns",
            observed=type(state_columns).__name__,
            expected="LsmStrikeGroupColumns",
        )
    strike_name = _require_column_name(strike_column, field="strike_column")
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
    typed_columns = state_columns
    typed_dividends = cast("DividendColumns", dividends)
    overlap = set(typed_columns.ordered()) & set(_dividend_column_names(typed_dividends))
    if overlap:
        raise PolarsAdapterTypeError(
            field="group-state columns",
            observed=repr(tuple(sorted(overlap))),
            expected="distinct state and dividend columns",
        )
    return strike_name


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


def _expected_group_state_schema(
    columns: LsmStrikeGroupColumns,
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


def _uint32_vector(
    series: pl.Series,
    *,
    column: str,
    policy: BufferPolicy,
) -> UInt32Vector:
    if series.dtype != pl.UInt32:
        raise PolarsSchemaError(
            missing=(),
            mismatches=((column, str(pl.UInt32), str(series.dtype)),),
        )
    if series.null_count() != 0:
        raise PolarsContentError(violations=((f"null_{column}", series.null_count()),))
    candidate = _single_chunk(series, column=column, policy=policy)
    try:
        converted = candidate.to_numpy(writable=False, allow_copy=False)
    except RuntimeError as error:
        raise PolarsBufferError(
            column=column,
            policy=policy,
            reason="UInt32 storage cannot be borrowed by NumPy after the declared policy",
        ) from error
    result = cast("UInt32Vector", converted)
    if (
        type(result) is not np.ndarray
        or result.dtype != _UINT32
        or result.ndim != 1
        or not result.flags.c_contiguous
    ):
        raise PolarsAdapterTypeError(
            field=column,
            observed=(
                f"{type(result).__name__}, {getattr(result, 'dtype', None)}, "
                f"{getattr(result, 'shape', None)}"
            ),
            expected="a C-contiguous UInt32 vector",
        )
    return result


def _validate_group_offsets(
    offsets: UInt32Vector,
    *,
    expected_groups: int,
    expected_rows: int,
) -> None:
    violations: list[tuple[str, int]] = []
    if offsets.size != expected_groups + 1:
        violations.append(("group_offset_count_mismatch", 1))
    if offsets.size == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != expected_rows:
        violations.append(("group_offset_boundary_mismatch", 1))
    if offsets.size > 1:
        invalid_boundaries = int(np.count_nonzero(offsets[1:] <= offsets[:-1]))
        if invalid_boundaries:
            violations.append(("empty_or_reversed_groups", invalid_boundaries))
    if violations:
        raise PolarsContentError(violations=tuple(violations))


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
    "LsmStrikeGroupColumns",
    "PolarsAdapterTypeError",
    "PolarsBufferError",
    "PolarsContentError",
    "PolarsSchemaError",
    "PreparedLsmCallBatch",
    "PreparedLsmStrikeGroups",
    "lsm_result_frame",
    "prepare_lsm_call_batch",
    "prepare_lsm_strike_groups",
    "price_lsm_call_frame",
    "price_lsm_strike_group_frames",
    "select_lsm_call_inputs",
    "select_lsm_strike_group_inputs",
]
