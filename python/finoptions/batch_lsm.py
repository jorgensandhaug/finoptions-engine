"""Contract-parallel LSM calls over strict contiguous financial columns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from . import _native
from .dividends import MaturityBoundary, RaggedCashDividends
from .lsm import Lsm, LsmError
from .model import DividendPolicy, InputError
from .simulation import CpuExecutor

Float64Vector = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]
UInt32Vector = npt.NDArray[np.uint32]
BoolVector = npt.NDArray[np.bool_]

_MAX_U32: Final = (2**32) - 1
_MIN_EVALUATION_PATHS: Final = 2
_VECTOR_DIMENSIONS: Final = 1


def _validate_vector(value: object, *, field: str, dtype: str) -> None:
    if type(value) is not np.ndarray:
        msg = f"{field} must be a plain numpy.ndarray"
        raise InputError(msg)
    array = cast("npt.NDArray[np.generic]", value)
    expected = np.dtype(dtype)
    if array.dtype != expected:
        msg = f"{field} must have dtype {expected}; observed {array.dtype}"
        raise InputError(msg)
    if array.ndim != _VECTOR_DIMENSIONS:
        msg = f"{field} must be one-dimensional; observed shape {array.shape}"
        raise InputError(msg)
    if not array.flags.c_contiguous:
        msg = f"{field} must be C-contiguous; implicit copies are forbidden"
        raise InputError(msg)


class OriginExercise(StrEnum):
    """Whether valuation origin is an explicit common exercise decision."""

    EXCLUDED = "excluded"
    INCLUDED = "included"


class EscrowedVolatilityConvention(StrEnum):
    """Meaning of the volatility vector for the escrowed prepaid-forward approximation."""

    PREPAID_FORWARD = "prepaid_forward"
    SPOT_MATCHED_AT_ORIGIN = "spot_matched_at_origin"


@dataclass(frozen=True, slots=True)
class UniformExerciseMesh:
    """Uniform Bermudan base mesh augmented by every exact cash-event time."""

    base_intervals: int
    origin_exercise: OriginExercise

    def __post_init__(self) -> None:
        """Reject empty, inferred, or unrepresentable meshes and implicit origin semantics."""
        if type(self.base_intervals) is not int or not 1 <= self.base_intervals <= _MAX_U32:
            msg = (
                f"base_intervals must be a plain int in [1, {_MAX_U32}]; "
                f"observed {self.base_intervals!r}"
            )
            raise InputError(msg)
        if type(self.origin_exercise) is not OriginExercise:
            msg = "origin_exercise must be OriginExercise"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class CashDividendExerciseMesh:
    """Exercise only at left limits of strictly pre-expiry deterministic cash events.

    Native pricing requires a cash-dividend model, nonnegative rates, and no cash event at expiry.
    Any row outside that complete theorem domain fails; it is not moved to a uniform mesh.
    """


@dataclass(frozen=True, slots=True)
class RaggedExerciseTimes:
    """Borrowable row-aligned Arrow-LargeList-style exact exercise times."""

    offsets: Int64Vector
    times: Float64Vector

    def __post_init__(self) -> None:
        """Reject malformed, invalid, duplicate, or unordered opportunities without copying."""
        _validate_vector(self.offsets, field="exercise offsets", dtype="int64")
        _validate_vector(self.times, field="exercise times", dtype="float64")
        _native.validate_ragged_exercise_times_f64(self.offsets, self.times)

    @property
    def rows(self) -> int:
        """Return the exact schedule-row count."""
        return int(self.offsets.size - 1)

    @property
    def events(self) -> int:
        """Return the total explicit opportunity count."""
        return int(self.times.size)


@dataclass(frozen=True, slots=True)
class ExplicitExerciseMesh:
    """Exact row-aligned pre-expiry Bermudan opportunities.

    Cash-event times become transition nodes but are exercisable only when also listed here.
    Expiry is always the final stopping opportunity. Calendar mapping belongs to the caller; this
    type never sorts, snaps, deduplicates, or infers a time.
    """

    schedules: RaggedExerciseTimes
    origin_exercise: OriginExercise

    def __post_init__(self) -> None:
        """Reject duck-typed schedules and implicit origin semantics."""
        if type(self.schedules) is not RaggedExerciseTimes:
            msg = "schedules must be RaggedExerciseTimes"
            raise InputError(msg)
        if type(self.origin_exercise) is not OriginExercise:
            msg = "origin_exercise must be OriginExercise"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class ContinuousDividendYieldBatch:
    """One constant nonnegative annualized proportional yield per option row."""

    annualized_decimals: Float64Vector

    def __post_init__(self) -> None:
        """Require one directly borrowable float64 vector; native pricing validates values."""
        _validate_vector(
            self.annualized_decimals,
            field="annualized_decimals",
            dtype="float64",
        )


@dataclass(frozen=True, slots=True)
class CashDividendBatch:
    """Exact cash schedules with explicit jump and terminal-boundary conventions."""

    schedules: RaggedCashDividends
    policy: DividendPolicy
    maturity_boundary: MaturityBoundary

    def __post_init__(self) -> None:
        """Reject duck-typed schedules and implicit cash-jump conventions."""
        if type(self.schedules) is not RaggedCashDividends:
            msg = "schedules must be RaggedCashDividends"
            raise InputError(msg)
        if type(self.policy) is not DividendPolicy:
            msg = "policy must be DividendPolicy"
            raise InputError(msg)
        if type(self.maturity_boundary) is not MaturityBoundary:
            msg = "maturity_boundary must be MaturityBoundary"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class EscrowedCashDividendBatch:
    """Risky prepaid forward plus the deterministic value of later cash dividends.

    This is an escrowed-dividend approximation, not the exact spot cash-jump model represented by
    :class:`CashDividendBatch`. ``volatility_convention`` explicitly states whether the supplied
    volatility vector belongs to the risky prepaid forward already or contains observed spot
    volatilities that must be rescaled to match initial dollar volatility.
    """

    schedules: RaggedCashDividends
    volatility_convention: EscrowedVolatilityConvention
    maturity_boundary: MaturityBoundary

    def __post_init__(self) -> None:
        """Reject duck-typed schedules and any implicit volatility interpretation."""
        if type(self.schedules) is not RaggedCashDividends:
            msg = "schedules must be RaggedCashDividends"
            raise InputError(msg)
        if type(self.volatility_convention) is not EscrowedVolatilityConvention:
            msg = "volatility_convention must be EscrowedVolatilityConvention"
            raise InputError(msg)
        if type(self.maturity_boundary) is not MaturityBoundary:
            msg = "maturity_boundary must be MaturityBoundary"
            raise InputError(msg)


type BatchDividends = ContinuousDividendYieldBatch | CashDividendBatch | EscrowedCashDividendBatch
type ExerciseMesh = UniformExerciseMesh | CashDividendExerciseMesh | ExplicitExerciseMesh


def _native_mesh_arguments(
    mesh: ExerciseMesh,
) -> tuple[str, int | None, str | None, Int64Vector | None, Float64Vector | None]:
    if type(mesh) is UniformExerciseMesh:
        return ("uniform", mesh.base_intervals, mesh.origin_exercise.value, None, None)
    if type(mesh) is CashDividendExerciseMesh:
        return ("cash_dividend_dates", None, None, None, None)
    if type(mesh) is ExplicitExerciseMesh:
        return (
            "explicit_times",
            None,
            mesh.origin_exercise.value,
            mesh.schedules.offsets,
            mesh.schedules.times,
        )
    msg = "exercise mesh changed type after request validation"
    raise LsmError(msg)


def _validate_mesh_dividends(*, mesh: ExerciseMesh, dividends: BatchDividends) -> None:
    if type(mesh) is CashDividendExerciseMesh and type(dividends) is ContinuousDividendYieldBatch:
        msg = "CashDividendExerciseMesh requires cash-dividend inputs"
        raise InputError(msg)


def _validate_mesh_rows(*, mesh: ExerciseMesh, expected: int) -> None:
    if type(mesh) is ExplicitExerciseMesh and mesh.schedules.rows != expected:
        msg = f"explicit exercise schedules have {mesh.schedules.rows} rows; expected {expected}"
        raise InputError(msg)


@dataclass(frozen=True, slots=True)
class LsmStrikeGroups:
    """Compact exact simulation states with one or more row-ordered strikes per state.

    ``group_offsets`` are contiguous uint32 strike-row boundaries. Spot, maturity, rate,
    volatility, and dividends have one entry or schedule per simulation group; ``strikes`` has one
    entry per output contract. Constructing groups is an explicit research-data operation: this
    type never infers equality, sorts contracts, or collapses rows.
    """

    group_offsets: UInt32Vector
    initial_spots: Float64Vector
    strikes: Float64Vector
    maturities: Float64Vector
    rates: Float64Vector
    volatilities: Float64Vector
    dividends: BatchDividends

    def __post_init__(self) -> None:
        """Require directly borrowable compact columns; native pricing revalidates all values."""
        _validate_vector(self.group_offsets, field="group_offsets", dtype="uint32")
        for field, values in (
            ("initial_spots", self.initial_spots),
            ("strikes", self.strikes),
            ("maturities", self.maturities),
            ("rates", self.rates),
            ("volatilities", self.volatilities),
        ):
            _validate_vector(values, field=field, dtype="float64")
        if type(self.dividends) not in (
            ContinuousDividendYieldBatch,
            CashDividendBatch,
            EscrowedCashDividendBatch,
        ):
            msg = (
                "dividends must be ContinuousDividendYieldBatch, CashDividendBatch, "
                "or EscrowedCashDividendBatch"
            )
            raise InputError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class LsmBatchResult:
    """Immutable row-ordered prices, sampling error, policy support, and stopping counts."""

    prices: Float64Vector
    sample_standard_deviations: Float64Vector
    standard_errors: Float64Vector
    origin_decisions_fitted: BoolVector
    origin_exercised: BoolVector
    regression_nodes: UInt32Vector
    no_training_support_nodes: UInt32Vector
    insufficient_exercise_support_nodes: UInt32Vector
    minimum_in_the_money_training_paths: UInt32Vector
    maximum_regression_condition_numbers: Float64Vector
    early_exercise_paths: UInt32Vector
    expiry_paths: UInt32Vector

    def __post_init__(self) -> None:
        """Reject manually assembled outputs outside the native immutable column contract."""
        fields = (
            ("prices", self.prices, "float64"),
            ("sample_standard_deviations", self.sample_standard_deviations, "float64"),
            ("standard_errors", self.standard_errors, "float64"),
            ("origin_decisions_fitted", self.origin_decisions_fitted, "bool"),
            ("origin_exercised", self.origin_exercised, "bool"),
            ("regression_nodes", self.regression_nodes, "uint32"),
            ("no_training_support_nodes", self.no_training_support_nodes, "uint32"),
            (
                "insufficient_exercise_support_nodes",
                self.insufficient_exercise_support_nodes,
                "uint32",
            ),
            (
                "minimum_in_the_money_training_paths",
                self.minimum_in_the_money_training_paths,
                "uint32",
            ),
            (
                "maximum_regression_condition_numbers",
                self.maximum_regression_condition_numbers,
                "float64",
            ),
            ("early_exercise_paths", self.early_exercise_paths, "uint32"),
            ("expiry_paths", self.expiry_paths, "uint32"),
        )
        rows = int(self.prices.size)
        for field, values, dtype in fields:
            _validate_vector(values, field=field, dtype=dtype)
            if values.size != rows:
                msg = f"{field} has {values.size} rows; expected {rows}"
                raise InputError(msg)
            if values.flags.writeable:
                msg = f"{field} must be immutable"
                raise InputError(msg)

    @property
    def rows(self) -> int:
        """Return the exact priced contract count."""
        return int(self.prices.size)


def _validate_request(  # noqa: PLR0913 - each financial column is independently typed
    *,
    initial_spots: Float64Vector,
    strikes: Float64Vector,
    maturities: Float64Vector,
    rates: Float64Vector,
    volatilities: Float64Vector,
    dividends: BatchDividends,
    mesh: ExerciseMesh,
    executor: CpuExecutor,
    method: Lsm,
) -> None:
    for field, values in (
        ("initial_spots", initial_spots),
        ("strikes", strikes),
        ("maturities", maturities),
        ("rates", rates),
        ("volatilities", volatilities),
    ):
        _validate_vector(values, field=field, dtype="float64")
    if type(dividends) not in (
        ContinuousDividendYieldBatch,
        CashDividendBatch,
        EscrowedCashDividendBatch,
    ):
        msg = (
            "dividends must be ContinuousDividendYieldBatch, CashDividendBatch, "
            "or EscrowedCashDividendBatch"
        )
        raise InputError(msg)
    if type(mesh) not in (
        UniformExerciseMesh,
        CashDividendExerciseMesh,
        ExplicitExerciseMesh,
    ):
        msg = "mesh must be UniformExerciseMesh, CashDividendExerciseMesh, or ExplicitExerciseMesh"
        raise InputError(msg)
    _validate_mesh_rows(mesh=mesh, expected=int(initial_spots.size))
    _validate_mesh_dividends(mesh=mesh, dividends=dividends)
    if type(executor) is not CpuExecutor:
        msg = "executor must be CpuExecutor"
        raise InputError(msg)
    if type(method) is not Lsm:
        msg = "method must be Lsm"
        raise InputError(msg)
    if method.evaluation_paths < _MIN_EVALUATION_PATHS:
        msg = "batched LSM requires at least two independent evaluation paths"
        raise InputError(msg)


def _freeze_native_result(raw: _native.NativeBatchLsmResult) -> LsmBatchResult:
    (
        prices,
        sample_standard_deviations,
        standard_errors,
        origin_decisions_fitted,
        origin_exercised,
        regression_nodes,
        no_training_support_nodes,
        insufficient_exercise_support_nodes,
        minimum_in_the_money_training_paths,
        maximum_regression_condition_numbers,
        early_exercise_paths,
        expiry_paths,
    ) = raw
    prices.flags.writeable = False
    sample_standard_deviations.flags.writeable = False
    standard_errors.flags.writeable = False
    origin_decisions_fitted.flags.writeable = False
    origin_exercised.flags.writeable = False
    regression_nodes.flags.writeable = False
    no_training_support_nodes.flags.writeable = False
    insufficient_exercise_support_nodes.flags.writeable = False
    minimum_in_the_money_training_paths.flags.writeable = False
    maximum_regression_condition_numbers.flags.writeable = False
    early_exercise_paths.flags.writeable = False
    expiry_paths.flags.writeable = False
    return LsmBatchResult(
        prices=prices,
        sample_standard_deviations=sample_standard_deviations,
        standard_errors=standard_errors,
        origin_decisions_fitted=origin_decisions_fitted,
        origin_exercised=origin_exercised,
        regression_nodes=regression_nodes,
        no_training_support_nodes=no_training_support_nodes,
        insufficient_exercise_support_nodes=insufficient_exercise_support_nodes,
        minimum_in_the_money_training_paths=minimum_in_the_money_training_paths,
        maximum_regression_condition_numbers=maximum_regression_condition_numbers,
        early_exercise_paths=early_exercise_paths,
        expiry_paths=expiry_paths,
    )


def price_lsm_call_batch(  # noqa: PLR0913 - each financial column is independently typed
    *,
    initial_spots: Float64Vector,
    strikes: Float64Vector,
    maturities: Float64Vector,
    rates: Float64Vector,
    volatilities: Float64Vector,
    dividends: BatchDividends,
    mesh: ExerciseMesh,
    executor: CpuExecutor,
    method: Lsm,
) -> LsmBatchResult:
    """Price constant-parameter Bermudan calls using native contract-parallel LSM.

    `rates` are constant continuously compounded annual decimals. `volatilities` are positive
    annual decimals. A uniform mesh is augmented by every exact cash-event time. The specialized
    cash mesh admits only strictly pre-expiry cash-event left limits under nonnegative rates. An
    explicit mesh uses only caller-listed pre-expiry opportunities and cash-event transition nodes.
    Each row uses `method.training_stream.domain + row` and
    `method.evaluation_stream.domain + row`; the native boundary rejects overflow or overlap.

    Input arrays are borrowed without dtype, layout, or ownership conversion. Do not mutate them
    concurrently with this synchronous call. Policy fitting and valuation use disjoint path
    populations; no failed row returns a partial result or changes the requested method.

    Raises:
        InputError: If a runtime type, array layout, model value, mesh, or random interval is
            invalid.
        LsmError: If allocation, simulation, regression, policy support, or evaluation fails.

    """
    _validate_request(
        initial_spots=initial_spots,
        strikes=strikes,
        maturities=maturities,
        rates=rates,
        volatilities=volatilities,
        dividends=dividends,
        mesh=mesh,
        executor=executor,
        method=method,
    )
    common = (
        initial_spots,
        strikes,
        maturities,
        rates,
        volatilities,
    )
    method_arguments = (
        *_native_mesh_arguments(mesh),
        executor._handle,  # noqa: SLF001 - facade owns the private native resource
        method.training_paths,
        method.evaluation_paths,
        method.training_stream.seed,
        method.training_stream.domain,
        method.evaluation_stream.seed,
        method.evaluation_stream.domain,
        method.polynomial_degree,
        method.rank_tolerance,
        method.condition_limit,
        method.tie_rule.value,
        method.insufficient_exercise_support.value,
    )
    if type(dividends) is ContinuousDividendYieldBatch:
        raw = _native.price_lsm_call_batch_continuous_yield_f64(
            *common,
            dividends.annualized_decimals,
            *method_arguments,
        )
    elif type(dividends) is CashDividendBatch:
        raw = _native.price_lsm_call_batch_cash_dividends_f64(
            *common,
            dividends.schedules.offsets,
            dividends.schedules.times,
            dividends.schedules.amounts,
            dividends.policy.value,
            dividends.maturity_boundary.value,
            *method_arguments,
        )
    elif type(dividends) is EscrowedCashDividendBatch:
        raw = _native.price_lsm_call_batch_escrowed_cash_dividends_f64(
            *common,
            dividends.schedules.offsets,
            dividends.schedules.times,
            dividends.schedules.amounts,
            dividends.volatility_convention.value,
            dividends.maturity_boundary.value,
            *method_arguments,
        )
    else:
        msg = "dividends changed type after request validation"
        raise LsmError(msg)
    result = _freeze_native_result(raw)
    stopping_paths = result.early_exercise_paths.astype(np.uint64) + result.expiry_paths.astype(
        np.uint64
    )
    if np.any(stopping_paths != method.evaluation_paths):
        msg = "native batched LSM stopping counts do not partition evaluation paths"
        raise LsmError(msg)
    return result


def price_lsm_call_grouped_batch(
    *,
    groups: LsmStrikeGroups,
    mesh: ExerciseMesh,
    executor: CpuExecutor,
    method: Lsm,
) -> LsmBatchResult:
    """Price strike groups while simulating each declared market state exactly once.

    Every strike fits its own stopping policy on the shared group training population and is valued
    out of sample on the shared, disjoint evaluation population. Sharing random paths across
    strikes is intentional common-random-number design; it does not share a stopping policy.

    Group columns, exact exercise schedules, and result order are borrowed and preserved exactly.
    The function never infers groups or calendar times, casts arrays, sorts strikes, repairs
    offsets, changes regression degree, or falls back after a financial, support, simulation, or
    numerical failure.

    Raises:
        InputError: If a runtime type, array layout, compact boundary, financial value, mesh, or
            group-addressed random interval is invalid.
        LsmError: If allocation, simulation, regression, policy support, or evaluation fails.

    """
    if type(groups) is not LsmStrikeGroups:
        msg = "groups must be LsmStrikeGroups"
        raise InputError(msg)
    if type(mesh) not in (
        UniformExerciseMesh,
        CashDividendExerciseMesh,
        ExplicitExerciseMesh,
    ):
        msg = "mesh must be UniformExerciseMesh, CashDividendExerciseMesh, or ExplicitExerciseMesh"
        raise InputError(msg)
    if type(executor) is not CpuExecutor:
        msg = "executor must be CpuExecutor"
        raise InputError(msg)
    if type(method) is not Lsm:
        msg = "method must be Lsm"
        raise InputError(msg)
    if method.evaluation_paths < _MIN_EVALUATION_PATHS:
        msg = "grouped batched LSM requires at least two independent evaluation paths"
        raise InputError(msg)
    _validate_mesh_rows(mesh=mesh, expected=int(groups.initial_spots.size))
    _validate_mesh_dividends(mesh=mesh, dividends=groups.dividends)

    common = (
        groups.group_offsets,
        groups.initial_spots,
        groups.strikes,
        groups.maturities,
        groups.rates,
        groups.volatilities,
    )
    method_arguments = (
        *_native_mesh_arguments(mesh),
        executor._handle,  # noqa: SLF001 - facade owns the private native resource
        method.training_paths,
        method.evaluation_paths,
        method.training_stream.seed,
        method.training_stream.domain,
        method.evaluation_stream.seed,
        method.evaluation_stream.domain,
        method.polynomial_degree,
        method.rank_tolerance,
        method.condition_limit,
        method.tie_rule.value,
        method.insufficient_exercise_support.value,
    )
    dividends = groups.dividends
    if type(dividends) is ContinuousDividendYieldBatch:
        raw = _native.price_lsm_call_grouped_batch_continuous_yield_f64(
            *common,
            dividends.annualized_decimals,
            *method_arguments,
        )
    elif type(dividends) is CashDividendBatch:
        raw = _native.price_lsm_call_grouped_batch_cash_dividends_f64(
            *common,
            dividends.schedules.offsets,
            dividends.schedules.times,
            dividends.schedules.amounts,
            dividends.policy.value,
            dividends.maturity_boundary.value,
            *method_arguments,
        )
    elif type(dividends) is EscrowedCashDividendBatch:
        raw = _native.price_lsm_call_grouped_batch_escrowed_cash_dividends_f64(
            *common,
            dividends.schedules.offsets,
            dividends.schedules.times,
            dividends.schedules.amounts,
            dividends.volatility_convention.value,
            dividends.maturity_boundary.value,
            *method_arguments,
        )
    else:
        msg = "group dividends changed type after request validation"
        raise LsmError(msg)
    result = _freeze_native_result(raw)
    stopping_paths = result.early_exercise_paths.astype(np.uint64) + result.expiry_paths.astype(
        np.uint64
    )
    if np.any(stopping_paths != method.evaluation_paths):
        msg = "native grouped LSM stopping counts do not partition evaluation paths"
        raise LsmError(msg)
    return result


__all__ = [
    "BatchDividends",
    "CashDividendBatch",
    "CashDividendExerciseMesh",
    "ContinuousDividendYieldBatch",
    "EscrowedCashDividendBatch",
    "EscrowedVolatilityConvention",
    "ExerciseMesh",
    "ExplicitExerciseMesh",
    "LsmBatchResult",
    "LsmStrikeGroups",
    "OriginExercise",
    "RaggedExerciseTimes",
    "UniformExerciseMesh",
    "price_lsm_call_batch",
    "price_lsm_call_grouped_batch",
]
