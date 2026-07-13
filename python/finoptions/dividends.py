"""Exact ragged cash dividends and explicit consumer-specific model tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from . import _native
from .model import InputError
from .simulation import CpuExecutor

Int64Vector = npt.NDArray[np.int64]
UInt32Vector = npt.NDArray[np.uint32]
Float64Vector = npt.NDArray[np.float64]
Float32Matrix = npt.NDArray[np.float32]
BoolMatrix = npt.NDArray[np.bool_]

BatchError = _native.BatchError
_MAX_U32: Final = (2**32) - 1
_VECTOR_DIMENSIONS: Final = 1
_MATRIX_DIMENSIONS: Final = 2


class MaturityBoundary(StrEnum):
    """Whether an event exactly at option maturity belongs to a model input."""

    STRICTLY_BEFORE = "strictly_before"
    AT_OR_BEFORE = "at_or_before"


class TimeBinClosure(StrEnum):
    """Exact interval closure for chronological fixed-width time bins."""

    LEFT_CLOSED_RIGHT_OPEN = "left_closed_right_open"
    LEFT_OPEN_RIGHT_CLOSED = "left_open_right_closed"


class CashAmountScale(StrEnum):
    """Explicit units for cash-dividend model features."""

    ABSOLUTE = "absolute"
    DIVIDE_BY_SPOT = "divide_by_spot"


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
        msg = f"{field} must be C-contiguous"
        raise InputError(msg)


def _validate_matrix(value: object, *, field: str, dtype: str) -> None:
    if type(value) is not np.ndarray:
        msg = f"{field} must be a plain numpy.ndarray"
        raise InputError(msg)
    array = cast("npt.NDArray[np.generic]", value)
    expected = np.dtype(dtype)
    if array.dtype != expected:
        msg = f"{field} must have dtype {expected}; observed {array.dtype}"
        raise InputError(msg)
    if array.ndim != _MATRIX_DIMENSIONS:
        msg = f"{field} must be two-dimensional; observed shape {array.shape}"
        raise InputError(msg)
    if not array.flags.c_contiguous:
        msg = f"{field} must be C-contiguous"
        raise InputError(msg)


@dataclass(frozen=True, slots=True)
class RaggedCashDividends:
    """Borrowable Arrow-LargeList-style offsets, times, and exact cash amounts.

    Offsets may start above zero when the child arrays are the corresponding slice from a sliced
    Arrow list. Arrays are borrowed only for each native call and revalidated because NumPy buffers
    remain mutable.
    """

    offsets: Int64Vector
    times: Float64Vector
    amounts: Float64Vector

    def __post_init__(self) -> None:
        """Fail immediately on malformed layout or invalid financial event values."""
        _validate_vector(self.offsets, field="offsets", dtype="int64")
        _validate_vector(self.times, field="times", dtype="float64")
        _validate_vector(self.amounts, field="amounts", dtype="float64")
        _native.validate_ragged_cash_dividends_f64(self.offsets, self.times, self.amounts)

    @property
    def rows(self) -> int:
        """Return the number of option rows."""
        return int(self.offsets.size - 1)

    @property
    def events(self) -> int:
        """Return the total event count in the child arrays."""
        return int(self.times.size)


@dataclass(frozen=True, slots=True)
class PaddedSequenceSpec:
    """Identity of one chronological right-padded dividend sequence layout."""

    capacity: int
    maturity_boundary: MaturityBoundary
    amount_scale: CashAmountScale

    def __post_init__(self) -> None:
        """Reject inferred capacity, boundary, or amount units."""
        if type(self.capacity) is not int or not 1 <= self.capacity <= _MAX_U32:
            msg = f"capacity must be a plain int in [1, {_MAX_U32}]; observed {self.capacity!r}"
            raise InputError(msg)
        if type(self.maturity_boundary) is not MaturityBoundary:
            msg = "maturity_boundary must be MaturityBoundary"
            raise InputError(msg)
        if type(self.amount_scale) is not CashAmountScale:
            msg = "amount_scale must be CashAmountScale"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class DividendBinSpec:
    """Identity of one chronological fixed-width dividend-bin layout."""

    bins: int
    horizon: float
    closure: TimeBinClosure
    maturity_boundary: MaturityBoundary
    amount_scale: CashAmountScale

    def __post_init__(self) -> None:
        """Reject inferred bin count, horizon, closure, boundary, or amount units."""
        if type(self.bins) is not int or not 1 <= self.bins <= _MAX_U32:
            msg = f"bins must be a plain int in [1, {_MAX_U32}]; observed {self.bins!r}"
            raise InputError(msg)
        if (
            type(self.horizon) is not float
            or not math.isfinite(self.horizon)
            or self.horizon <= 0.0
        ):
            msg = f"horizon must be a finite positive plain float; observed {self.horizon!r}"
            raise InputError(msg)
        if type(self.closure) is not TimeBinClosure:
            msg = "closure must be TimeBinClosure"
            raise InputError(msg)
        if type(self.maturity_boundary) is not MaturityBoundary:
            msg = "maturity_boundary must be MaturityBoundary"
            raise InputError(msg)
        if type(self.amount_scale) is not CashAmountScale:
            msg = "amount_scale must be CashAmountScale"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class PaddedCashDividends:
    """Owned chronological float32 sequences, explicit masks, and exact lengths."""

    times: Float32Matrix
    amounts: Float32Matrix
    mask: BoolMatrix
    lengths: UInt32Vector

    def __post_init__(self) -> None:
        """Reject manually assembled results outside the native output contract."""
        _validate_matrix(self.times, field="times", dtype="float32")
        _validate_matrix(self.amounts, field="amounts", dtype="float32")
        _validate_matrix(self.mask, field="mask", dtype="bool")
        _validate_vector(self.lengths, field="lengths", dtype="uint32")
        if self.amounts.shape != self.times.shape or self.mask.shape != self.times.shape:
            msg = "padded times, amounts, and mask must have identical shapes"
            raise InputError(msg)
        if self.lengths.shape != (self.times.shape[0],):
            msg = "padded lengths must contain exactly one value per row"
            raise InputError(msg)

    @property
    def rows(self) -> int:
        """Return the number of option rows."""
        return int(self.times.shape[0])

    @property
    def capacity(self) -> int:
        """Return the exact slots allocated per row."""
        return int(self.times.shape[1])


@dataclass(frozen=True, slots=True)
class BinnedCashDividends:
    """Owned chronological row-major float32 cash-amount bins."""

    amounts: Float32Matrix

    def __post_init__(self) -> None:
        """Reject manually assembled results outside the native output contract."""
        _validate_matrix(self.amounts, field="amounts", dtype="float32")

    @property
    def rows(self) -> int:
        """Return the number of option rows."""
        return int(self.amounts.shape[0])

    @property
    def bins(self) -> int:
        """Return the chronological bin count."""
        return int(self.amounts.shape[1])


def _validate_tensor_request(
    *,
    dividends: RaggedCashDividends,
    maturities: Float64Vector,
    amount_scale: CashAmountScale,
    spots: Float64Vector | None,
    executor: CpuExecutor,
) -> None:
    if type(dividends) is not RaggedCashDividends:
        msg = f"dividends must be RaggedCashDividends; observed {type(dividends).__name__}"
        raise InputError(msg)
    _validate_vector(maturities, field="maturities", dtype="float64")
    if type(executor) is not CpuExecutor:
        msg = f"executor must be CpuExecutor; observed {type(executor).__name__}"
        raise InputError(msg)
    if amount_scale is CashAmountScale.ABSOLUTE:
        if spots is not None:
            msg = "spots must be None when amount_scale is ABSOLUTE"
            raise InputError(msg)
    elif amount_scale is CashAmountScale.DIVIDE_BY_SPOT:
        if spots is None:
            msg = "spots are required when amount_scale is DIVIDE_BY_SPOT"
            raise InputError(msg)
        _validate_vector(spots, field="spots", dtype="float64")
    else:
        msg = "amount_scale must be CashAmountScale"
        raise InputError(msg)


def pack_padded_cash_dividends(
    *,
    dividends: RaggedCashDividends,
    maturities: Float64Vector,
    spec: PaddedSequenceSpec,
    executor: CpuExecutor,
    spots: Float64Vector | None = None,
) -> PaddedCashDividends:
    """Build one explicit chronological padded representation without source copies.

    The native kernel checks source values, maturity semantics, capacity, spot units, shape, and
    every float64-to-float32 narrowing. It never changes capacity, precision, scale, or execution
    policy and always emits a validity mask plus exact lengths.
    """
    if type(spec) is not PaddedSequenceSpec:
        msg = f"spec must be PaddedSequenceSpec; observed {type(spec).__name__}"
        raise InputError(msg)
    _validate_tensor_request(
        dividends=dividends,
        maturities=maturities,
        amount_scale=spec.amount_scale,
        spots=spots,
        executor=executor,
    )
    times, amounts, mask, lengths = _native.pack_padded_dividends_f32(
        dividends.offsets,
        dividends.times,
        dividends.amounts,
        maturities,
        spec.amount_scale.value,
        spec.capacity,
        spec.maturity_boundary.value,
        executor._handle,  # noqa: SLF001 - facade owns the private native resource
        spots,
    )
    return PaddedCashDividends(times=times, amounts=amounts, mask=mask, lengths=lengths)


def bin_cash_dividends(
    *,
    dividends: RaggedCashDividends,
    maturities: Float64Vector,
    spec: DividendBinSpec,
    executor: CpuExecutor,
    spots: Float64Vector | None = None,
) -> BinnedCashDividends:
    """Build one explicit chronological fixed-width bin representation without source copies.

    Bin closure, maturity inclusion, horizon, units, float32 precision, and executor are all part of
    the declared request. No alternate convention or kernel is selected on failure.
    """
    if type(spec) is not DividendBinSpec:
        msg = f"spec must be DividendBinSpec; observed {type(spec).__name__}"
        raise InputError(msg)
    _validate_tensor_request(
        dividends=dividends,
        maturities=maturities,
        amount_scale=spec.amount_scale,
        spots=spots,
        executor=executor,
    )
    amounts = _native.bin_dividends_f32(
        dividends.offsets,
        dividends.times,
        dividends.amounts,
        maturities,
        spec.amount_scale.value,
        spec.bins,
        spec.horizon,
        spec.closure.value,
        spec.maturity_boundary.value,
        executor._handle,  # noqa: SLF001 - facade owns the private native resource
        spots,
    )
    return BinnedCashDividends(amounts=amounts)


__all__ = [
    "BatchError",
    "BinnedCashDividends",
    "CashAmountScale",
    "DividendBinSpec",
    "MaturityBoundary",
    "PaddedCashDividends",
    "PaddedSequenceSpec",
    "RaggedCashDividends",
    "TimeBinClosure",
    "bin_cash_dividends",
    "pack_padded_cash_dividends",
]
