"""Fast, reproducible stationary bootstrap of additive panel statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from . import _native
from .model import InputError
from .simulation import CpuExecutor, RandomStream

Float64Matrix = npt.NDArray[np.float64]
UInt64Vector = npt.NDArray[np.uint64]
BootstrapError = _native.BootstrapError
_MATRIX_DIMENSIONS: Final = 2
_MAX_U32: Final = (2**32) - 1
_MIN_STRATUM_OFFSETS: Final = 2


@dataclass(frozen=True, slots=True)
class StationaryBootstrap:
    """Circular geometric-block bootstrap with an explicit Philox namespace.

    Replication ``b`` uses the path lane ``replication_path_start + b`` in ``random_stream``.
    Circular origins use unbiased 64-bit multiply-high rejection. Block lengths use a fixed
    open-unit 52-bit midpoint inverse CDF, so output is invariant to executor width on one native
    numerical build.
    """

    replications: int
    mean_block_length: float
    random_stream: RandomStream
    replication_path_start: int = 0

    def __post_init__(self) -> None:
        """Reject incomplete resources or wrapping random namespaces before native work."""
        if type(self.replications) is not int or not 1 <= self.replications <= _MAX_U32:
            msg = (
                f"replications must be a plain int in [1, {_MAX_U32}]; "
                f"observed {self.replications!r}"
            )
            raise InputError(msg)
        if (
            type(self.mean_block_length) is not float
            or not math.isfinite(self.mean_block_length)
            or self.mean_block_length < 1.0
        ):
            msg = (
                "mean_block_length must be a finite plain float of at least one; "
                f"observed {self.mean_block_length!r}"
            )
            raise InputError(msg)
        if type(self.random_stream) is not RandomStream:
            msg = (
                f"random_stream must be RandomStream; observed {type(self.random_stream).__name__}"
            )
            raise InputError(msg)
        if (
            type(self.replication_path_start) is not int
            or not 0 <= self.replication_path_start <= _MAX_U32
            or self.replication_path_start + self.replications - 1 > _MAX_U32
        ):
            msg = (
                "replication_path_start must be a plain unsigned 32-bit start whose complete "
                f"replication range fits; observed {self.replication_path_start!r}"
            )
            raise InputError(msg)

    def sum_panels(
        self,
        panel_statistics: Float64Matrix,
        *,
        executor: CpuExecutor,
    ) -> Float64Matrix:
        """Return one row of resampled additive sums per replication.

        ``panel_statistics`` is borrowed without conversion and must be a nonempty C-contiguous
        Float64 matrix shaped ``[ordered_panels, additive_statistics]``. Each output row sums
        exactly the observed number of panel positions. The native prefix kernel evaluates one
        circular block in time proportional to the statistic count rather than its length.

        Raises:
            InputError: If types, shape, layout, values, or random resources are invalid.
            BootstrapError: If finite input overflows a prefix or resampled sum, or allocation
                fails.

        """
        _validate_panel_statistics(panel_statistics)
        if type(executor) is not CpuExecutor:
            msg = f"executor must be CpuExecutor; observed {type(executor).__name__}"
            raise InputError(msg)
        result = _native.stationary_bootstrap_panel_sums_f64(
            panel_statistics,
            executor._handle,  # noqa: SLF001 - wrapper owns the private native resource
            self.replications,
            self.mean_block_length,
            self.random_stream.seed,
            self.random_stream.domain,
            self.replication_path_start,
        )
        result.flags.writeable = False
        return result

    def sum_stratified_panels(
        self,
        panel_statistics: Float64Matrix,
        stratum_offsets: UInt64Vector,
        *,
        executor: CpuExecutor,
    ) -> Float64Matrix:
        """Return additive sums while preserving every contiguous stratum's panel count.

        ``stratum_offsets`` must be an immutable-by-contract C-contiguous UInt64 vector of the
        form ``[0, ..., ordered_panels]`` with strictly increasing values. Circular block origins
        wrap only inside each declared stratum; every output therefore contains exactly the source
        number of positions from every stratum. No dtype or layout conversion is implicit.

        Raises:
            InputError: If matrix, offset, executor, or random-resource contracts are invalid.
            BootstrapError: If finite input overflows a prefix or resampled sum, or allocation
                fails.

        """
        _validate_panel_statistics(panel_statistics)
        _validate_stratum_offsets(stratum_offsets)
        if type(executor) is not CpuExecutor:
            msg = f"executor must be CpuExecutor; observed {type(executor).__name__}"
            raise InputError(msg)
        result = _native.stationary_bootstrap_stratified_panel_sums_f64(
            panel_statistics,
            stratum_offsets,
            executor._handle,  # noqa: SLF001 - wrapper owns the private native resource
            self.replications,
            self.mean_block_length,
            self.random_stream.seed,
            self.random_stream.domain,
            self.replication_path_start,
        )
        result.flags.writeable = False
        return result


def _validate_panel_statistics(values: object) -> None:
    if type(values) is not np.ndarray:
        msg = f"panel_statistics must be a plain numpy.ndarray; observed {type(values).__name__}"
        raise InputError(msg)
    array = cast("Float64Matrix", values)
    if array.dtype != np.dtype(np.float64):
        msg = f"panel_statistics must have dtype float64; observed {array.dtype}"
        raise InputError(msg)
    if array.ndim != _MATRIX_DIMENSIONS or 0 in array.shape:
        msg = (
            "panel_statistics must have nonempty shape [ordered_panels, additive_statistics]; "
            f"observed {array.shape!r}"
        )
        raise InputError(msg)
    if not array.flags.c_contiguous:
        msg = "panel_statistics must be C-contiguous; implicit copies are forbidden"
        raise InputError(msg)


def _validate_stratum_offsets(values: object) -> None:
    if type(values) is not np.ndarray:
        msg = f"stratum_offsets must be a plain numpy.ndarray; observed {type(values).__name__}"
        raise InputError(msg)
    array = cast("UInt64Vector", values)
    if array.dtype != np.dtype(np.uint64):
        msg = f"stratum_offsets must have dtype uint64; observed {array.dtype}"
        raise InputError(msg)
    if array.ndim != 1 or array.size < _MIN_STRATUM_OFFSETS:
        msg = f"stratum_offsets must be a one-dimensional partition; observed {array.shape!r}"
        raise InputError(msg)
    if not array.flags.c_contiguous:
        msg = "stratum_offsets must be C-contiguous; implicit copies are forbidden"
        raise InputError(msg)


__all__ = ["BootstrapError", "StationaryBootstrap"]
