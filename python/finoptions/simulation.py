"""Strict, zero-copy-input Python surface for native path simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Final

import numpy as np
import numpy.typing as npt

from . import _native
from .model import (
    ConstantVolatility,
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EventSchedule,
    InputError,
    LogDiscountCurve,
)

Float64Matrix = npt.NDArray[np.float64]
ExecutionError = _native.ExecutionError
SimulationError = _native.SimulationError
_MATRIX_DIMENSIONS: Final = 2
_MAX_SEED: Final = (2**64) - 1
_MAX_DOMAIN: Final = (2**32) - 1
_MAX_PATHS: Final = (2**32) - 1


class CpuMathBackend(StrEnum):
    """Named floating-point implementation for vectorizable CPU pricing kernels."""

    RUST_STD_F64 = "rust_std_f64"
    APPLE_ACCELERATE = "apple_accelerate"


@dataclass(frozen=True, slots=True)
class CpuExecutor:
    """Reusable CPU resource with exact worker and numerical backend choices."""

    threads: int
    math_backend: CpuMathBackend = CpuMathBackend.RUST_STD_F64
    _handle: _native.CpuExecutor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject inferred/empty thread counts and construct the native pool once."""
        if type(self.threads) is not int or self.threads <= 0:
            msg = f"threads must be a positive plain int; observed {self.threads!r}"
            raise InputError(msg)
        if type(self.math_backend) is not CpuMathBackend:
            msg = (
                f"math_backend must be CpuMathBackend; observed {type(self.math_backend).__name__}"
            )
            raise InputError(msg)
        object.__setattr__(
            self,
            "_handle",
            _native.CpuExecutor(self.threads, self.math_backend.value),
        )


@dataclass(frozen=True, slots=True)
class RandomStream:
    """Explicit reproducible Philox namespace for one statistically distinct path set.

    Use different `domain` values with the same seed for roles that must not reuse paths, such as
    LSM policy fitting and independent policy evaluation.
    """

    seed: int
    domain: int

    def __post_init__(self) -> None:
        """Reject inferred signs, booleans, and values outside the native counter layout."""
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            msg = f"seed must be a plain int in [0, {_MAX_SEED}]; observed {self.seed!r}"
            raise InputError(msg)
        if type(self.domain) is not int or not 0 <= self.domain <= _MAX_DOMAIN:
            msg = f"domain must be a plain int in [0, {_MAX_DOMAIN}]; observed {self.domain!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class GbmPaths:
    """Owned time-major pre-event states at exercise nodes and expiry."""

    observation_nodes: tuple[int, ...]
    states: Float64Matrix

    def __post_init__(self) -> None:
        """Reject manually constructed results that violate the native output contract."""
        if (
            type(self.observation_nodes) is not tuple
            or not self.observation_nodes
            or any(
                type(node) is not int or node < 0 or node > (2**32 - 1)
                for node in self.observation_nodes
            )
        ):
            msg = "observation_nodes must be an immutable tuple[int, ...]"
            raise InputError(msg)
        if any(right <= left for left, right in pairwise(self.observation_nodes)):
            msg = "observation_nodes must be strictly increasing"
            raise InputError(msg)
        if type(self.states) is not np.ndarray:
            msg = "states must be a plain numpy.ndarray"
            raise InputError(msg)
        if self.states.dtype != np.dtype(np.float64):
            msg = f"states must have dtype float64; observed {self.states.dtype}"
            raise InputError(msg)
        if self.states.ndim != _MATRIX_DIMENSIONS or self.states.shape[0] != len(
            self.observation_nodes
        ):
            msg = (
                f"states must have shape [observation_nodes, paths]; observed {self.states.shape!r}"
            )
            raise InputError(msg)
        if self.states.shape[1] == 0:
            msg = "states must contain at least one path"
            raise InputError(msg)
        if not self.states.flags.c_contiguous:
            msg = "states must be C-contiguous"
            raise InputError(msg)

    @property
    def path_count(self) -> int:
        """Return the number of simulated paths."""
        return self.states.shape[1]

    def values_at(self, node: int) -> npt.NDArray[np.float64]:
        """Return the zero-copy path row for an observed node.

        Raises:
            InputError: If `node` is not a plain integer.
            KeyError: If the node is not an exercise or expiry observation.

        """
        if type(node) is not int:
            msg = f"observation node must be a plain int; observed {type(node).__name__}"
            raise InputError(msg)
        try:
            row = self.observation_nodes.index(node)
        except ValueError as error:
            raise KeyError(node) from error
        return self.states[row]


def _validate_simulation_specs(  # noqa: PLR0913 - financial inputs remain independently typed
    schedule: EventSchedule,
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    executor: CpuExecutor,
) -> None:
    if type(schedule) is not EventSchedule:
        msg = f"schedule must be EventSchedule; observed {type(schedule).__name__}"
        raise InputError(msg)
    if type(curve) is not LogDiscountCurve:
        msg = f"curve must be LogDiscountCurve; observed {type(curve).__name__}"
        raise InputError(msg)
    if type(dividend_yield_curve) is not CumulativeDividendYieldCurve:
        msg = (
            "dividend_yield_curve must be CumulativeDividendYieldCurve; "
            f"observed {type(dividend_yield_curve).__name__}"
        )
        raise InputError(msg)
    if type(volatility) is not ConstantVolatility:
        msg = f"volatility must be ConstantVolatility; observed {type(volatility).__name__}"
        raise InputError(msg)
    if type(dividend_policy) is not DividendPolicy:
        msg = f"dividend_policy must be DividendPolicy; observed {type(dividend_policy).__name__}"
        raise InputError(msg)
    if type(executor) is not CpuExecutor:
        msg = f"executor must be CpuExecutor; observed {type(executor).__name__}"
        raise InputError(msg)


def _validate_normal_matrix(normals: Float64Matrix) -> None:
    if type(normals) is not np.ndarray:
        msg = f"normals must be a plain numpy.ndarray; observed {type(normals).__name__}"
        raise InputError(msg)
    if normals.dtype != np.dtype(np.float64):
        msg = f"normals must have dtype float64; observed {normals.dtype}"
        raise InputError(msg)
    if normals.ndim != _MATRIX_DIMENSIONS:
        msg = f"normals must have shape [intervals, paths]; observed {normals.shape!r}"
        raise InputError(msg)
    if not normals.flags.c_contiguous:
        msg = "normals must be C-contiguous; implicit copies are forbidden"
        raise InputError(msg)


def _validate_initial_spot(initial_spot: float) -> None:
    if type(initial_spot) is not float:
        msg = f"initial_spot must be a plain float; observed {type(initial_spot).__name__}"
        raise InputError(msg)


def _paths_from_native(schedule: EventSchedule, states: Float64Matrix) -> GbmPaths:
    states.flags.writeable = False
    return GbmPaths(
        observation_nodes=(*schedule.exercise_nodes, schedule.expiry_node),
        states=states,
    )


def simulate_gbm(  # noqa: PLR0913 - each explicit financial input is independently typed
    *,
    initial_spot: float,
    schedule: EventSchedule,
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    executor: CpuExecutor,
    normals: Float64Matrix,
) -> GbmPaths:
    """Simulate exact adjacent-node GBM paths from explicit standard-normal variates.

    `normals` must be a C-contiguous `float64` matrix with shape `[intervals, paths]`. It is
    borrowed without casting or copying. The native kernel validates every variate before path
    computation and returns only pre-event exercise states plus terminal expiry.

    Raises:
        InputError: If a runtime type, grid, shape, value, or financial input is invalid.
        SimulationError: If a mathematically valid input produces a non-finite path transition.

    """
    _validate_initial_spot(initial_spot)
    _validate_simulation_specs(
        schedule,
        curve,
        dividend_yield_curve,
        volatility,
        dividend_policy,
        executor,
    )
    _validate_normal_matrix(normals)

    states = _native.simulate_gbm_f64(
        initial_spot,
        list(schedule.times),
        list(curve.times),
        list(curve.log_discounts),
        list(dividend_yield_curve.times),
        list(dividend_yield_curve.cumulative_yields),
        [dividend.node for dividend in schedule.dividends],
        [dividend.amount for dividend in schedule.dividends],
        list(schedule.exercise_nodes),
        schedule.expiry_node,
        volatility.annualized_decimal,
        dividend_policy.value,
        executor._handle,  # noqa: SLF001 - wrapper owns the private native resource
        normals,
    )
    return _paths_from_native(schedule, states)


def simulate_gbm_seeded(  # noqa: PLR0913 - each explicit financial input is independently typed
    *,
    initial_spot: float,
    schedule: EventSchedule,
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    executor: CpuExecutor,
    paths: int,
    random_stream: RandomStream,
) -> GbmPaths:
    """Generate and evolve reproducible GBM paths without allocating a shock matrix.

    Path `i` is determined by `(seed, domain, i)` and is invariant to worker count and scheduling on
    the same numerical build. The engine uses Random123-compatible Philox4x32-10 and a bounded
    ZIGNOR standard-normal transform. A counter-budget or rejection-limit failure raises instead of
    changing the stream, sampler, precision, or backend.

    Raises:
        InputError: If a runtime type, grid, path count, or financial input is invalid.
        SimulationError: If random generation or a finite-input path transition fails.

    """
    _validate_initial_spot(initial_spot)
    _validate_simulation_specs(
        schedule,
        curve,
        dividend_yield_curve,
        volatility,
        dividend_policy,
        executor,
    )
    if type(paths) is not int or not 1 <= paths <= _MAX_PATHS:
        msg = f"paths must be a plain int in [1, {_MAX_PATHS}]; observed {paths!r}"
        raise InputError(msg)
    if type(random_stream) is not RandomStream:
        msg = f"random_stream must be RandomStream; observed {type(random_stream).__name__}"
        raise InputError(msg)

    states = _native.simulate_gbm_seeded_f64(
        initial_spot,
        list(schedule.times),
        list(curve.times),
        list(curve.log_discounts),
        list(dividend_yield_curve.times),
        list(dividend_yield_curve.cumulative_yields),
        [dividend.node for dividend in schedule.dividends],
        [dividend.amount for dividend in schedule.dividends],
        list(schedule.exercise_nodes),
        schedule.expiry_node,
        volatility.annualized_decimal,
        dividend_policy.value,
        executor._handle,  # noqa: SLF001 - wrapper owns the private native resource
        paths,
        random_stream.seed,
        random_stream.domain,
    )
    return _paths_from_native(schedule, states)


__all__ = [
    "CpuExecutor",
    "CpuMathBackend",
    "ExecutionError",
    "GbmPaths",
    "RandomStream",
    "SimulationError",
    "simulate_gbm",
    "simulate_gbm_seeded",
]
