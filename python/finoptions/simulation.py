"""Strict, zero-copy-input Python surface for native path simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Final

import numpy as np
import numpy.typing as npt

from . import _native
from .model import (
    ConstantVolatility,
    DividendPolicy,
    EventSchedule,
    InputError,
    LogDiscountCurve,
)

Float64Matrix = npt.NDArray[np.float64]
SimulationError = _native.SimulationError
_MATRIX_DIMENSIONS: Final = 2


@dataclass(frozen=True, slots=True)
class CpuExecutor:
    """Reusable dedicated CPU execution resource with an exact worker count."""

    threads: int
    _handle: _native.CpuExecutor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject inferred/empty thread counts and construct the native pool once."""
        if type(self.threads) is not int or self.threads <= 0:
            msg = f"threads must be a positive plain int; observed {self.threads!r}"
            raise InputError(msg)
        object.__setattr__(self, "_handle", _native.CpuExecutor(self.threads))


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


def _validate_simulation_specs(
    schedule: EventSchedule,
    curve: LogDiscountCurve,
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


def simulate_gbm(  # noqa: PLR0913 - each explicit financial input is independently typed
    *,
    initial_spot: float,
    schedule: EventSchedule,
    curve: LogDiscountCurve,
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
    if type(initial_spot) is not float:
        msg = f"initial_spot must be a plain float; observed {type(initial_spot).__name__}"
        raise InputError(msg)
    _validate_simulation_specs(schedule, curve, volatility, dividend_policy, executor)
    _validate_normal_matrix(normals)

    states = _native.simulate_gbm_f64(
        initial_spot,
        list(schedule.times),
        list(curve.times),
        list(curve.log_discounts),
        [dividend.node for dividend in schedule.dividends],
        [dividend.amount for dividend in schedule.dividends],
        list(schedule.exercise_nodes),
        schedule.expiry_node,
        volatility.annualized_decimal,
        dividend_policy.value,
        executor._handle,  # noqa: SLF001 - wrapper owns the private native resource
        normals,
    )
    states.flags.writeable = False
    return GbmPaths(
        observation_nodes=(*schedule.exercise_nodes, schedule.expiry_node),
        states=states,
    )


__all__ = ["CpuExecutor", "GbmPaths", "SimulationError", "simulate_gbm"]
