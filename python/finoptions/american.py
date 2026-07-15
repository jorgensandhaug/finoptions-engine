"""Fast deterministic American calls with continuous proportional yields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt

from . import _native
from .model import InputError
from .simulation import CpuExecutor

AmericanOptionError = _native.AmericanOptionError

Float64Vector = npt.NDArray[np.float64]
BoolVector = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class AndersenLake:
    """Exact fixed controls for one grouped Andersen-Lake method.

    ``iterations`` is the exact number of ordinary system-B fixed-point steps. The explicit
    obstacle-projection limit bounds finite-precision restoration of the stopping lower bound and
    may be zero. The method never adapts, retries, or changes solver.
    """

    boundary_quadrature_order: int
    iterations: int
    collocation_nodes: int
    price_quadrature_order: int
    maximum_relative_boundary_residual: float
    maximum_relative_obstacle_projection: float
    _handle: _native.AndersenLake = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Construct one reusable native rule after strict scalar-type checks."""
        for name, value in (
            ("boundary_quadrature_order", self.boundary_quadrature_order),
            ("iterations", self.iterations),
            ("collocation_nodes", self.collocation_nodes),
            ("price_quadrature_order", self.price_quadrature_order),
        ):
            if type(value) is not int:
                msg = f"{name} must be a plain int; observed {type(value).__name__}"
                raise InputError(msg)
        if type(self.maximum_relative_boundary_residual) is not float:
            msg = (
                "maximum_relative_boundary_residual must be a plain float; observed "
                f"{type(self.maximum_relative_boundary_residual).__name__}"
            )
            raise InputError(msg)
        if type(self.maximum_relative_obstacle_projection) is not float:
            msg = (
                "maximum_relative_obstacle_projection must be a plain float; observed "
                f"{type(self.maximum_relative_obstacle_projection).__name__}"
            )
            raise InputError(msg)
        object.__setattr__(
            self,
            "_handle",
            _native.AndersenLake(
                self.boundary_quadrature_order,
                self.iterations,
                self.collocation_nodes,
                self.price_quadrature_order,
                self.maximum_relative_boundary_residual,
                self.maximum_relative_obstacle_projection,
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class AndersenLakeBatchResult:
    """Contract prices and one deterministic convergence record per natural group."""

    prices: Float64Vector
    exercised_at_origin: BoolVector
    maximum_relative_boundary_residuals: Float64Vector
    maximum_relative_obstacle_projections: Float64Vector
    used_european_identity: BoolVector

    def __post_init__(self) -> None:
        """Require immutable, shape-aligned native result vectors."""
        rows = int(self.prices.size)
        groups = int(self.maximum_relative_boundary_residuals.size)
        for name, values, dtype, size in (
            ("prices", self.prices, np.dtype(np.float64), rows),
            ("exercised_at_origin", self.exercised_at_origin, np.dtype(np.bool_), rows),
            (
                "maximum_relative_boundary_residuals",
                self.maximum_relative_boundary_residuals,
                np.dtype(np.float64),
                groups,
            ),
            (
                "maximum_relative_obstacle_projections",
                self.maximum_relative_obstacle_projections,
                np.dtype(np.float64),
                groups,
            ),
            ("used_european_identity", self.used_european_identity, np.dtype(np.bool_), groups),
        ):
            _require_vector(name, values, dtype)
            if values.size != size:
                msg = f"{name} has {values.size} values; expected {size}"
                raise InputError(msg)
            if values.flags.writeable:
                msg = f"{name} must be immutable"
                raise InputError(msg)
        if rows == 0 or groups == 0:
            msg = "Andersen-Lake results require at least one group and one price"
            raise InputError(msg)

    @property
    def rows(self) -> int:
        """Return the priced contract count."""
        return int(self.prices.size)

    @property
    def groups(self) -> int:
        """Return the solved or theorem-routed natural state count."""
        return int(self.maximum_relative_boundary_residuals.size)


def _require_vector[Scalar: np.generic](
    name: str,
    value: object,
    dtype: np.dtype[Scalar],
) -> npt.NDArray[Scalar]:
    if type(value) is not np.ndarray:
        msg = f"{name} must be a numpy.ndarray; observed {type(value).__name__}"
        raise InputError(msg)
    array = value
    if array.dtype != dtype:
        msg = f"{name} must have dtype {dtype}; observed {array.dtype}"
        raise InputError(msg)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional; observed {array.ndim} dimensions"
        raise InputError(msg)
    if not array.flags.c_contiguous:
        msg = f"{name} must be C-contiguous"
        raise InputError(msg)
    return cast("npt.NDArray[Scalar]", array)


def price_andersen_lake_call_grouped(  # noqa: PLR0913 - columns stay explicit and zero-copy
    *,
    group_offsets: npt.NDArray[np.int64],
    group_spots: npt.NDArray[np.float64],
    group_maturities: npt.NDArray[np.float64],
    group_constant_continuously_compounded_rates: npt.NDArray[np.float64],
    group_annualized_dividend_yields: npt.NDArray[np.float64],
    group_constant_volatilities: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    executor: CpuExecutor,
    method: AndersenLake,
) -> AndersenLakeBatchResult:
    """Price grouped continuous-yield American calls without simulation.

    A group shares spot, maturity, rate, yield, and volatility across one or more strikes. The
    implementation uses exact American call-put symmetry and one normalized put exercise boundary
    per group. For zero-yield calls with nonnegative rates, the exact no-early-exercise theorem uses
    Black-Scholes directly. Every other admitted group uses fixed Chebyshev collocation with
    shape-preserving reconstruction of the transformed boundary. The exact stopping payoff is used
    in the solved exercise region. A continuation-side value below the stopping obstacle is
    projected and reported only within the method's explicit normalized limit; a larger gap, an
    excessive boundary residual, or a one-boundary invariant failure raises.

    Raises:
        InputError: If a Python, structural, or financial input is outside the admitted domain.
        AmericanOptionError: If construction, convergence, interpolation, or pricing fails.

    """
    if type(executor) is not CpuExecutor:
        msg = f"executor must be CpuExecutor; observed {type(executor).__name__}"
        raise InputError(msg)
    if type(method) is not AndersenLake:
        msg = f"method must be AndersenLake; observed {type(method).__name__}"
        raise InputError(msg)
    offsets = _require_vector("group_offsets", group_offsets, np.dtype(np.int64))
    spots = _require_vector("group_spots", group_spots, np.dtype(np.float64))
    maturities = _require_vector("group_maturities", group_maturities, np.dtype(np.float64))
    rates = _require_vector(
        "group_constant_continuously_compounded_rates",
        group_constant_continuously_compounded_rates,
        np.dtype(np.float64),
    )
    yields = _require_vector(
        "group_annualized_dividend_yields",
        group_annualized_dividend_yields,
        np.dtype(np.float64),
    )
    volatilities = _require_vector(
        "group_constant_volatilities",
        group_constant_volatilities,
        np.dtype(np.float64),
    )
    strike_values = _require_vector("strikes", strikes, np.dtype(np.float64))
    prices, exercised_at_origin, residuals, obstacle_projections, european_identities = (
        _native.price_andersen_lake_call_grouped_f64(
            offsets,
            spots,
            maturities,
            rates,
            yields,
            volatilities,
            strike_values,
            executor._handle,  # noqa: SLF001 - wrapper owns the private native resource
            method._handle,  # noqa: SLF001 - wrapper owns the private native resource
        )
    )
    for values in (
        prices,
        exercised_at_origin,
        residuals,
        obstacle_projections,
        european_identities,
    ):
        values.flags.writeable = False
    return AndersenLakeBatchResult(
        prices=prices,
        exercised_at_origin=exercised_at_origin,
        maximum_relative_boundary_residuals=residuals,
        maximum_relative_obstacle_projections=obstacle_projections,
        used_european_identity=european_identities,
    )


__all__ = [
    "AmericanOptionError",
    "AndersenLake",
    "AndersenLakeBatchResult",
    "price_andersen_lake_call_grouped",
]
