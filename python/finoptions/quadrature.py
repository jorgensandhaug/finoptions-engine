"""Deterministic event-aligned direct integration for finite call schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from . import _native
from .model import (
    ConstantVolatility,
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EventSchedule,
    InputError,
    LogDiscountCurve,
)
from .simulation import CpuExecutor

QuadratureError = _native.QuadratureError
_SUPPORTED_NORMAL_ORDERS: Final = (32, 64, 128, 256, 512, 1_024, 2_048)


@dataclass(frozen=True, slots=True)
class EventAlignedQuadrature:
    """Explicit stock grid, upper boundary, and standard-normal quadrature order."""

    upper_stock: float
    stock_points: int
    normal_order: int
    _handle: _native.EventAlignedQuadrature = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject inferred scalar types and construct one reusable native rule."""
        if type(self.upper_stock) is not float:
            msg = f"upper_stock must be a plain float; observed {type(self.upper_stock).__name__}"
            raise InputError(msg)
        if type(self.stock_points) is not int:
            msg = f"stock_points must be a plain int; observed {type(self.stock_points).__name__}"
            raise InputError(msg)
        if type(self.normal_order) is not int:
            msg = f"normal_order must be a plain int; observed {type(self.normal_order).__name__}"
            raise InputError(msg)
        if self.normal_order not in _SUPPORTED_NORMAL_ORDERS:
            msg = (
                f"normal_order must be one of {_SUPPORTED_NORMAL_ORDERS}; "
                f"observed {self.normal_order}"
            )
            raise InputError(msg)
        object.__setattr__(
            self,
            "_handle",
            _native.EventAlignedQuadrature(
                self.upper_stock,
                self.stock_points,
                self.normal_order,
            ),
        )


@dataclass(frozen=True, slots=True)
class QuadratureCallResult:
    """Deterministic value and the largest computed-grid/affine-tail boundary gap."""

    price: float
    maximum_upper_boundary_gap: float


def _validate_specs(  # noqa: PLR0913 - every financial input remains independently typed
    *,
    initial_spot: float,
    strike: float,
    schedule: EventSchedule,
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    executor: CpuExecutor,
    method: EventAlignedQuadrature,
) -> None:
    for name, value in (("initial_spot", initial_spot), ("strike", strike)):
        if type(value) is not float:
            msg = f"{name} must be a plain float; observed {type(value).__name__}"
            raise InputError(msg)
    for name, value, expected in (
        ("schedule", schedule, EventSchedule),
        ("curve", curve, LogDiscountCurve),
        ("dividend_yield_curve", dividend_yield_curve, CumulativeDividendYieldCurve),
        ("volatility", volatility, ConstantVolatility),
        ("dividend_policy", dividend_policy, DividendPolicy),
        ("executor", executor, CpuExecutor),
        ("method", method, EventAlignedQuadrature),
    ):
        if type(value) is not expected:
            msg = f"{name} must be {expected.__name__}; observed {type(value).__name__}"
            raise InputError(msg)


def price_event_aligned_call(  # noqa: PLR0913 - financial inputs are not bundled implicitly
    *,
    initial_spot: float,
    strike: float,
    schedule: EventSchedule,
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    executor: CpuExecutor,
    method: EventAlignedQuadrature,
) -> QuadratureCallResult:
    """Price the declared finite call exercise schedule without simulation or regression.

    Exercise observes the pre-dividend stock. A same-node cash jump follows exercise, while a
    terminal cash jump precedes expiry payoff. Every numerical control belongs to ``method`` and is
    used exactly; the implementation never refines, clips, retries, or changes backend.

    Raises:
        InputError: If a Python or financial input lies outside the declared domain.
        QuadratureError: If construction, allocation, transition, or arithmetic fails.

    """
    _validate_specs(
        initial_spot=initial_spot,
        strike=strike,
        schedule=schedule,
        curve=curve,
        dividend_yield_curve=dividend_yield_curve,
        volatility=volatility,
        dividend_policy=dividend_policy,
        executor=executor,
        method=method,
    )
    price, maximum_upper_boundary_gap = _native.price_event_aligned_call_f64(
        initial_spot,
        strike,
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
        method._handle,  # noqa: SLF001 - wrapper owns the private native resource
    )
    return QuadratureCallResult(
        price=price,
        maximum_upper_boundary_gap=maximum_upper_boundary_gap,
    )


__all__ = [
    "EventAlignedQuadrature",
    "QuadratureCallResult",
    "QuadratureError",
    "price_event_aligned_call",
]
