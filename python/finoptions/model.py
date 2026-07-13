"""Small immutable Python specifications validated by the Rust financial core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from . import _native

InputError = _native.InputError


def _require_plain_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        msg = f"{field} must be a plain int; observed {type(value).__name__}"
        raise InputError(msg)
    if value < 0 or value > (2**32 - 1):
        msg = f"{field} must be in the unsigned 32-bit node domain; observed {value}"
        raise InputError(msg)
    return value


def _require_float_tuple(values: object, *, field: str) -> tuple[float, ...]:
    if type(values) is not tuple:
        msg = f"{field} must be an immutable tuple[float, ...]"
        raise InputError(msg)
    if any(type(value) is not float for value in values):
        msg = f"every {field} value must be a plain float"
        raise InputError(msg)
    return cast("tuple[float, ...]", values)


def _require_int_tuple(values: object, *, field: str) -> tuple[int, ...]:
    if type(values) is not tuple:
        msg = f"{field} must be an immutable tuple[int, ...]"
        raise InputError(msg)
    return tuple(_require_plain_int(value, field=field) for value in values)


class DividendPolicy(StrEnum):
    """Explicit limited-liability treatment of deterministic cash dividends."""

    LIQUIDATOR = "liquidator"
    SURVIVOR = "survivor"


@dataclass(frozen=True, slots=True)
class CashDividend:
    """A validated nonnegative cash amount attached to an exact node ID."""

    node: int
    amount: float

    def __post_init__(self) -> None:
        """Validate exact runtime types, node domain, finiteness, sign, and canonical zero."""
        _require_plain_int(self.node, field="cash-dividend node")
        if type(self.amount) is not float:
            msg = (
                f"cash-dividend amount must be a plain float; observed {type(self.amount).__name__}"
            )
            raise InputError(msg)
        _native.validate_cash_dividend_amount(self.amount)


@dataclass(frozen=True, slots=True)
class EventSchedule:
    """Finite event grid with pre-dividend exercise and separate terminal expiry."""

    times: tuple[float, ...]
    dividends: tuple[CashDividend, ...]
    exercise_nodes: tuple[int, ...]
    expiry_node: int

    def __post_init__(self) -> None:
        """Validate the complete schedule in the Rust financial core."""
        times = _require_float_tuple(self.times, field="event times")
        if type(self.dividends) is not tuple or any(
            type(dividend) is not CashDividend for dividend in self.dividends
        ):
            msg = "dividends must be an immutable tuple[CashDividend, ...]"
            raise InputError(msg)
        exercise_nodes = _require_int_tuple(self.exercise_nodes, field="exercise nodes")
        expiry_node = _require_plain_int(self.expiry_node, field="expiry node")
        _native.validate_event_schedule(
            list(times),
            [dividend.node for dividend in self.dividends],
            [dividend.amount for dividend in self.dividends],
            list(exercise_nodes),
            expiry_node,
        )


@dataclass(frozen=True, slots=True)
class LogDiscountCurve:
    """Validated `log(P(0,t))` values aligned with an exact time grid."""

    times: tuple[float, ...]
    log_discounts: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate node alignment, exact origins, finiteness, and representable intervals."""
        times = _require_float_tuple(self.times, field="curve times")
        log_discounts = _require_float_tuple(self.log_discounts, field="log discounts")
        _native.validate_log_discount_curve(list(times), list(log_discounts))
