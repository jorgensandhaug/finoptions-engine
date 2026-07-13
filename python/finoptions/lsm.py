"""Independent-policy least-squares Monte Carlo for explicitly scheduled calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
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
from .simulation import CpuExecutor, RandomStream

LsmError = _native.LsmError
_MAX_PATHS: Final = (2**32) - 1
_MAX_POLYNOMIAL_DEGREE: Final = 4


class ExerciseTieRule(StrEnum):
    """Explicit decision when intrinsic and fitted continuation compare equal."""

    STRICTLY_GREATER = "strictly_greater"
    GREATER_OR_EQUAL = "greater_or_equal"


@dataclass(frozen=True, slots=True)
class Lsm:
    """Complete simulation/regression identity for independent-policy LSM.

    The degree-four defaults are part of this immutable value; they are never silently changed.
    Training and evaluation must use different random domains.
    """

    training_paths: int
    evaluation_paths: int
    training_stream: RandomStream
    evaluation_stream: RandomStream
    polynomial_degree: int = 4
    rank_tolerance: float = 1.0e-12
    condition_limit: float = 1.0e8
    tie_rule: ExerciseTieRule = ExerciseTieRule.STRICTLY_GREATER

    def __post_init__(self) -> None:
        """Reject inferred types, unsupported degrees, invalid gates, and reused domains."""
        for field, value in (
            ("training_paths", self.training_paths),
            ("evaluation_paths", self.evaluation_paths),
        ):
            if type(value) is not int or not 1 <= value <= _MAX_PATHS:
                msg = f"{field} must be a plain int in [1, {_MAX_PATHS}]; observed {value!r}"
                raise InputError(msg)
        if type(self.training_stream) is not RandomStream:
            msg = "training_stream must be RandomStream"
            raise InputError(msg)
        if type(self.evaluation_stream) is not RandomStream:
            msg = "evaluation_stream must be RandomStream"
            raise InputError(msg)
        if self.training_stream.domain == self.evaluation_stream.domain:
            msg = (
                "training_stream and evaluation_stream must use different domains; "
                f"both use {self.training_stream.domain}"
            )
            raise InputError(msg)
        if (
            type(self.polynomial_degree) is not int
            or not 0 <= self.polynomial_degree <= _MAX_POLYNOMIAL_DEGREE
        ):
            msg = (
                "polynomial_degree must be a plain int in "
                f"[0, {_MAX_POLYNOMIAL_DEGREE}]; observed {self.polynomial_degree!r}"
            )
            raise InputError(msg)
        if (
            type(self.rank_tolerance) is not float
            or not isfinite(self.rank_tolerance)
            or not 0.0 < self.rank_tolerance < 1.0
        ):
            msg = (
                "rank_tolerance must be a finite plain float strictly between 0 and 1; "
                f"observed {self.rank_tolerance!r}"
            )
            raise InputError(msg)
        maximum_condition = 1.0 / self.rank_tolerance
        if (
            type(self.condition_limit) is not float
            or not isfinite(self.condition_limit)
            or not 1.0 <= self.condition_limit <= maximum_condition
        ):
            msg = (
                "condition_limit must be a finite plain float in "
                f"[1, {maximum_condition}]; observed {self.condition_limit!r}"
            )
            raise InputError(msg)
        if type(self.tie_rule) is not ExerciseTieRule:
            msg = f"tie_rule must be ExerciseTieRule; observed {type(self.tie_rule).__name__}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class OriginDecisionDiagnostics:
    """Common valuation-origin policy decision fitted from training continuation."""

    node: int
    exercise: bool
    estimated_continuation: float


@dataclass(frozen=True, slots=True)
class NoInTheMoneyPathsDiagnostics:
    """Training had no in-the-money state at this exercise node."""

    node: int


@dataclass(frozen=True, slots=True)
class RegressionFitDiagnostics:
    """Accepted continuation regression population and numerical diagnostics."""

    node: int
    in_the_money_paths: int
    observations: int
    columns: int
    rank: int
    condition_number: float
    predictor_mean: float
    predictor_scale: float


type ExerciseFitDiagnostics = (
    OriginDecisionDiagnostics | NoInTheMoneyPathsDiagnostics | RegressionFitDiagnostics
)


@dataclass(frozen=True, slots=True)
class LsmResult:
    """Independent mean payoff, sampling error, stopping counts, and fit diagnostics."""

    price: float
    paths: int
    sample_standard_deviation: float | None
    standard_error: float | None
    observation_nodes: tuple[int, ...]
    exercise_counts: tuple[int, ...]
    fit_diagnostics: tuple[ExerciseFitDiagnostics, ...]


def _required[T](value: T | None, *, field: str) -> T:
    if value is None:
        msg = f"native LSM result omitted required {field}"
        raise LsmError(msg)
    return value


def _fit_diagnostic(raw: _native.NativeExerciseDiagnostic) -> ExerciseFitDiagnostics:
    (
        node,
        kind,
        exercise,
        estimated_continuation,
        in_the_money_paths,
        observations,
        columns,
        rank,
        condition_number,
        predictor_mean,
        predictor_scale,
    ) = raw
    if kind == "origin_decision":
        return OriginDecisionDiagnostics(
            node=node,
            exercise=_required(exercise, field="origin exercise decision"),
            estimated_continuation=_required(
                estimated_continuation,
                field="origin continuation estimate",
            ),
        )
    if kind == "no_in_the_money_paths":
        return NoInTheMoneyPathsDiagnostics(node=node)
    if kind == "regression":
        return RegressionFitDiagnostics(
            node=node,
            in_the_money_paths=_required(in_the_money_paths, field="in-the-money path count"),
            observations=_required(observations, field="regression observation count"),
            columns=_required(columns, field="regression column count"),
            rank=_required(rank, field="regression rank"),
            condition_number=_required(condition_number, field="regression condition"),
            predictor_mean=_required(predictor_mean, field="regression predictor mean"),
            predictor_scale=_required(predictor_scale, field="regression predictor scale"),
        )
    msg = f"native LSM result returned unsupported diagnostic kind {kind!r}"
    raise LsmError(msg)


def _validate_specs(  # noqa: PLR0913 - financial inputs remain independently typed
    *,
    initial_spot: float,
    strike: float,
    schedule: EventSchedule,
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    executor: CpuExecutor,
    method: Lsm,
) -> None:
    for field, value in (("initial_spot", initial_spot), ("strike", strike)):
        if type(value) is not float:
            msg = f"{field} must be a plain float; observed {type(value).__name__}"
            raise InputError(msg)
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
    if type(method) is not Lsm:
        msg = f"method must be Lsm; observed {type(method).__name__}"
        raise InputError(msg)


def price_lsm_call(  # noqa: PLR0913 - financial inputs remain independently typed
    *,
    initial_spot: float,
    strike: float,
    schedule: EventSchedule,
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    executor: CpuExecutor,
    method: Lsm,
) -> LsmResult:
    """Fit a call stopping policy and evaluate it only on its independent path population.

    Exercise states are observed before any same-node cash dividend. Every estimator-defining
    numerical value is carried by `method`; failures never reduce degree, add regularization, reuse
    training paths, or change the requested backend.

    Raises:
        InputError: If an explicit Python or financial specification is invalid.
        LsmError: If simulation, regression, support, or fixed-policy evaluation fails.

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
    raw = _native.price_lsm_call_seeded_f64(
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
    )
    (
        price,
        paths,
        sample_standard_deviation,
        standard_error,
        observation_nodes,
        exercise_counts,
        diagnostics,
    ) = raw
    return LsmResult(
        price=price,
        paths=paths,
        sample_standard_deviation=sample_standard_deviation,
        standard_error=standard_error,
        observation_nodes=tuple(observation_nodes),
        exercise_counts=tuple(exercise_counts),
        fit_diagnostics=tuple(_fit_diagnostic(diagnostic) for diagnostic in diagnostics),
    )


__all__ = [
    "ExerciseFitDiagnostics",
    "ExerciseTieRule",
    "Lsm",
    "LsmError",
    "LsmResult",
    "NoInTheMoneyPathsDiagnostics",
    "OriginDecisionDiagnostics",
    "RegressionFitDiagnostics",
    "price_lsm_call",
]
