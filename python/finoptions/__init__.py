"""Financially explicit, high-performance options research engine."""

from __future__ import annotations

from typing import Final, NamedTuple

from . import _native
from .lsm import (
    ExerciseFitDiagnostics,
    ExerciseTieRule,
    Lsm,
    LsmError,
    LsmResult,
    NoInTheMoneyPathsDiagnostics,
    OriginDecisionDiagnostics,
    RegressionFitDiagnostics,
    price_lsm_call,
)
from .model import (
    CashDividend,
    ConstantVolatility,
    DividendPolicy,
    EventSchedule,
    InputError,
    LogDiscountCurve,
)
from .simulation import (
    CpuExecutor,
    GbmPaths,
    RandomStream,
    SimulationError,
    simulate_gbm,
    simulate_gbm_seeded,
)


class BuildInfo(NamedTuple):
    """Native build identity recorded by reproducible run manifests."""

    package_version: str
    rustc_version: str


def build_info() -> BuildInfo:
    """Return the loaded native extension's package and compiler identity."""
    return BuildInfo(*_native.build_info())


__version__: Final = build_info().package_version
__all__ = [
    "BuildInfo",
    "CashDividend",
    "ConstantVolatility",
    "CpuExecutor",
    "DividendPolicy",
    "EventSchedule",
    "ExerciseFitDiagnostics",
    "ExerciseTieRule",
    "GbmPaths",
    "InputError",
    "LogDiscountCurve",
    "Lsm",
    "LsmError",
    "LsmResult",
    "NoInTheMoneyPathsDiagnostics",
    "OriginDecisionDiagnostics",
    "RandomStream",
    "RegressionFitDiagnostics",
    "SimulationError",
    "__version__",
    "build_info",
    "price_lsm_call",
    "simulate_gbm",
    "simulate_gbm_seeded",
]
