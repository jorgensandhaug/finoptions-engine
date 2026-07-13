"""Financially explicit, high-performance options research engine."""

from __future__ import annotations

from typing import Final, NamedTuple

from . import _native
from .dividends import (
    BatchError,
    BinnedCashDividends,
    CashAmountScale,
    DividendBinSpec,
    MaturityBoundary,
    PaddedCashDividends,
    PaddedSequenceSpec,
    RaggedCashDividends,
    TimeBinClosure,
    bin_cash_dividends,
    pack_padded_cash_dividends,
)
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
    ExecutionError,
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
    "BatchError",
    "BinnedCashDividends",
    "BuildInfo",
    "CashAmountScale",
    "CashDividend",
    "ConstantVolatility",
    "CpuExecutor",
    "DividendBinSpec",
    "DividendPolicy",
    "EventSchedule",
    "ExecutionError",
    "ExerciseFitDiagnostics",
    "ExerciseTieRule",
    "GbmPaths",
    "InputError",
    "LogDiscountCurve",
    "Lsm",
    "LsmError",
    "LsmResult",
    "MaturityBoundary",
    "NoInTheMoneyPathsDiagnostics",
    "OriginDecisionDiagnostics",
    "PaddedCashDividends",
    "PaddedSequenceSpec",
    "RaggedCashDividends",
    "RandomStream",
    "RegressionFitDiagnostics",
    "SimulationError",
    "TimeBinClosure",
    "__version__",
    "bin_cash_dividends",
    "build_info",
    "pack_padded_cash_dividends",
    "price_lsm_call",
    "simulate_gbm",
    "simulate_gbm_seeded",
]
