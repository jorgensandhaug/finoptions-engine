"""Financially explicit, high-performance options research engine."""

from __future__ import annotations

from typing import Final, NamedTuple

from . import _native
from .batch_lsm import (
    BatchDividends,
    CashDividendBatch,
    ContinuousDividendYieldBatch,
    EscrowedCashDividendBatch,
    EscrowedVolatilityConvention,
    LsmBatchResult,
    LsmStrikeGroups,
    OriginExercise,
    UniformExerciseMesh,
    price_lsm_call_batch,
    price_lsm_call_grouped_batch,
)
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
    InsufficientExerciseSupportDiagnostics,
    InsufficientExerciseSupportPolicy,
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
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EventSchedule,
    InputError,
    LogDiscountCurve,
)
from .simulation import (
    CpuExecutor,
    CpuMathBackend,
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
    "BatchDividends",
    "BatchError",
    "BinnedCashDividends",
    "BuildInfo",
    "CashAmountScale",
    "CashDividend",
    "CashDividendBatch",
    "ConstantVolatility",
    "ContinuousDividendYieldBatch",
    "CpuExecutor",
    "CpuMathBackend",
    "CumulativeDividendYieldCurve",
    "DividendBinSpec",
    "DividendPolicy",
    "EscrowedCashDividendBatch",
    "EscrowedVolatilityConvention",
    "EventSchedule",
    "ExecutionError",
    "ExerciseFitDiagnostics",
    "ExerciseTieRule",
    "GbmPaths",
    "InputError",
    "InsufficientExerciseSupportDiagnostics",
    "InsufficientExerciseSupportPolicy",
    "LogDiscountCurve",
    "Lsm",
    "LsmBatchResult",
    "LsmError",
    "LsmResult",
    "LsmStrikeGroups",
    "MaturityBoundary",
    "NoInTheMoneyPathsDiagnostics",
    "OriginDecisionDiagnostics",
    "OriginExercise",
    "PaddedCashDividends",
    "PaddedSequenceSpec",
    "RaggedCashDividends",
    "RandomStream",
    "RegressionFitDiagnostics",
    "SimulationError",
    "TimeBinClosure",
    "UniformExerciseMesh",
    "__version__",
    "bin_cash_dividends",
    "build_info",
    "pack_padded_cash_dividends",
    "price_lsm_call",
    "price_lsm_call_batch",
    "price_lsm_call_grouped_batch",
    "simulate_gbm",
    "simulate_gbm_seeded",
]
