"""Measure public batch-LSM throughput and incremental process high-water RSS."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
from time import perf_counter
from typing import Final

import numpy as np
from finoptions import (
    ContinuousDividendYieldBatch,
    CpuExecutor,
    CpuMathBackend,
    ExerciseTieRule,
    InsufficientExerciseSupportPolicy,
    Lsm,
    OriginExercise,
    RandomStream,
    UniformExerciseMesh,
    price_lsm_call_batch,
)

_DEFAULT_ROWS: Final = 4_096
_DEFAULT_THREADS: Final = 4
_PATHS: Final = 500
_STEPS: Final = 150


def _max_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1_024


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS)
    parser.add_argument("--threads", type=int, default=_DEFAULT_THREADS)
    parser.add_argument(
        "--math-backend",
        choices=tuple(backend.value for backend in CpuMathBackend),
        default=CpuMathBackend.RUST_STD_F64.value,
    )
    return parser.parse_args()


def main() -> None:
    """Run one fresh-process project-shaped public API measurement and emit JSON."""
    arguments = _arguments()
    rows = arguments.rows
    threads = arguments.threads
    math_backend = CpuMathBackend(arguments.math_backend)
    if rows <= 0 or threads <= 0:
        msg = "rows and threads must be positive"
        raise ValueError(msg)

    initial_spots = np.full(rows, 150.0, dtype=np.float64)
    strikes = np.full(rows, 100.0, dtype=np.float64)
    maturities = np.full(rows, 1.0, dtype=np.float64)
    rates = np.full(rows, 0.05, dtype=np.float64)
    volatilities = np.full(rows, 0.25, dtype=np.float64)
    yields = np.full(rows, 0.02, dtype=np.float64)
    executor = CpuExecutor(threads=threads, math_backend=math_backend)
    method = Lsm(
        training_paths=_PATHS,
        evaluation_paths=_PATHS,
        training_stream=RandomStream(seed=10_000, domain=0),
        evaluation_stream=RandomStream(seed=20_000, domain=1_000_000),
        insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
        polynomial_degree=4,
        rank_tolerance=1.0e-12,
        condition_limit=1.0e8,
        tie_rule=ExerciseTieRule.STRICTLY_GREATER,
    )
    gc.collect()
    baseline = _max_rss_bytes()
    started = perf_counter()
    result = price_lsm_call_batch(
        initial_spots=initial_spots,
        strikes=strikes,
        maturities=maturities,
        rates=rates,
        volatilities=volatilities,
        dividends=ContinuousDividendYieldBatch(annualized_decimals=yields),
        mesh=UniformExerciseMesh(
            base_intervals=_STEPS,
            origin_exercise=OriginExercise.EXCLUDED,
        ),
        executor=executor,
        method=method,
    )
    elapsed = perf_counter() - started
    peak = _max_rss_bytes()
    if result.rows != rows:
        msg = f"native result returned {result.rows} rows; expected {rows}"
        raise RuntimeError(msg)
    report = {
        "baseline_max_rss_bytes": baseline,
        "contracts_per_second": rows / elapsed,
        "elapsed_seconds": elapsed,
        "incremental_peak_rss_bytes": peak - baseline,
        "math_backend": math_backend.value,
        "peak_max_rss_bytes": peak,
        "rows": rows,
        "threads": threads,
    }
    sys.stdout.write(f"{json.dumps(report, sort_keys=True)}\n")


if __name__ == "__main__":
    main()
