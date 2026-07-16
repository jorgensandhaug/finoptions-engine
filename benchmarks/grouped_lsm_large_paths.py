"""Measure grouped cash-date LSM at publication-scale path counts."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from finoptions import (
    CashDividendBatch,
    CashDividendExerciseMesh,
    CpuExecutor,
    CpuMathBackend,
    DividendPolicy,
    ExerciseTieRule,
    InsufficientExerciseSupportPolicy,
    Lsm,
    LsmStrikeGroups,
    MaturityBoundary,
    RaggedCashDividends,
    RandomStream,
    price_lsm_call_grouped_batch,
)


def main() -> None:
    """Run one deterministic grouped workload and report complete boundary time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--paths", type=int, default=32_768)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument(
        "--math-backend",
        choices=tuple(backend.value for backend in CpuMathBackend),
        default=CpuMathBackend.RUST_STD_F64.value,
    )
    arguments = parser.parse_args()

    relative_strikes = np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20], dtype=np.float64)
    groups = arguments.groups
    strikes_per_group = int(relative_strikes.size)
    group_offsets = np.arange(
        0,
        (groups + 1) * strikes_per_group,
        strikes_per_group,
        dtype=np.uint32,
    )
    spots = np.linspace(80.0, 140.0, groups, dtype=np.float64)
    maturities = np.linspace(0.25, 1.0, groups, dtype=np.float64)
    strikes = np.ascontiguousarray((spots[:, None] * relative_strikes).reshape(-1))
    dividends = CashDividendBatch(
        schedules=RaggedCashDividends(
            offsets=np.arange(groups + 1, dtype=np.int64),
            times=np.ascontiguousarray(maturities * 0.5),
            amounts=np.full(groups, 2.0, dtype=np.float64),
        ),
        policy=DividendPolicy.LIQUIDATOR,
        maturity_boundary=MaturityBoundary.STRICTLY_BEFORE,
    )
    request = LsmStrikeGroups(
        group_offsets=group_offsets,
        initial_spots=spots,
        strikes=strikes,
        maturities=maturities,
        rates=np.full(groups, 0.03, dtype=np.float64),
        volatilities=np.full(groups, 0.25, dtype=np.float64),
        dividends=dividends,
    )
    method = Lsm(
        training_paths=arguments.paths,
        evaluation_paths=arguments.paths,
        training_stream=RandomStream(seed=7, domain=10_000),
        evaluation_stream=RandomStream(seed=11, domain=1_000_000),
        polynomial_degree=4,
        rank_tolerance=1.0e-12,
        condition_limit=1.0e8,
        tie_rule=ExerciseTieRule.STRICTLY_GREATER,
        insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
    )
    executor = CpuExecutor(
        threads=arguments.threads,
        math_backend=CpuMathBackend(arguments.math_backend),
    )
    started = time.perf_counter()
    result = price_lsm_call_grouped_batch(
        groups=request,
        mesh=CashDividendExerciseMesh(),
        executor=executor,
        method=method,
    )
    elapsed = time.perf_counter() - started
    sys.stdout.write(
        json.dumps(
            {
                "checksum": float(np.sum(result.prices, dtype=np.float64)),
                "contracts": result.rows,
                "elapsed_seconds": elapsed,
                "groups": groups,
                "math_backend": arguments.math_backend,
                "paths_per_role": arguments.paths,
                "threads": arguments.threads,
            },
            sort_keys=True,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
