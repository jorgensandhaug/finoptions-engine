"""Measure the explicit Polars-to-native LSM preparation boundary."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import sys
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Final

import numpy as np
import polars as pl
from finoptions import DividendPolicy, MaturityBoundary
from finoptions.polars import (
    BufferPolicy,
    CashDividendColumns,
    ContinuousYieldColumn,
    DividendColumns,
    LsmCallColumns,
    prepare_lsm_call_batch,
    select_lsm_call_inputs,
)

_DEFAULT_ROWS: Final = 1_000_000
_DEFAULT_ITERATIONS: Final = 6
_MINIMUM_ITERATIONS: Final = 2


class DividendRepresentation(StrEnum):
    """Synthetic dividend input admitted by this boundary benchmark."""

    CONTINUOUS_YIELD = "continuous_yield"
    CASH = "cash"


@dataclass(frozen=True, slots=True)
class BenchmarkArguments:
    """Validated command-line configuration."""

    rows: int
    dividends: DividendRepresentation
    iterations: int


@dataclass(frozen=True, slots=True)
class BoundaryTiming:
    """One complete projection, collection and preparation observation."""

    plan_seconds: float
    collect_seconds: float
    preparation_seconds: float


def _max_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1_024


def _arguments() -> BenchmarkArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS)
    parser.add_argument("--iterations", type=int, default=_DEFAULT_ITERATIONS)
    parser.add_argument(
        "--dividends",
        choices=tuple(representation.value for representation in DividendRepresentation),
        default=DividendRepresentation.CONTINUOUS_YIELD.value,
    )
    parsed = parser.parse_args()
    if type(parsed.rows) is not int or parsed.rows <= 0:
        message = "rows must be a positive plain integer"
        raise ValueError(message)
    if type(parsed.iterations) is not int or parsed.iterations < _MINIMUM_ITERATIONS:
        message = "iterations must be a plain integer of at least two"
        raise ValueError(message)
    return BenchmarkArguments(
        rows=parsed.rows,
        dividends=DividendRepresentation(parsed.dividends),
        iterations=parsed.iterations,
    )


def _source(arguments: BenchmarkArguments) -> pl.DataFrame:
    rows = arguments.rows
    row = np.arange(rows, dtype=np.float64)
    columns: dict[str, pl.Series | np.ndarray] = {
        "source_row": np.arange(rows, dtype=np.uint64),
        "spot": np.ascontiguousarray(80.0 + np.remainder(row, 12_001.0) / 100.0),
        "strike": np.ascontiguousarray(75.0 + np.remainder(row, 13_001.0) / 100.0),
        "maturity": np.ascontiguousarray(0.05 + np.remainder(row, 340.0) / 365.0),
        "rate": np.ascontiguousarray(0.01 + np.remainder(row, 400.0) / 10_000.0),
        "volatility": np.ascontiguousarray(0.12 + np.remainder(row, 600.0) / 2_000.0),
        "unused_feature": np.ascontiguousarray(np.remainder(row, 997.0)),
    }
    if arguments.dividends is DividendRepresentation.CONTINUOUS_YIELD:
        columns["dividend_yield"] = np.ascontiguousarray(
            0.005 + np.remainder(row, 300.0) / 10_000.0,
        )
    else:
        columns["dividend_times"] = pl.repeat(
            pl.lit([0.20, 0.45], dtype=pl.List(pl.Float64)),
            n=rows,
            eager=True,
        ).alias("dividend_times")
        columns["dividend_amounts"] = pl.repeat(
            pl.lit([0.50, 0.55], dtype=pl.List(pl.Float64)),
            n=rows,
            eager=True,
        ).alias("dividend_amounts")
    return pl.DataFrame(columns)


def _measure_boundary(
    source: pl.DataFrame,
    *,
    columns: LsmCallColumns,
    dividends: DividendColumns,
    rows: int,
) -> BoundaryTiming:
    plan_started = perf_counter()
    selected = select_lsm_call_inputs(source.lazy(), columns=columns, dividends=dividends)
    plan_seconds = perf_counter() - plan_started
    collect_started = perf_counter()
    materialized = selected.collect()
    collect_seconds = perf_counter() - collect_started
    preparation_started = perf_counter()
    prepared = prepare_lsm_call_batch(
        materialized,
        columns=columns,
        dividends=dividends,
        buffer_policy=BufferPolicy.REQUIRE_ZERO_COPY,
    )
    preparation_seconds = perf_counter() - preparation_started

    if prepared.rows != rows:
        message = f"prepared boundary returned {prepared.rows} rows; expected {rows}"
        raise RuntimeError(message)
    materialized_spots = materialized.get_column("spot").to_numpy(
        writable=False,
        allow_copy=False,
    )
    if not np.shares_memory(materialized_spots, prepared.initial_spots):
        message = "prepared scalar storage does not borrow the materialized Polars buffer"
        raise RuntimeError(message)
    if prepared.initial_spots.flags.writeable:
        message = "prepared scalar storage unexpectedly permits mutation"
        raise RuntimeError(message)
    return BoundaryTiming(
        plan_seconds=plan_seconds,
        collect_seconds=collect_seconds,
        preparation_seconds=preparation_seconds,
    )


def main() -> None:
    """Run one fresh-process boundary measurement and emit canonical JSON."""
    arguments = _arguments()
    source = _source(arguments)
    columns = LsmCallColumns(
        initial_spot="spot",
        strike="strike",
        maturity="maturity",
        rate="rate",
        volatility="volatility",
    )
    if arguments.dividends is DividendRepresentation.CONTINUOUS_YIELD:
        dividends = ContinuousYieldColumn("dividend_yield")
    else:
        dividends = CashDividendColumns(
            times="dividend_times",
            amounts="dividend_amounts",
            policy=DividendPolicy.LIQUIDATOR,
            maturity_boundary=MaturityBoundary.STRICTLY_BEFORE,
        )

    gc.collect()
    baseline_rss = _max_rss_bytes()
    timings = tuple(
        _measure_boundary(
            source,
            columns=columns,
            dividends=dividends,
            rows=arguments.rows,
        )
        for _ in range(arguments.iterations)
    )
    peak_rss = _max_rss_bytes()
    cold = timings[0]
    steady = timings[1:]
    collect_seconds = statistics.median(timing.collect_seconds for timing in steady)
    plan_seconds = statistics.median(timing.plan_seconds for timing in steady)
    preparation_seconds = statistics.median(timing.preparation_seconds for timing in steady)

    event_count = 0
    if arguments.dividends is DividendRepresentation.CASH:
        event_count = int(arguments.rows * 2)
    report = {
        "cold_collect_seconds": cold.collect_seconds,
        "cold_plan_seconds": cold.plan_seconds,
        "cold_preparation_seconds": cold.preparation_seconds,
        "collect_rows_per_second": arguments.rows / collect_seconds,
        "collect_seconds": collect_seconds,
        "dividend_representation": arguments.dividends.value,
        "events": event_count,
        "format": "finoptions-polars-boundary-benchmark-v2",
        "incremental_peak_rss_bytes": peak_rss - baseline_rss,
        "iterations": arguments.iterations,
        "numpy_version": np.__version__,
        "peak_max_rss_bytes": peak_rss,
        "plan_seconds": plan_seconds,
        "platform_machine": platform.machine(),
        "polars_version": pl.__version__,
        "preparation_rows_per_second": arguments.rows / preparation_seconds,
        "preparation_seconds": preparation_seconds,
        "python_version": platform.python_version(),
        "rows": arguments.rows,
        "scalar_zero_copy_verified": True,
        "steady_collect_seconds_minimum": min(timing.collect_seconds for timing in steady),
        "steady_plan_seconds_minimum": min(timing.plan_seconds for timing in steady),
        "steady_preparation_seconds_minimum": min(timing.preparation_seconds for timing in steady),
    }
    sys.stdout.write(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )


if __name__ == "__main__":
    main()
