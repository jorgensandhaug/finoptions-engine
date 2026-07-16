"""Measure the compact Polars-to-native grouped LSM preparation boundary."""

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
from finoptions import (
    CashDividendBatch,
    ContinuousDividendYieldBatch,
    DividendPolicy,
    LsmStrikeGroups,
    MaturityBoundary,
)
from finoptions.polars import (
    BufferPolicy,
    CashDividendColumns,
    ContinuousYieldColumn,
    DividendColumns,
    LsmStrikeGroupColumns,
    prepare_lsm_strike_groups,
    select_lsm_strike_group_inputs,
)

_DEFAULT_GROUPS: Final = 100_000
_DEFAULT_STRIKE_ROWS: Final = 1_000_000
_DEFAULT_ITERATIONS: Final = 6
_MINIMUM_ITERATIONS: Final = 2
_UINT32_MAX: Final = (2**32) - 1


class DividendRepresentation(StrEnum):
    """Synthetic group-level dividend input admitted by this benchmark."""

    CONTINUOUS_YIELD = "continuous_yield"
    CASH = "cash"


@dataclass(frozen=True, slots=True)
class BenchmarkArguments:
    """Validated compact-axis size and representation request."""

    groups: int
    strike_rows: int
    dividends: DividendRepresentation
    iterations: int


@dataclass(frozen=True, slots=True)
class BoundaryTiming:
    """One projection, explicit collection, and native preparation observation."""

    plan_seconds: float
    collect_seconds: float
    preparation_seconds: float


@dataclass(frozen=True, slots=True)
class BoundaryMeasurement:
    """One timing plus the verified compact native request size."""

    timing: BoundaryTiming
    native_request_bytes: int


@dataclass(frozen=True, slots=True)
class SyntheticAxes:
    """Materialized compact state, strike, and boundary sources."""

    states: pl.DataFrame
    strikes: pl.DataFrame
    offsets: pl.Series


def _max_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1_024


def _arguments() -> BenchmarkArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=_DEFAULT_GROUPS)
    parser.add_argument("--strike-rows", type=int, default=_DEFAULT_STRIKE_ROWS)
    parser.add_argument("--iterations", type=int, default=_DEFAULT_ITERATIONS)
    parser.add_argument(
        "--dividends",
        choices=tuple(representation.value for representation in DividendRepresentation),
        default=DividendRepresentation.CONTINUOUS_YIELD.value,
    )
    parsed = parser.parse_args()
    if type(parsed.groups) is not int or parsed.groups <= 0:
        message = "groups must be a positive plain integer"
        raise ValueError(message)
    if (
        type(parsed.strike_rows) is not int
        or not parsed.groups <= parsed.strike_rows <= _UINT32_MAX
    ):
        message = "strike-rows must be a plain integer in [groups, 2^32 - 1]"
        raise ValueError(message)
    if type(parsed.iterations) is not int or parsed.iterations < _MINIMUM_ITERATIONS:
        message = "iterations must be a plain integer of at least two"
        raise ValueError(message)
    return BenchmarkArguments(
        groups=parsed.groups,
        strike_rows=parsed.strike_rows,
        dividends=DividendRepresentation(parsed.dividends),
        iterations=parsed.iterations,
    )


def _sources(arguments: BenchmarkArguments) -> SyntheticAxes:
    group_row = np.arange(arguments.groups, dtype=np.float64)
    strike_row = np.arange(arguments.strike_rows, dtype=np.float64)
    state_columns: dict[str, pl.Series | np.ndarray] = {
        "state_id": np.arange(arguments.groups, dtype=np.uint64),
        "spot": np.ascontiguousarray(80.0 + np.remainder(group_row, 12_001.0) / 100.0),
        "maturity": np.ascontiguousarray(0.05 + np.remainder(group_row, 340.0) / 365.0),
        "rate": np.ascontiguousarray(0.01 + np.remainder(group_row, 400.0) / 10_000.0),
        "volatility": np.ascontiguousarray(0.12 + np.remainder(group_row, 600.0) / 2_000.0),
        "unused_state_feature": np.ascontiguousarray(np.remainder(group_row, 997.0)),
    }
    if arguments.dividends is DividendRepresentation.CONTINUOUS_YIELD:
        state_columns["dividend_yield"] = np.ascontiguousarray(
            0.005 + np.remainder(group_row, 300.0) / 10_000.0,
        )
    else:
        state_columns["dividend_times"] = pl.repeat(
            pl.lit([0.20, 0.45], dtype=pl.List(pl.Float64)),
            n=arguments.groups,
            eager=True,
        ).alias("dividend_times")
        state_columns["dividend_amounts"] = pl.repeat(
            pl.lit([0.50, 0.55], dtype=pl.List(pl.Float64)),
            n=arguments.groups,
            eager=True,
        ).alias("dividend_amounts")
    state_frame = pl.DataFrame(state_columns)
    strike_frame = pl.DataFrame(
        {
            "source_row": np.arange(arguments.strike_rows, dtype=np.uint64),
            "strike": np.ascontiguousarray(75.0 + np.remainder(strike_row, 13_001.0) / 100.0),
            "unused_strike_feature": np.ascontiguousarray(np.remainder(strike_row, 991.0)),
        }
    )
    sizes = np.full(
        arguments.groups,
        arguments.strike_rows // arguments.groups,
        dtype=np.uint32,
    )
    sizes[: arguments.strike_rows % arguments.groups] += np.uint32(1)
    offsets = np.empty(arguments.groups + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(sizes, dtype=np.uint32, out=offsets[1:])
    return SyntheticAxes(
        states=state_frame,
        strikes=strike_frame,
        offsets=pl.Series("group_offsets", offsets),
    )


def _measure_boundary(
    axes: SyntheticAxes,
    *,
    columns: LsmStrikeGroupColumns,
    dividends: DividendColumns,
    arguments: BenchmarkArguments,
) -> BoundaryMeasurement:
    plan_started = perf_counter()
    state_plan, strike_plan = select_lsm_strike_group_inputs(
        axes.states.lazy(),
        axes.strikes.lazy(),
        state_columns=columns,
        strike_column="strike",
        dividends=dividends,
    )
    plan_seconds = perf_counter() - plan_started
    collect_started = perf_counter()
    state_frame, strike_frame = pl.collect_all((state_plan, strike_plan))
    collect_seconds = perf_counter() - collect_started
    preparation_started = perf_counter()
    prepared = prepare_lsm_strike_groups(
        state_frame,
        strike_frame,
        group_offsets=axes.offsets,
        state_columns=columns,
        strike_column="strike",
        dividends=dividends,
        buffer_policy=BufferPolicy.REQUIRE_ZERO_COPY,
    )
    preparation_seconds = perf_counter() - preparation_started

    if prepared.group_count != arguments.groups or prepared.rows != arguments.strike_rows:
        message = "prepared grouped boundary changed a declared axis length"
        raise RuntimeError(message)
    _require_borrowed(
        state_frame.get_column("spot"),
        prepared.native.initial_spots,
        label="group-state spot",
    )
    _require_borrowed(
        strike_frame.get_column("strike"),
        prepared.native.strikes,
        label="strike",
    )
    _require_borrowed(axes.offsets, prepared.native.group_offsets, label="group offsets")
    native_request_bytes = _native_request_bytes(prepared.native)
    return BoundaryMeasurement(
        timing=BoundaryTiming(
            plan_seconds=plan_seconds,
            collect_seconds=collect_seconds,
            preparation_seconds=preparation_seconds,
        ),
        native_request_bytes=native_request_bytes,
    )


def _require_borrowed(series: pl.Series, array: np.ndarray, *, label: str) -> None:
    materialized = series.to_numpy(writable=False, allow_copy=False)
    if not np.shares_memory(materialized, array):
        message = f"prepared {label} does not borrow the materialized Polars buffer"
        raise RuntimeError(message)
    if array.flags.writeable:
        message = f"prepared {label} unexpectedly permits mutation"
        raise RuntimeError(message)


def _native_request_bytes(groups: LsmStrikeGroups) -> int:
    native = groups
    arrays = (
        native.group_offsets,
        native.initial_spots,
        native.strikes,
        native.maturities,
        native.rates,
        native.volatilities,
    )
    total = sum(array.nbytes for array in arrays)
    dividends = native.dividends
    if type(dividends) is CashDividendBatch:
        schedules = dividends.schedules
        total += schedules.offsets.nbytes + schedules.times.nbytes + schedules.amounts.nbytes
    elif type(dividends) is ContinuousDividendYieldBatch:
        total += dividends.annualized_decimals.nbytes
    else:
        message = "benchmark admitted an unsupported dividend request"
        raise RuntimeError(message)
    return int(total)


def main() -> None:
    """Run one fresh-process grouped boundary measurement and emit canonical JSON."""
    arguments = _arguments()
    axes = _sources(arguments)
    columns = LsmStrikeGroupColumns(
        initial_spot="spot",
        maturity="maturity",
        rate="rate",
        volatility="volatility",
    )
    if arguments.dividends is DividendRepresentation.CONTINUOUS_YIELD:
        dividends: DividendColumns = ContinuousYieldColumn("dividend_yield")
    else:
        dividends = CashDividendColumns(
            times="dividend_times",
            amounts="dividend_amounts",
            policy=DividendPolicy.LIQUIDATOR,
            maturity_boundary=MaturityBoundary.STRICTLY_BEFORE,
        )

    gc.collect()
    baseline_rss = _max_rss_bytes()
    measurements = tuple(
        _measure_boundary(
            axes,
            columns=columns,
            dividends=dividends,
            arguments=arguments,
        )
        for _ in range(arguments.iterations)
    )
    peak_rss = _max_rss_bytes()
    cold = measurements[0].timing
    steady = tuple(measurement.timing for measurement in measurements[1:])
    collect_seconds = statistics.median(timing.collect_seconds for timing in steady)
    plan_seconds = statistics.median(timing.plan_seconds for timing in steady)
    preparation_seconds = statistics.median(timing.preparation_seconds for timing in steady)
    native_sizes = {measurement.native_request_bytes for measurement in measurements}
    if len(native_sizes) != 1:
        message = "native request size changed across identical iterations"
        raise RuntimeError(message)
    report = {
        "cold_collect_seconds": cold.collect_seconds,
        "cold_plan_seconds": cold.plan_seconds,
        "cold_preparation_seconds": cold.preparation_seconds,
        "collect_strike_rows_per_second": arguments.strike_rows / collect_seconds,
        "collect_seconds": collect_seconds,
        "dividend_representation": arguments.dividends.value,
        "events": (
            0
            if arguments.dividends is DividendRepresentation.CONTINUOUS_YIELD
            else arguments.groups * 2
        ),
        "format": "finoptions-polars-grouped-boundary-benchmark-v1",
        "groups": arguments.groups,
        "incremental_peak_rss_bytes": peak_rss - baseline_rss,
        "iterations": arguments.iterations,
        "native_request_bytes": native_sizes.pop(),
        "numpy_version": np.__version__,
        "peak_max_rss_bytes": peak_rss,
        "plan_seconds": plan_seconds,
        "platform_machine": platform.machine(),
        "polars_version": pl.__version__,
        "preparation_groups_per_second": arguments.groups / preparation_seconds,
        "preparation_seconds": preparation_seconds,
        "preparation_strike_rows_per_second": arguments.strike_rows / preparation_seconds,
        "python_version": platform.python_version(),
        "scalar_and_offset_zero_copy_verified": True,
        "state_rows_avoided": arguments.strike_rows - arguments.groups,
        "steady_collect_seconds_minimum": min(timing.collect_seconds for timing in steady),
        "steady_plan_seconds_minimum": min(timing.plan_seconds for timing in steady),
        "steady_preparation_seconds_minimum": min(timing.preparation_seconds for timing in steady),
        "strike_rows": arguments.strike_rows,
    }
    sys.stdout.write(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
