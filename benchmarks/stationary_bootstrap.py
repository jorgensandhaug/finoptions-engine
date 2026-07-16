"""Measure the public additive-panel stationary-bootstrap boundary."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from finoptions import CpuExecutor, RandomStream, StationaryBootstrap


def main() -> None:
    """Run one deterministic matrix across declared geometric mean lengths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=int, default=4_000)
    parser.add_argument("--statistics", type=int, default=25)
    parser.add_argument("--replications", type=int, default=19_999)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--strata", type=int, default=1)
    parser.add_argument("--mean-block-length", type=float, action="append", required=True)
    arguments = parser.parse_args()

    values = np.ascontiguousarray(
        np.random.default_rng(7).normal(size=(arguments.panels, arguments.statistics)),
        dtype=np.float64,
    )
    executor = CpuExecutor(threads=arguments.threads)
    if not 1 <= arguments.strata <= arguments.panels:
        msg = "strata must be in [1, panels]"
        raise ValueError(msg)
    offsets = np.linspace(0, arguments.panels, arguments.strata + 1, dtype=np.uint64)
    if np.unique(offsets).size != offsets.size:
        msg = "strata produce an empty integer panel range"
        raise ValueError(msg)
    for index, mean_block_length in enumerate(arguments.mean_block_length):
        request = StationaryBootstrap(
            replications=arguments.replications,
            mean_block_length=mean_block_length,
            random_stream=RandomStream(seed=0x13198A2E03707344, domain=2_000_000 + index),
        )
        started = time.perf_counter()
        result = (
            request.sum_panels(values, executor=executor)
            if arguments.strata == 1
            else request.sum_stratified_panels(values, offsets, executor=executor)
        )
        elapsed = time.perf_counter() - started
        sys.stdout.write(
            json.dumps(
                {
                    "checksum": float(np.sum(result, dtype=np.float64)),
                    "elapsed_seconds": elapsed,
                    "mean_block_length": mean_block_length,
                    "panels": arguments.panels,
                    "replications": arguments.replications,
                    "statistics": arguments.statistics,
                    "strata": arguments.strata,
                    "thread_count": arguments.threads,
                },
                sort_keys=True,
            )
            + "\n",
        )


if __name__ == "__main__":
    main()
