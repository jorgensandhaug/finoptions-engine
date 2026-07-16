"""Native stationary-bootstrap boundary and statistical-invariant tests."""

from __future__ import annotations

import numpy as np
import pytest
from finoptions import (
    CpuExecutor,
    InputError,
    RandomStream,
    StationaryBootstrap,
)


def bootstrap(*, replications: int = 1_003, mean_block_length: float = 5.0) -> StationaryBootstrap:
    """Build one explicit test namespace."""
    return StationaryBootstrap(
        replications=replications,
        mean_block_length=mean_block_length,
        random_stream=RandomStream(seed=0x13198A2E03707344, domain=2_000_000),
    )


def test_each_draw_contains_exactly_the_observed_panel_count() -> None:
    """A constant sufficient statistic reveals dropped or duplicated sequence positions."""
    values = np.ones((17, 3), dtype=np.float64)
    for mean in (1.0, 5.0, 20.0, 60.0):
        result = bootstrap(mean_block_length=mean).sum_panels(
            values,
            executor=CpuExecutor(threads=4),
        )
        assert result.shape == (1_003, 3)
        assert np.array_equal(result, np.full((1_003, 3), 17.0))
        assert not result.flags.writeable


def test_random_identity_is_bitwise_invariant_to_executor_width() -> None:
    """Path-addressed replications do not depend on Rayon scheduling."""
    values = np.arange(1, 59, dtype=np.float64).reshape(29, 2)
    request = bootstrap(replications=2_003, mean_block_length=10.0)
    serial = request.sum_panels(values, executor=CpuExecutor(threads=1))
    parallel = request.sum_panels(values, executor=CpuExecutor(threads=4))
    assert np.array_equal(serial, parallel)
    assert np.unique(serial[:, 0]).size > 1


def test_stratified_draws_preserve_each_source_stratum_and_thread_identity() -> None:
    """Circular blocks never cross declared stratum boundaries or alter stratum weights."""
    values = np.array(
        [
            [1.0, 10.0, 1.0],
            [1.0, 10.0, 2.0],
            [1.0, 100.0, 4.0],
            [1.0, 100.0, 8.0],
            [1.0, 100.0, 16.0],
            [1.0, 100.0, 32.0],
        ],
        dtype=np.float64,
    )
    offsets = np.array([0, 2, 6], dtype=np.uint64)
    request = bootstrap(replications=2_003, mean_block_length=3.0)
    serial = request.sum_stratified_panels(
        values,
        offsets,
        executor=CpuExecutor(threads=1),
    )
    parallel = request.sum_stratified_panels(
        values,
        offsets,
        executor=CpuExecutor(threads=4),
    )
    assert np.array_equal(serial, parallel)
    assert np.array_equal(serial[:, 0], np.full(2_003, 6.0))
    assert np.array_equal(serial[:, 1], np.full(2_003, 420.0))
    assert np.unique(serial[:, 2]).size > 1
    assert not serial.flags.writeable


def test_rejects_implicit_conversion_nonfinite_values_and_wrapping_paths() -> None:
    """No dtype/layout repair, row removal, or counter wrapping is implicit."""
    request = bootstrap()
    with pytest.raises(InputError, match="dtype float64"):
        request.sum_panels(
            np.ones((3, 2), dtype=np.float32),
            executor=CpuExecutor(threads=1),
        )
    with pytest.raises(InputError, match="C-contiguous"):
        request.sum_panels(
            np.ones((3, 2), dtype=np.float64).T,
            executor=CpuExecutor(threads=1),
        )
    invalid = np.ones((3, 2), dtype=np.float64)
    invalid[1, 0] = np.nan
    with pytest.raises(InputError, match="panel 1 statistic 0"):
        request.sum_panels(invalid, executor=CpuExecutor(threads=1))
    with pytest.raises(InputError, match="complete replication range"):
        StationaryBootstrap(
            replications=2,
            mean_block_length=5.0,
            random_stream=RandomStream(seed=1, domain=2),
            replication_path_start=(2**32) - 1,
        )


def test_rejects_invalid_stratum_offset_buffers_and_partitions() -> None:
    """Stratified execution never repairs dtype, layout, gaps, overlaps, or empty strata."""
    request = bootstrap()
    values = np.ones((3, 2), dtype=np.float64)
    executor = CpuExecutor(threads=1)
    with pytest.raises(InputError, match="dtype uint64"):
        request.sum_stratified_panels(
            values,
            np.array([0, 3], dtype=np.int64),
            executor=executor,
        )
    with pytest.raises(InputError, match="C-contiguous"):
        request.sum_stratified_panels(
            values,
            np.array([0, 1, 2, 3], dtype=np.uint64)[::2],
            executor=executor,
        )
    with pytest.raises(InputError, match="strictly increasing"):
        request.sum_stratified_panels(
            values,
            np.array([0, 2, 2, 3], dtype=np.uint64),
            executor=executor,
        )
    with pytest.raises(InputError, match="expected panel count 3"):
        request.sum_stratified_panels(
            values,
            np.array([0, 2], dtype=np.uint64),
            executor=executor,
        )
