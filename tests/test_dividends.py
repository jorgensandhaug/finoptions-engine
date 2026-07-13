"""Independent contracts for ragged cash-dividend model tensors."""

from __future__ import annotations

import numpy as np
import pytest
from finoptions import (
    BatchError,
    CashAmountScale,
    CpuExecutor,
    DividendBinSpec,
    InputError,
    MaturityBoundary,
    PaddedSequenceSpec,
    RaggedCashDividends,
    TimeBinClosure,
    bin_cash_dividends,
    pack_padded_cash_dividends,
)

_ROWS = 3
_PAD_CAPACITY = 3
_BIN_COUNT = 4


def inputs() -> tuple[RaggedCashDividends, np.ndarray, np.ndarray]:
    """Return three rows using nonzero Arrow-slice-compatible offsets."""
    dividends = RaggedCashDividends(
        offsets=np.array([7, 9, 9, 11], dtype=np.int64),
        times=np.array([0.25, 1.0, 0.25, 0.75], dtype=np.float64),
        amounts=np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64),
    )
    maturities = np.array([1.0, 0.5, 1.0], dtype=np.float64)
    spots = np.array([100.0, 50.0, 200.0], dtype=np.float64)
    return dividends, maturities, spots


def test_padded_sequences_match_an_independent_reference_and_expose_masks() -> None:
    """Sequence values agree with a separate scalar implementation and include validity."""
    dividends, maturities, spots = inputs()
    spec = PaddedSequenceSpec(
        capacity=3,
        maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
        amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
    )
    output = pack_padded_cash_dividends(
        dividends=dividends,
        maturities=maturities,
        spots=spots,
        spec=spec,
        executor=CpuExecutor(threads=1),
    )

    expected_times = np.zeros((_ROWS, _PAD_CAPACITY), dtype=np.float32)
    expected_amounts = np.zeros((_ROWS, _PAD_CAPACITY), dtype=np.float32)
    expected_mask = np.zeros((_ROWS, _PAD_CAPACITY), dtype=np.bool_)
    expected_lengths = np.array([2, 0, 2], dtype=np.uint32)
    schedules = (
        ((0.25, 1.0), (1.0, 2.0)),
        ((), ()),
        ((0.25, 0.75), (3.0, 1.0)),
    )
    for row, ((times, amounts), spot) in enumerate(zip(schedules, spots, strict=True)):
        for event, (time, amount) in enumerate(zip(times, amounts, strict=True)):
            expected_times[row, event] = np.float32(time)
            expected_amounts[row, event] = np.float32(amount / spot)
            expected_mask[row, event] = True

    assert output.rows == _ROWS
    assert output.capacity == _PAD_CAPACITY
    assert np.array_equal(output.times, expected_times)
    assert np.array_equal(output.amounts, expected_amounts)
    assert np.array_equal(output.mask, expected_mask)
    assert np.array_equal(output.lengths, expected_lengths)
    assert np.array_equal(output.mask.sum(axis=1), output.lengths)


@pytest.mark.parametrize(
    ("closure", "expected_first_row"),
    [
        (TimeBinClosure.LEFT_CLOSED_RIGHT_OPEN, [0.0, 1.0, 0.0, 2.0]),
        (TimeBinClosure.LEFT_OPEN_RIGHT_CLOSED, [1.0, 0.0, 0.0, 2.0]),
    ],
)
def test_bin_closure_is_an_exact_declared_contract(
    closure: TimeBinClosure,
    expected_first_row: list[float],
) -> None:
    """Events on exact bin edges obey the requested left/right closure."""
    dividends, maturities, _ = inputs()
    output = bin_cash_dividends(
        dividends=dividends,
        maturities=maturities,
        spec=DividendBinSpec(
            bins=_BIN_COUNT,
            horizon=1.0,
            closure=closure,
            maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
            amount_scale=CashAmountScale.ABSOLUTE,
        ),
        executor=CpuExecutor(threads=1),
    )
    assert output.rows == _ROWS
    assert output.bins == _BIN_COUNT
    assert np.array_equal(output.amounts[0], np.array(expected_first_row, dtype=np.float32))
    assert np.array_equal(output.amounts.sum(axis=1), np.array([3.0, 0.0, 4.0]))


def test_worker_count_never_changes_tensor_values() -> None:
    """Dedicated serial and parallel executors produce identical model tensors."""
    dividends, maturities, spots = inputs()
    sequence_spec = PaddedSequenceSpec(
        capacity=4,
        maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
        amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
    )
    bin_spec = DividendBinSpec(
        bins=12,
        horizon=1.0,
        closure=TimeBinClosure.LEFT_OPEN_RIGHT_CLOSED,
        maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
        amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
    )
    serial_sequence = pack_padded_cash_dividends(
        dividends=dividends,
        maturities=maturities,
        spots=spots,
        spec=sequence_spec,
        executor=CpuExecutor(threads=1),
    )
    parallel_sequence = pack_padded_cash_dividends(
        dividends=dividends,
        maturities=maturities,
        spots=spots,
        spec=sequence_spec,
        executor=CpuExecutor(threads=4),
    )
    assert np.array_equal(serial_sequence.times, parallel_sequence.times)
    assert np.array_equal(serial_sequence.amounts, parallel_sequence.amounts)
    assert np.array_equal(serial_sequence.mask, parallel_sequence.mask)
    assert np.array_equal(serial_sequence.lengths, parallel_sequence.lengths)

    serial_bins = bin_cash_dividends(
        dividends=dividends,
        maturities=maturities,
        spots=spots,
        spec=bin_spec,
        executor=CpuExecutor(threads=1),
    )
    parallel_bins = bin_cash_dividends(
        dividends=dividends,
        maturities=maturities,
        spots=spots,
        spec=bin_spec,
        executor=CpuExecutor(threads=4),
    )
    assert np.array_equal(serial_bins.amounts, parallel_bins.amounts)


def test_mutable_numpy_sources_are_revalidated_at_every_native_call() -> None:
    """Mutation after construction cannot bypass native financial validation."""
    dividends, maturities, _ = inputs()
    dividends.times[1] = 1.5
    with pytest.raises(BatchError, match="after maturity"):
        pack_padded_cash_dividends(
            dividends=dividends,
            maturities=maturities,
            spec=PaddedSequenceSpec(
                capacity=4,
                maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
                amount_scale=CashAmountScale.ABSOLUTE,
            ),
            executor=CpuExecutor(threads=1),
        )


def test_invalid_layout_units_boundaries_capacity_and_precision_fail_explicitly() -> None:
    """Every unsupported layout, unit, boundary, capacity, and cast raises early."""
    with pytest.raises(InputError, match="dtype int64"):
        RaggedCashDividends(
            offsets=np.array([0, 1], dtype=np.int32),  # type: ignore[arg-type]
            times=np.array([0.25], dtype=np.float64),
            amounts=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(BatchError, match="nondecreasing"):
        RaggedCashDividends(
            offsets=np.array([0, 2], dtype=np.int64),
            times=np.array([0.5, 0.25], dtype=np.float64),
            amounts=np.array([1.0, 2.0], dtype=np.float64),
        )

    dividends, maturities, spots = inputs()
    absolute = PaddedSequenceSpec(
        capacity=4,
        maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
        amount_scale=CashAmountScale.ABSOLUTE,
    )
    with pytest.raises(InputError, match="spots must be None"):
        pack_padded_cash_dividends(
            dividends=dividends,
            maturities=maturities,
            spots=spots,
            spec=absolute,
            executor=CpuExecutor(threads=1),
        )
    with pytest.raises(InputError, match="spots are required"):
        pack_padded_cash_dividends(
            dividends=dividends,
            maturities=maturities,
            spec=PaddedSequenceSpec(
                capacity=4,
                maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
                amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
            ),
            executor=CpuExecutor(threads=1),
        )
    with pytest.raises(BatchError, match="exceeding requested capacity"):
        pack_padded_cash_dividends(
            dividends=dividends,
            maturities=maturities,
            spec=PaddedSequenceSpec(
                capacity=1,
                maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
                amount_scale=CashAmountScale.ABSOLUTE,
            ),
            executor=CpuExecutor(threads=1),
        )
    with pytest.raises(BatchError, match="strictly before maturity"):
        pack_padded_cash_dividends(
            dividends=dividends,
            maturities=maturities,
            spec=PaddedSequenceSpec(
                capacity=4,
                maturity_boundary=MaturityBoundary.STRICTLY_BEFORE,
                amount_scale=CashAmountScale.ABSOLUTE,
            ),
            executor=CpuExecutor(threads=1),
        )

    tiny = RaggedCashDividends(
        offsets=np.array([0, 1], dtype=np.int64),
        times=np.array([np.finfo(np.float64).tiny], dtype=np.float64),
        amounts=np.array([1.0], dtype=np.float64),
    )
    with pytest.raises(BatchError, match="safely represented as float32"):
        pack_padded_cash_dividends(
            dividends=tiny,
            maturities=np.array([1.0], dtype=np.float64),
            spec=absolute,
            executor=CpuExecutor(threads=1),
        )
