//! Borrowed, validated Arrow-style ragged cash-dividend schedules.

use std::sync::atomic::{AtomicUsize, Ordering};

use finoptions_runtime::CpuExecutor;
use rayon::prelude::*;

use crate::BatchError;

/// One borrowed row of exact `f64` future cash-dividend data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CashDividendRow<'a> {
    times: &'a [f64],
    amounts: &'a [f64],
}

impl<'a> CashDividendRow<'a> {
    /// Borrows nondecreasing, finite, strictly positive event year fractions.
    #[must_use]
    pub const fn times(self) -> &'a [f64] {
        self.times
    }

    /// Borrows aligned, finite, canonical nonnegative cash amounts.
    #[must_use]
    pub const fn amounts(self) -> &'a [f64] {
        self.amounts
    }

    /// Returns this row's event count.
    #[must_use]
    pub const fn len(self) -> usize {
        self.times.len()
    }

    /// Returns whether this row has no future cash dividends.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.times.is_empty()
    }
}

/// Validated zero-copy view over Arrow `LargeList`-compatible offset and value slices.
///
/// `offsets` may begin at a nonzero base, as occurs for a sliced Arrow array. The supplied child
/// slices must span exactly `offsets[0]..offsets[-1]`; no normalization or value copy is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaggedCashDividends<'a> {
    offsets: &'a [i64],
    times: &'a [f64],
    amounts: &'a [f64],
    base: usize,
}

impl<'a> RaggedCashDividends<'a> {
    /// Validates structural and per-event financial invariants without copying source values.
    ///
    /// Duplicate event times are retained because one ex-date may contain multiple distinct cash
    /// components. Times must otherwise be nondecreasing within each row.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed offsets, mismatched columns, invalid future times,
    /// unordered rows, or invalid deterministic cash amounts.
    pub fn try_new(
        offsets: &'a [i64],
        times: &'a [f64],
        amounts: &'a [f64],
    ) -> Result<Self, BatchError> {
        let view = Self::try_new_structure(offsets, times, amounts)?;
        view.validate_events_serial()?;
        Ok(view)
    }

    /// Validates the identical zero-copy view using an explicitly selected execution resource.
    ///
    /// Structural offset checks remain ordered and serial. Per-row financial event checks use the
    /// executor's serial or dedicated-pool kernel and deterministically report the lowest failing
    /// row.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as [`Self::try_new`]. No global pool or alternate worker count
    /// is used.
    pub fn try_new_with_executor(
        offsets: &'a [i64],
        times: &'a [f64],
        amounts: &'a [f64],
        executor: &CpuExecutor,
    ) -> Result<Self, BatchError> {
        let view = Self::try_new_structure(offsets, times, amounts)?;
        executor.dispatch(
            || view.validate_events_serial(),
            || view.validate_events_parallel(),
        )?;
        Ok(view)
    }

    fn try_new_structure(
        offsets: &'a [i64],
        times: &'a [f64],
        amounts: &'a [f64],
    ) -> Result<Self, BatchError> {
        if offsets.len() < 2 {
            return Err(BatchError::InsufficientOffsets {
                observed: offsets.len(),
            });
        }
        if times.len() != amounts.len() {
            return Err(BatchError::EventColumnLengthMismatch {
                times: times.len(),
                amounts: amounts.len(),
            });
        }
        for (index, &offset) in offsets.iter().enumerate() {
            if offset < 0 {
                return Err(BatchError::NegativeOffset {
                    index,
                    observed: offset,
                });
            }
            if index > 0 && offsets[index - 1] > offset {
                return Err(BatchError::DecreasingOffset {
                    previous_index: index - 1,
                    previous: offsets[index - 1],
                    index,
                    observed: offset,
                });
            }
        }

        let first = offsets[0];
        let Some(&last) = offsets.last() else {
            return Err(BatchError::InsufficientOffsets { observed: 0 });
        };
        let base = usize::try_from(first).map_err(|_| BatchError::OffsetIndexOverflow {
            index: 0,
            observed: first,
        })?;
        let final_index = usize::try_from(last).map_err(|_| BatchError::OffsetIndexOverflow {
            index: offsets.len() - 1,
            observed: last,
        })?;
        let offset_events = final_index
            .checked_sub(base)
            .ok_or(BatchError::DecreasingOffset {
                previous_index: 0,
                previous: first,
                index: offsets.len() - 1,
                observed: last,
            })?;
        if offset_events != times.len() {
            return Err(BatchError::OffsetSpanMismatch {
                first,
                last,
                offset_events,
                buffer_events: times.len(),
            });
        }

        Ok(Self {
            offsets,
            times,
            amounts,
            base,
        })
    }

    fn validate_events_serial(self) -> Result<(), BatchError> {
        for row in 0..self.rows() {
            self.validate_event_row(row)?;
        }
        Ok(())
    }

    fn validate_events_parallel(self) -> Result<(), BatchError> {
        let first_failure = AtomicUsize::new(usize::MAX);
        (0..self.rows()).into_par_iter().for_each(|row| {
            if row > first_failure.load(Ordering::Relaxed) {
                return;
            }
            if self.validate_event_row(row).is_err() {
                first_failure.fetch_min(row, Ordering::Relaxed);
            }
        });
        let failed_row = first_failure.load(Ordering::Relaxed);
        if failed_row == usize::MAX {
            return Ok(());
        }
        match self.validate_event_row(failed_row) {
            Err(error) => Err(error),
            Ok(()) => Err(BatchError::ParallelFailureReplayMismatch { row: failed_row }),
        }
    }

    fn validate_event_row(self, row: usize) -> Result<(), BatchError> {
        let schedule = self.row_unchecked(row);
        for (event, (&time, &amount)) in schedule.times.iter().zip(schedule.amounts).enumerate() {
            if !time.is_finite() || time <= 0.0 {
                return Err(BatchError::InvalidEventTime {
                    row,
                    event,
                    observed: time,
                });
            }
            if event > 0 && schedule.times[event - 1] > time {
                return Err(BatchError::DecreasingEventTime {
                    row,
                    event,
                    previous: schedule.times[event - 1],
                    observed: time,
                });
            }
            if !amount.is_finite() || amount < 0.0 || (amount == 0.0 && amount.is_sign_negative()) {
                return Err(BatchError::InvalidCashAmount {
                    row,
                    event,
                    observed: amount,
                });
            }
        }
        Ok(())
    }

    /// Returns the number of option rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.offsets.len() - 1
    }

    /// Returns the total number of cash events in the borrowed child slices.
    #[must_use]
    pub const fn events(self) -> usize {
        self.times.len()
    }

    /// Borrows one validated row, or returns `None` when `row >= self.rows()`.
    #[must_use]
    pub fn row(self, row: usize) -> Option<CashDividendRow<'a>> {
        (row < self.rows()).then(|| self.row_unchecked(row))
    }

    pub(crate) fn row_unchecked(self, row: usize) -> CashDividendRow<'a> {
        let start = usize::try_from(self.offsets[row]).expect("validated offset") - self.base;
        let end = usize::try_from(self.offsets[row + 1]).expect("validated offset") - self.base;
        CashDividendRow {
            times: &self.times[start..end],
            amounts: &self.amounts[start..end],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrows_zero_based_and_sliced_arrow_offsets_without_copying() {
        let zero = RaggedCashDividends::try_new(&[0, 2, 2, 3], &[0.1, 0.2, 0.4], &[1.0, 2.0, 3.0])
            .unwrap();
        let sliced =
            RaggedCashDividends::try_new(&[7, 9, 9, 10], &[0.1, 0.2, 0.4], &[1.0, 2.0, 3.0])
                .unwrap();
        assert_eq!(zero.rows(), 3);
        assert_eq!(zero.events(), 3);
        assert_eq!(zero.row(0), sliced.row(0));
        assert!(zero.row(1).unwrap().is_empty());
        assert_eq!(zero.row(2).unwrap().amounts(), &[3.0]);
        assert_eq!(zero.row(3), None);
    }

    #[test]
    fn rejects_malformed_offsets_before_indexing_children() {
        assert!(matches!(
            RaggedCashDividends::try_new(&[], &[], &[]),
            Err(BatchError::InsufficientOffsets { observed: 0 })
        ));
        assert!(matches!(
            RaggedCashDividends::try_new(&[-1, 0], &[0.1], &[1.0]),
            Err(BatchError::NegativeOffset { index: 0, .. })
        ));
        assert!(matches!(
            RaggedCashDividends::try_new(&[0, 2, 1], &[0.1], &[1.0]),
            Err(BatchError::DecreasingOffset { index: 2, .. })
        ));
        assert!(matches!(
            RaggedCashDividends::try_new(&[4, 6], &[0.1], &[1.0]),
            Err(BatchError::OffsetSpanMismatch { .. })
        ));
    }

    #[test]
    fn validates_financial_event_values_and_retains_duplicate_times() {
        let duplicates = RaggedCashDividends::try_new(&[0, 2], &[0.25, 0.25], &[1.0, 2.0]).unwrap();
        assert_eq!(duplicates.row(0).unwrap().len(), 2);

        for (times, amounts) in [
            (vec![0.0], vec![1.0]),
            (vec![-0.0], vec![1.0]),
            (vec![f64::NAN], vec![1.0]),
            (vec![0.25], vec![-0.0]),
            (vec![0.25], vec![-1.0]),
            (vec![0.25], vec![f64::INFINITY]),
        ] {
            assert!(RaggedCashDividends::try_new(&[0, 1], &times, &amounts).is_err());
        }
        assert!(matches!(
            RaggedCashDividends::try_new(&[0, 2], &[0.5, 0.25], &[1.0, 2.0]),
            Err(BatchError::DecreasingEventTime {
                row: 0,
                event: 1,
                ..
            })
        ));
    }

    #[test]
    fn parallel_validation_reports_the_lowest_invalid_row_deterministically() {
        let executor = CpuExecutor::try_new(4).unwrap();
        let error = RaggedCashDividends::try_new_with_executor(
            &[0, 1, 3],
            &[0.0, 0.5, 0.25],
            &[1.0, 1.0, 1.0],
            &executor,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchError::InvalidEventTime {
                row: 0,
                event: 0,
                ..
            }
        ));
    }
}
