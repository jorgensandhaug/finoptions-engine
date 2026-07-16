//! Borrowed, validated Arrow-style ragged explicit exercise times.

use std::sync::atomic::{AtomicUsize, Ordering};

use finoptions_runtime::CpuExecutor;
use rayon::prelude::*;

use crate::BatchError;

/// One borrowed row of exact `f64` exercise opportunities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExerciseTimeRow<'a> {
    times: &'a [f64],
}

impl<'a> ExerciseTimeRow<'a> {
    /// Borrows strictly increasing, finite, strictly positive year fractions.
    #[must_use]
    pub const fn times(self) -> &'a [f64] {
        self.times
    }

    /// Returns this row's exercise-opportunity count.
    #[must_use]
    pub const fn len(self) -> usize {
        self.times.len()
    }

    /// Returns whether this row contains no pre-expiry exercise opportunity.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.times.is_empty()
    }
}

/// Validated zero-copy view over Arrow `LargeList`-compatible exercise-time buffers.
///
/// `offsets` may begin above zero when the supplied values are the corresponding child slice from
/// a sliced Arrow array. Times are exact contract inputs; no sorting, deduplication, snapping, or
/// calendar inference occurs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaggedExerciseTimes<'a> {
    offsets: &'a [i64],
    times: &'a [f64],
    base: usize,
}

impl<'a> RaggedExerciseTimes<'a> {
    /// Validates structural and per-row chronological invariants without copying source values.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed offsets, invalid positive times, or any duplicate or
    /// decreasing time within a row.
    pub fn try_new(offsets: &'a [i64], times: &'a [f64]) -> Result<Self, BatchError> {
        let view = Self::try_new_structure(offsets, times)?;
        view.validate_rows_serial()?;
        Ok(view)
    }

    /// Validates the same zero-copy view on an explicitly selected CPU resource.
    ///
    /// # Errors
    ///
    /// Returns the same deterministic lowest-row error as [`Self::try_new`].
    pub fn try_new_with_executor(
        offsets: &'a [i64],
        times: &'a [f64],
        executor: &CpuExecutor,
    ) -> Result<Self, BatchError> {
        let view = Self::try_new_structure(offsets, times)?;
        executor.dispatch(
            || view.validate_rows_serial(),
            || view.validate_rows_parallel(),
        )?;
        Ok(view)
    }

    fn try_new_structure(offsets: &'a [i64], times: &'a [f64]) -> Result<Self, BatchError> {
        if offsets.len() < 2 {
            return Err(BatchError::InsufficientOffsets {
                observed: offsets.len(),
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
        let last = offsets[offsets.len() - 1];
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
            base,
        })
    }

    fn validate_rows_serial(self) -> Result<(), BatchError> {
        for row in 0..self.rows() {
            self.validate_row(row)?;
        }
        Ok(())
    }

    fn validate_rows_parallel(self) -> Result<(), BatchError> {
        let first_failure = AtomicUsize::new(usize::MAX);
        (0..self.rows()).into_par_iter().for_each(|row| {
            if row <= first_failure.load(Ordering::Relaxed) && self.validate_row(row).is_err() {
                first_failure.fetch_min(row, Ordering::Relaxed);
            }
        });
        let failed_row = first_failure.load(Ordering::Relaxed);
        if failed_row == usize::MAX {
            return Ok(());
        }
        match self.validate_row(failed_row) {
            Err(error) => Err(error),
            Ok(()) => Err(BatchError::ParallelFailureReplayMismatch { row: failed_row }),
        }
    }

    fn validate_row(self, row: usize) -> Result<(), BatchError> {
        let times = self.row_unchecked(row).times;
        for (event, &time) in times.iter().enumerate() {
            if !time.is_finite() || time <= 0.0 {
                return Err(BatchError::InvalidExerciseTime {
                    row,
                    event,
                    observed: time,
                });
            }
            if event > 0 && times[event - 1] >= time {
                return Err(BatchError::NonIncreasingExerciseTime {
                    row,
                    event,
                    previous: times[event - 1],
                    observed: time,
                });
            }
        }
        Ok(())
    }

    /// Returns the number of exercise-schedule rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.offsets.len() - 1
    }

    /// Returns the total number of explicit exercise opportunities.
    #[must_use]
    pub const fn events(self) -> usize {
        self.times.len()
    }

    /// Borrows one validated row, or returns `None` when the row is out of range.
    #[must_use]
    pub fn row(self, row: usize) -> Option<ExerciseTimeRow<'a>> {
        (row < self.rows()).then(|| self.row_unchecked(row))
    }

    fn row_unchecked(self, row: usize) -> ExerciseTimeRow<'a> {
        let start = usize::try_from(self.offsets[row]).expect("validated offset") - self.base;
        let end = usize::try_from(self.offsets[row + 1]).expect("validated offset") - self.base;
        ExerciseTimeRow {
            times: &self.times[start..end],
        }
    }
}

#[cfg(test)]
mod tests {
    use finoptions_runtime::CpuExecutor;

    use super::*;

    #[test]
    fn borrows_sliced_offsets_without_normalizing_or_copying() {
        let times = [0.25, 0.75];
        let schedules = RaggedExerciseTimes::try_new(&[4, 5, 6], &times).unwrap();
        assert_eq!(schedules.rows(), 2);
        assert_eq!(schedules.events(), 2);
        assert_eq!(schedules.row(0).unwrap().times(), &[0.25]);
        assert_eq!(schedules.row(1).unwrap().times(), &[0.75]);
        assert!(schedules.row(2).is_none());
    }

    #[test]
    fn rejects_duplicate_and_invalid_times() {
        assert_eq!(
            RaggedExerciseTimes::try_new(&[0, 2], &[0.5, 0.5]).unwrap_err(),
            BatchError::NonIncreasingExerciseTime {
                row: 0,
                event: 1,
                previous: 0.5,
                observed: 0.5,
            }
        );
        assert_eq!(
            RaggedExerciseTimes::try_new(&[0, 1], &[-0.0]).unwrap_err(),
            BatchError::InvalidExerciseTime {
                row: 0,
                event: 0,
                observed: -0.0,
            }
        );
    }

    #[test]
    fn serial_and_parallel_validation_report_the_same_lowest_row() {
        let offsets = [0, 1, 2, 3];
        let times = [0.25, f64::INFINITY, 0.75];
        let serial = RaggedExerciseTimes::try_new(&offsets, &times).unwrap_err();
        let parallel = RaggedExerciseTimes::try_new_with_executor(
            &offsets,
            &times,
            &CpuExecutor::try_new(2).unwrap(),
        )
        .unwrap_err();
        assert_eq!(serial, parallel);
    }
}
