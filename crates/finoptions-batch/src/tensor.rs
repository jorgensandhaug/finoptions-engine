//! Consumer-specific float32 tensors derived from exact ragged cash-dividend schedules.

use std::sync::atomic::{AtomicUsize, Ordering};

use finoptions_runtime::CpuExecutor;
use rayon::prelude::*;

use crate::{BatchError, CashDividendRow, RaggedCashDividends};

/// Whether a cash event exactly at option maturity belongs to the requested model input.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MaturityBoundary {
    /// Cash events must occur strictly before maturity.
    StrictlyBefore,
    /// Cash events may occur at or before maturity.
    AtOrBefore,
}

/// Exact interval closure for chronological fixed-width time bins.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TimeBinClosure {
    /// Use `[0, edge_1), ..., [edge_n-1, horizon]`.
    LeftClosedRightOpen,
    /// Use `(0, edge_1], ..., (edge_n-1, horizon]`.
    LeftOpenRightClosed,
}

/// Explicit cash-amount units for a model tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum CashAmountScale<'a> {
    /// Retain source currency amounts.
    Absolute,
    /// Divide every event amount by its row's strictly positive spot price.
    DivideBySpot(&'a [f64]),
}

/// Exact layout and maturity policy for right-padded chronological dividend sequences.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PaddedSequenceSpec {
    capacity: usize,
    maturity_boundary: MaturityBoundary,
}

impl PaddedSequenceSpec {
    /// Constructs an explicit fixed-capacity sequence specification.
    ///
    /// # Errors
    ///
    /// Rejects zero or a capacity outside the unsigned 32-bit model-length domain.
    pub fn try_new(
        capacity: usize,
        maturity_boundary: MaturityBoundary,
    ) -> Result<Self, BatchError> {
        if capacity == 0 {
            return Err(BatchError::EmptySequenceCapacity);
        }
        if u32::try_from(capacity).is_err() {
            return Err(BatchError::SequenceCapacityOverflow { observed: capacity });
        }
        Ok(Self {
            capacity,
            maturity_boundary,
        })
    }

    /// Returns the exact slots allocated per row.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity
    }

    /// Returns the exact maturity inclusion policy.
    #[must_use]
    pub const fn maturity_boundary(self) -> MaturityBoundary {
        self.maturity_boundary
    }
}

/// Exact layout, horizon, closure, and maturity policy for chronological dividend bins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DividendBinSpec {
    bins: usize,
    horizon: f64,
    closure: TimeBinClosure,
    maturity_boundary: MaturityBoundary,
}

impl DividendBinSpec {
    /// Constructs an explicit fixed-width chronological bin specification.
    ///
    /// # Errors
    ///
    /// Rejects zero or unrepresentable bin counts and non-finite/non-positive horizons.
    pub fn try_new(
        bins: usize,
        horizon: f64,
        closure: TimeBinClosure,
        maturity_boundary: MaturityBoundary,
    ) -> Result<Self, BatchError> {
        if bins == 0 {
            return Err(BatchError::EmptyBinCount);
        }
        if u32::try_from(bins).is_err() {
            return Err(BatchError::BinCountOverflow { observed: bins });
        }
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(BatchError::InvalidHorizon { observed: horizon });
        }
        Ok(Self {
            bins,
            horizon,
            closure,
            maturity_boundary,
        })
    }

    /// Returns the exact chronological bin count.
    #[must_use]
    pub const fn bins(self) -> usize {
        self.bins
    }

    /// Returns the fixed positive year-fraction horizon.
    #[must_use]
    pub const fn horizon(self) -> f64 {
        self.horizon
    }

    /// Returns the exact interval closure policy.
    #[must_use]
    pub const fn closure(self) -> TimeBinClosure {
        self.closure
    }

    /// Returns the exact maturity inclusion policy.
    #[must_use]
    pub const fn maturity_boundary(self) -> MaturityBoundary {
        self.maturity_boundary
    }
}

/// Owned chronological right-padded model tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct PaddedDividendBatch {
    rows: usize,
    capacity: usize,
    times: Vec<f32>,
    amounts: Vec<f32>,
    mask: Vec<bool>,
    lengths: Vec<u32>,
}

impl PaddedDividendBatch {
    /// Returns the number of option rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the exact padded width.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Borrows row-major `[rows, capacity]` event year fractions.
    #[must_use]
    pub fn times(&self) -> &[f32] {
        &self.times
    }

    /// Borrows row-major `[rows, capacity]` cash amounts in the requested units.
    #[must_use]
    pub fn amounts(&self) -> &[f32] {
        &self.amounts
    }

    /// Borrows row-major `[rows, capacity]` validity indicators.
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Borrows exact valid-event counts for every row.
    #[must_use]
    pub fn lengths(&self) -> &[u32] {
        &self.lengths
    }

    /// Transfers all owned buffers without copying.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f32>, Vec<f32>, Vec<bool>, Vec<u32>) {
        (self.times, self.amounts, self.mask, self.lengths)
    }
}

/// Owned chronological fixed-width cash-amount bins.
#[derive(Clone, Debug, PartialEq)]
pub struct BinnedDividendBatch {
    rows: usize,
    bins: usize,
    amounts: Vec<f32>,
}

impl BinnedDividendBatch {
    /// Returns the number of option rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the exact chronological bin count.
    #[must_use]
    pub const fn bins(&self) -> usize {
        self.bins
    }

    /// Borrows row-major `[rows, bins]` cash amounts in the requested units.
    #[must_use]
    pub fn amounts(&self) -> &[f32] {
        &self.amounts
    }

    /// Transfers the owned row-major amount buffer without copying.
    #[must_use]
    pub fn into_amounts(self) -> Vec<f32> {
        self.amounts
    }
}

/// Builds only chronological padded sequences, their mask, and exact lengths.
///
/// Source values remain `f64`; each emitted model value is checked before explicit float32
/// narrowing. The requested executor chooses a distinct serial or dedicated-pool kernel and is
/// never replaced.
///
/// # Errors
///
/// Returns a typed row/event error for maturity, capacity, scaling, shape, or precision violations.
pub fn pack_padded_dividends_f32(
    schedules: RaggedCashDividends<'_>,
    maturities: &[f64],
    scale: CashAmountScale<'_>,
    spec: PaddedSequenceSpec,
    executor: &CpuExecutor,
) -> Result<PaddedDividendBatch, BatchError> {
    validate_request(
        schedules,
        maturities,
        scale,
        spec.maturity_boundary,
        RequestDomain::Sequence {
            capacity: spec.capacity,
        },
        executor,
    )?;
    let values = checked_shape("padded dividend", schedules.rows(), spec.capacity)?;
    executor.dispatch(
        || pack_padded_serial(schedules, scale, spec, values),
        || pack_padded_parallel(schedules, scale, spec, values),
    )
}

/// Builds only chronological fixed-width amount bins.
///
/// # Errors
///
/// Returns a typed row/event error for maturity, horizon, scaling, shape, precision, or float32
/// accumulation violations. Parallel failures are replayed at the lowest failing row.
pub fn bin_dividends_f32(
    schedules: RaggedCashDividends<'_>,
    maturities: &[f64],
    scale: CashAmountScale<'_>,
    spec: DividendBinSpec,
    executor: &CpuExecutor,
) -> Result<BinnedDividendBatch, BatchError> {
    validate_request(
        schedules,
        maturities,
        scale,
        spec.maturity_boundary,
        RequestDomain::Bins {
            horizon: spec.horizon,
        },
        executor,
    )?;
    let values = checked_shape("binned dividend", schedules.rows(), spec.bins)?;
    executor.dispatch(
        || bin_serial(schedules, scale, spec, values),
        || bin_parallel(schedules, scale, spec, values),
    )
}

#[derive(Clone, Copy, Debug)]
enum RequestDomain {
    Sequence { capacity: usize },
    Bins { horizon: f64 },
}

fn validate_request(
    schedules: RaggedCashDividends<'_>,
    maturities: &[f64],
    scale: CashAmountScale<'_>,
    boundary: MaturityBoundary,
    domain: RequestDomain,
    executor: &CpuExecutor,
) -> Result<(), BatchError> {
    if maturities.len() != schedules.rows() {
        return Err(BatchError::RowCountMismatch {
            field: "maturities",
            expected: schedules.rows(),
            observed: maturities.len(),
        });
    }
    let spots = match scale {
        CashAmountScale::Absolute => None,
        CashAmountScale::DivideBySpot(spots) => {
            if spots.len() != schedules.rows() {
                return Err(BatchError::RowCountMismatch {
                    field: "spots",
                    expected: schedules.rows(),
                    observed: spots.len(),
                });
            }
            Some(spots)
        }
    };

    executor.dispatch(
        || validate_request_serial(schedules, maturities, spots, boundary, domain),
        || validate_request_parallel(schedules, maturities, spots, boundary, domain),
    )
}

fn validate_request_serial(
    schedules: RaggedCashDividends<'_>,
    maturities: &[f64],
    spots: Option<&[f64]>,
    boundary: MaturityBoundary,
    domain: RequestDomain,
) -> Result<(), BatchError> {
    for row in 0..schedules.rows() {
        validate_request_row(schedules, maturities, spots, boundary, domain, row)?;
    }
    Ok(())
}

fn validate_request_parallel(
    schedules: RaggedCashDividends<'_>,
    maturities: &[f64],
    spots: Option<&[f64]>,
    boundary: MaturityBoundary,
    domain: RequestDomain,
) -> Result<(), BatchError> {
    let first_failure = AtomicUsize::new(usize::MAX);
    (0..schedules.rows()).into_par_iter().for_each(|row| {
        if row > first_failure.load(Ordering::Relaxed) {
            return;
        }
        if validate_request_row(schedules, maturities, spots, boundary, domain, row).is_err() {
            first_failure.fetch_min(row, Ordering::Relaxed);
        }
    });
    let failed_row = first_failure.load(Ordering::Relaxed);
    if failed_row == usize::MAX {
        return Ok(());
    }
    match validate_request_row(schedules, maturities, spots, boundary, domain, failed_row) {
        Err(error) => Err(error),
        Ok(()) => Err(BatchError::ParallelFailureReplayMismatch { row: failed_row }),
    }
}

fn validate_request_row(
    schedules: RaggedCashDividends<'_>,
    maturities: &[f64],
    spots: Option<&[f64]>,
    boundary: MaturityBoundary,
    domain: RequestDomain,
    row: usize,
) -> Result<(), BatchError> {
    let maturity = maturities[row];
    if !maturity.is_finite() || maturity <= 0.0 {
        return Err(BatchError::InvalidMaturity {
            row,
            observed: maturity,
        });
    }
    match domain {
        RequestDomain::Sequence { capacity } => {
            let events = schedules.row_unchecked(row).len();
            if events > capacity {
                return Err(BatchError::SequenceCapacityExceeded {
                    row,
                    events,
                    capacity,
                });
            }
            if u32::try_from(events).is_err() {
                return Err(BatchError::EventCountOverflow { row, events });
            }
        }
        RequestDomain::Bins { horizon } if maturity > horizon => {
            return Err(BatchError::MaturityAfterHorizon {
                row,
                maturity,
                horizon,
            });
        }
        RequestDomain::Bins { .. } => {}
    }
    let spot = spots.map(|values| values[row]);
    if let Some(value) = spot
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(BatchError::InvalidSpot {
            row,
            observed: value,
        });
    }
    let schedule = schedules.row_unchecked(row);
    for (event, (&time, &amount)) in schedule.times().iter().zip(schedule.amounts()).enumerate() {
        if time > maturity {
            return Err(BatchError::EventAfterMaturity {
                row,
                event,
                event_time: time,
                maturity,
            });
        }
        if boundary == MaturityBoundary::StrictlyBefore && time.to_bits() == maturity.to_bits() {
            return Err(BatchError::EventAtExcludedMaturity {
                row,
                event,
                maturity,
            });
        }
        checked_f32("event time", row, event, time)?;
        checked_f32("cash amount", row, event, scaled_amount(amount, spot))?;
    }
    Ok(())
}

fn checked_shape(output: &'static str, rows: usize, width: usize) -> Result<usize, BatchError> {
    rows.checked_mul(width).ok_or(BatchError::ShapeOverflow {
        output,
        rows,
        width,
    })
}

fn checked_f32(
    field: &'static str,
    row: usize,
    element: usize,
    value: f64,
) -> Result<f32, BatchError> {
    let narrowed = narrow_f64_to_f32(value);
    let underflowed = value != 0.0 && narrowed == 0.0;
    let relative_error = if value == 0.0 {
        0.0
    } else {
        (f64::from(narrowed) - value).abs() / value.abs()
    };
    if !narrowed.is_finite()
        || underflowed
        || !relative_error.is_finite()
        || relative_error > f64::from(f32::EPSILON)
    {
        return Err(BatchError::UnsafeFloat32Cast {
            field,
            row,
            element,
            observed: value,
        });
    }
    Ok(narrowed)
}

#[allow(clippy::cast_possible_truncation)]
fn narrow_f64_to_f32(value: f64) -> f32 {
    // Every production call either occurs inside `checked_f32` or after that exact source value
    // has passed `checked_f32` during request validation.
    value as f32
}

fn scaled_amount(amount: f64, spot: Option<f64>) -> f64 {
    spot.map_or(amount, |value| amount / value)
}

fn pack_padded_serial(
    schedules: RaggedCashDividends<'_>,
    scale: CashAmountScale<'_>,
    spec: PaddedSequenceSpec,
    values: usize,
) -> Result<PaddedDividendBatch, BatchError> {
    let mut output = empty_padded(schedules.rows(), spec.capacity, values)?;
    for row in 0..schedules.rows() {
        let start = row * spec.capacity;
        fill_padded_row(
            schedules.row_unchecked(row),
            scale_spot(scale, row),
            &mut output.times[start..start + spec.capacity],
            &mut output.amounts[start..start + spec.capacity],
            &mut output.mask[start..start + spec.capacity],
            &mut output.lengths[row],
        );
    }
    Ok(output)
}

fn pack_padded_parallel(
    schedules: RaggedCashDividends<'_>,
    scale: CashAmountScale<'_>,
    spec: PaddedSequenceSpec,
    values: usize,
) -> Result<PaddedDividendBatch, BatchError> {
    let mut output = empty_padded(schedules.rows(), spec.capacity, values)?;
    output
        .times
        .par_chunks_mut(spec.capacity)
        .zip(output.amounts.par_chunks_mut(spec.capacity))
        .zip(output.mask.par_chunks_mut(spec.capacity))
        .zip(output.lengths.par_iter_mut())
        .enumerate()
        .for_each(|(row, (((times, amounts), mask), length))| {
            fill_padded_row(
                schedules.row_unchecked(row),
                scale_spot(scale, row),
                times,
                amounts,
                mask,
                length,
            );
        });
    Ok(output)
}

fn empty_padded(
    rows: usize,
    capacity: usize,
    values: usize,
) -> Result<PaddedDividendBatch, BatchError> {
    Ok(PaddedDividendBatch {
        rows,
        capacity,
        times: filled_vec("padded dividend times", values, 0.0)?,
        amounts: filled_vec("padded dividend amounts", values, 0.0)?,
        mask: filled_vec("padded dividend mask", values, false)?,
        lengths: filled_vec("padded dividend lengths", rows, 0)?,
    })
}

fn fill_padded_row(
    schedule: CashDividendRow<'_>,
    spot: Option<f64>,
    output_times: &mut [f32],
    output_amounts: &mut [f32],
    output_mask: &mut [bool],
    output_length: &mut u32,
) {
    *output_length = u32::try_from(schedule.len()).expect("validated event count");
    for (event, (&time, &amount)) in schedule.times().iter().zip(schedule.amounts()).enumerate() {
        output_times[event] = narrow_f64_to_f32(time);
        output_amounts[event] = narrow_f64_to_f32(scaled_amount(amount, spot));
        output_mask[event] = true;
    }
}

fn bin_serial(
    schedules: RaggedCashDividends<'_>,
    scale: CashAmountScale<'_>,
    spec: DividendBinSpec,
    values: usize,
) -> Result<BinnedDividendBatch, BatchError> {
    let mut amounts = filled_vec("binned dividend amounts", values, 0.0)?;
    for row in 0..schedules.rows() {
        let start = row * spec.bins;
        fill_bin_row(
            row,
            schedules.row_unchecked(row),
            scale_spot(scale, row),
            spec,
            &mut amounts[start..start + spec.bins],
        )?;
    }
    Ok(BinnedDividendBatch {
        rows: schedules.rows(),
        bins: spec.bins,
        amounts,
    })
}

fn bin_parallel(
    schedules: RaggedCashDividends<'_>,
    scale: CashAmountScale<'_>,
    spec: DividendBinSpec,
    values: usize,
) -> Result<BinnedDividendBatch, BatchError> {
    let mut amounts = filled_vec("binned dividend amounts", values, 0.0)?;
    let first_failure = AtomicUsize::new(usize::MAX);
    amounts
        .par_chunks_mut(spec.bins)
        .enumerate()
        .for_each(|(row, output)| {
            if row > first_failure.load(Ordering::Relaxed) {
                return;
            }
            if fill_bin_row(
                row,
                schedules.row_unchecked(row),
                scale_spot(scale, row),
                spec,
                output,
            )
            .is_err()
            {
                first_failure.fetch_min(row, Ordering::Relaxed);
            }
        });
    let failed_row = first_failure.load(Ordering::Relaxed);
    if failed_row != usize::MAX {
        let mut diagnostic = filled_vec("dividend-bin diagnostic", spec.bins, 0.0)?;
        return match fill_bin_row(
            failed_row,
            schedules.row_unchecked(failed_row),
            scale_spot(scale, failed_row),
            spec,
            &mut diagnostic,
        ) {
            Err(error) => Err(error),
            Ok(()) => Err(BatchError::ParallelFailureReplayMismatch { row: failed_row }),
        };
    }
    Ok(BinnedDividendBatch {
        rows: schedules.rows(),
        bins: spec.bins,
        amounts,
    })
}

fn fill_bin_row(
    row: usize,
    schedule: CashDividendRow<'_>,
    spot: Option<f64>,
    spec: DividendBinSpec,
    output: &mut [f32],
) -> Result<(), BatchError> {
    for (event, (&time, &amount)) in schedule.times().iter().zip(schedule.amounts()).enumerate() {
        let bin = bin_index(time, spec);
        let value = checked_f32("cash amount", row, event, scaled_amount(amount, spot))?;
        output[bin] += value;
        if !output[bin].is_finite() {
            return Err(BatchError::BinSumOverflow { row, bin });
        }
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bin_index(time: f64, spec: DividendBinSpec) -> usize {
    if time.to_bits() == spec.horizon.to_bits() {
        return spec.bins - 1;
    }
    // Common validation proves `0 < time < horizon`, while construction proves
    // `1 <= bins <= u32::MAX`. The scaled value is therefore finite and within `[0, bins)`.
    let scaled = (time / spec.horizon) * f64::from(u32::try_from(spec.bins).unwrap());
    let index = match spec.closure {
        TimeBinClosure::LeftClosedRightOpen => scaled.floor() as usize,
        TimeBinClosure::LeftOpenRightClosed => (scaled.ceil() as usize).saturating_sub(1),
    };
    index.min(spec.bins - 1)
}

fn scale_spot(scale: CashAmountScale<'_>, row: usize) -> Option<f64> {
    match scale {
        CashAmountScale::Absolute => None,
        CashAmountScale::DivideBySpot(spots) => Some(spots[row]),
    }
}

fn filled_vec<T: Clone>(
    output: &'static str,
    elements: usize,
    value: T,
) -> Result<Vec<T>, BatchError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| BatchError::AllocationFailure { output, elements })?;
    values.resize(elements, value);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedules() -> RaggedCashDividends<'static> {
        RaggedCashDividends::try_new(
            &[0, 2, 2, 4],
            &[0.25, 1.0, 0.25, 0.75],
            &[1.0, 2.0, 3.0, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn padded_sequence_is_chronological_right_padded_masked_and_scaled() {
        let spec = PaddedSequenceSpec::try_new(3, MaturityBoundary::AtOrBefore).unwrap();
        let spots = [100.0, 50.0, 200.0];
        let output = pack_padded_dividends_f32(
            schedules(),
            &[1.0, 0.5, 1.0],
            CashAmountScale::DivideBySpot(&spots),
            spec,
            &CpuExecutor::try_new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(output.rows(), 3);
        assert_eq!(output.capacity(), 3);
        assert_eq!(
            output.times(),
            &[0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75, 0.0]
        );
        assert_eq!(
            output.amounts(),
            &[0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.015, 0.005, 0.0]
        );
        assert_eq!(
            output.mask(),
            &[true, true, false, false, false, false, true, true, false]
        );
        assert_eq!(output.lengths(), &[2, 0, 2]);
    }

    #[test]
    fn bin_closure_and_maturity_boundary_are_explicit() {
        let left = DividendBinSpec::try_new(
            4,
            1.0,
            TimeBinClosure::LeftClosedRightOpen,
            MaturityBoundary::AtOrBefore,
        )
        .unwrap();
        let right = DividendBinSpec::try_new(
            4,
            1.0,
            TimeBinClosure::LeftOpenRightClosed,
            MaturityBoundary::AtOrBefore,
        )
        .unwrap();
        let executor = CpuExecutor::try_new(1).unwrap();
        let left = bin_dividends_f32(
            schedules(),
            &[1.0, 0.5, 1.0],
            CashAmountScale::Absolute,
            left,
            &executor,
        )
        .unwrap();
        let right = bin_dividends_f32(
            schedules(),
            &[1.0, 0.5, 1.0],
            CashAmountScale::Absolute,
            right,
            &executor,
        )
        .unwrap();
        assert_eq!(&left.amounts()[..4], &[0.0, 1.0, 0.0, 2.0]);
        assert_eq!(&right.amounts()[..4], &[1.0, 0.0, 0.0, 2.0]);

        let strict = PaddedSequenceSpec::try_new(3, MaturityBoundary::StrictlyBefore).unwrap();
        assert!(matches!(
            pack_padded_dividends_f32(
                schedules(),
                &[1.0, 0.5, 1.0],
                CashAmountScale::Absolute,
                strict,
                &executor,
            ),
            Err(BatchError::EventAtExcludedMaturity {
                row: 0,
                event: 1,
                ..
            })
        ));
    }

    #[test]
    fn serial_and_explicit_parallel_outputs_are_bitwise_identical() {
        let sequence = PaddedSequenceSpec::try_new(4, MaturityBoundary::AtOrBefore).unwrap();
        let bins = DividendBinSpec::try_new(
            12,
            1.0,
            TimeBinClosure::LeftOpenRightClosed,
            MaturityBoundary::AtOrBefore,
        )
        .unwrap();
        let maturities = [1.0, 0.5, 1.0];
        let serial = CpuExecutor::try_new(1).unwrap();
        let parallel = CpuExecutor::try_new(4).unwrap();
        assert_eq!(
            pack_padded_dividends_f32(
                schedules(),
                &maturities,
                CashAmountScale::Absolute,
                sequence,
                &serial,
            )
            .unwrap(),
            pack_padded_dividends_f32(
                schedules(),
                &maturities,
                CashAmountScale::Absolute,
                sequence,
                &parallel,
            )
            .unwrap(),
        );
        assert_eq!(
            bin_dividends_f32(
                schedules(),
                &maturities,
                CashAmountScale::Absolute,
                bins,
                &serial,
            )
            .unwrap(),
            bin_dividends_f32(
                schedules(),
                &maturities,
                CashAmountScale::Absolute,
                bins,
                &parallel,
            )
            .unwrap(),
        );
    }

    #[test]
    fn fails_early_on_shape_domain_and_precision_violations() {
        let executor = CpuExecutor::try_new(1).unwrap();
        let capacity = PaddedSequenceSpec::try_new(1, MaturityBoundary::AtOrBefore).unwrap();
        assert!(matches!(
            pack_padded_dividends_f32(
                schedules(),
                &[1.0, 0.5, 1.0],
                CashAmountScale::Absolute,
                capacity,
                &executor,
            ),
            Err(BatchError::SequenceCapacityExceeded { row: 0, .. })
        ));
        assert!(matches!(
            pack_padded_dividends_f32(
                schedules(),
                &[0.9, 0.5, 1.0],
                CashAmountScale::Absolute,
                PaddedSequenceSpec::try_new(2, MaturityBoundary::AtOrBefore).unwrap(),
                &executor,
            ),
            Err(BatchError::EventAfterMaturity {
                row: 0,
                event: 1,
                ..
            })
        ));

        let tiny = RaggedCashDividends::try_new(&[0, 1], &[f64::MIN_POSITIVE], &[1.0]).unwrap();
        assert!(matches!(
            pack_padded_dividends_f32(
                tiny,
                &[1.0],
                CashAmountScale::Absolute,
                PaddedSequenceSpec::try_new(1, MaturityBoundary::AtOrBefore).unwrap(),
                &executor,
            ),
            Err(BatchError::UnsafeFloat32Cast {
                field: "event time",
                ..
            })
        ));
    }

    #[test]
    fn parallel_request_validation_reports_the_lowest_invalid_row() {
        let schedules =
            RaggedCashDividends::try_new(&[0, 1, 2], &[0.75, 0.25], &[1.0, 1.0]).unwrap();
        let error = pack_padded_dividends_f32(
            schedules,
            &[0.5, 1.0],
            CashAmountScale::DivideBySpot(&[100.0, -1.0]),
            PaddedSequenceSpec::try_new(1, MaturityBoundary::AtOrBefore).unwrap(),
            &CpuExecutor::try_new(4).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchError::EventAfterMaturity {
                row: 0,
                event: 0,
                ..
            }
        ));
    }
}
