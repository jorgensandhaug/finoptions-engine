//! Reproducible circular stationary bootstrap for additive panel statistics.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use thiserror::Error;

use crate::random::PhiloxPathStream;
use crate::{CpuExecutor, RandomStream};

const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;
const MAX_EXACT_F64_INTEGER_F64: f64 = 9_007_199_254_740_992.0;
const OPEN_F64_DENOMINATOR: f64 = 4_503_599_627_370_496.0;
const TWO_POW_32_F64: f64 = 4_294_967_296.0;

/// A fail-fast violation of the additive-panel stationary-bootstrap contract.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum BootstrapError {
    /// At least one observed panel is required.
    #[error("stationary bootstrap requires at least one observed panel")]
    EmptyPanels,

    /// At least one additive statistic is required.
    #[error("stationary bootstrap requires at least one additive statistic")]
    EmptyStatistics,

    /// The declared matrix shape cannot be represented by one contiguous slice.
    #[error("panel/statistic shape {panels} x {statistics} overflows addressable length")]
    ShapeOverflow {
        /// Observed panel count.
        panels: usize,
        /// Additive statistic count.
        statistics: usize,
    },

    /// The flat values do not contain exactly one row per panel.
    #[error(
        "panel/statistic values have length {observed}; expected {expected} for shape {panels} x {statistics}"
    )]
    ShapeMismatch {
        /// Observed flat length.
        observed: usize,
        /// Required flat length.
        expected: usize,
        /// Observed panel count.
        panels: usize,
        /// Additive statistic count.
        statistics: usize,
    },

    /// A stratified request needs at least one `[start, end)` range.
    #[error(
        "stratified stationary bootstrap requires at least two stratum offsets; observed {observed}"
    )]
    StratumOffsetCount {
        /// Rejected offset count.
        observed: usize,
    },

    /// Contiguous stratum offsets must begin at the first panel.
    #[error("stratum offsets must start at zero; observed {observed}")]
    StratumOffsetsDoNotStartAtZero {
        /// Rejected first offset.
        observed: u64,
    },

    /// Every stratum must contain at least one panel and follow its predecessor.
    #[error(
        "stratum {stratum} has invalid range [{start}, {end}); offsets must be strictly increasing"
    )]
    InvalidStratumRange {
        /// Zero-based stratum index.
        stratum: usize,
        /// Rejected inclusive start.
        start: u64,
        /// Rejected exclusive end.
        end: u64,
    },

    /// Contiguous strata must cover every source panel exactly once.
    #[error("final stratum offset is {observed}; expected panel count {expected}")]
    StratumOffsetEndMismatch {
        /// Rejected final offset.
        observed: u64,
        /// Required final offset.
        expected: u64,
    },

    /// A source statistic cannot enter a checked floating-point reduction.
    #[error("panel {panel} statistic {statistic} must be finite; observed {observed:?}")]
    NonFiniteInput {
        /// Zero-based panel index.
        panel: usize,
        /// Zero-based statistic index.
        statistic: usize,
        /// Rejected value.
        observed: f64,
    },

    /// Panel counts beyond the exact-integer range cannot use the declared inverse-CDF mapping.
    #[error("panel count {observed} exceeds the exact f64 integer maximum {maximum}")]
    PanelCountExceedsExactF64Maximum {
        /// Rejected panel count.
        observed: usize,
        /// Largest admitted panel count.
        maximum: u64,
    },

    /// At least one bootstrap replication is required.
    #[error("stationary bootstrap requires at least one replication")]
    ZeroReplications,

    /// Replications are addressed by one unsigned 32-bit Philox counter lane.
    #[error("replication count {observed} exceeds the supported maximum {maximum}")]
    ReplicationCountOverflow {
        /// Rejected replication count.
        observed: usize,
        /// Largest supported count.
        maximum: u32,
    },

    /// The requested path range would wrap its Philox counter lane.
    #[error(
        "replication path range starting at {start} with count {replications} exceeds u32::MAX"
    )]
    ReplicationPathRangeOverflow {
        /// First declared path identifier.
        start: u32,
        /// Declared replication count.
        replications: u32,
    },

    /// A stationary-bootstrap geometric mean must be finite and at least one.
    #[error("mean block length must be finite and at least one; observed {observed:?}")]
    InvalidMeanBlockLength {
        /// Rejected mean block length.
        observed: f64,
    },

    /// Prefix storage could not be represented or allocated.
    #[error("cannot allocate bootstrap prefix matrix with {elements} f64 elements")]
    PrefixAllocationFailed {
        /// Requested element count.
        elements: usize,
    },

    /// Output storage could not be represented or allocated.
    #[error("cannot allocate bootstrap output matrix with {elements} f64 elements")]
    ResultAllocationFailed {
        /// Requested element count.
        elements: usize,
    },

    /// Ordered prefix accumulation left the finite numerical domain.
    #[error(
        "bootstrap prefix became non-finite at panel {panel}, statistic {statistic}: {observed:?}"
    )]
    NonFinitePrefix {
        /// Inclusive source panel that produced the prefix.
        panel: usize,
        /// Zero-based statistic index.
        statistic: usize,
        /// Rejected prefix value.
        observed: f64,
    },

    /// A resampled additive sum left the finite numerical domain.
    #[error(
        "bootstrap replication {replication} statistic {statistic} became non-finite: {observed:?}"
    )]
    NonFiniteReplicationSum {
        /// Zero-based replication index.
        replication: usize,
        /// Zero-based statistic index.
        statistic: usize,
        /// Rejected sum.
        observed: f64,
    },

    /// Parallel failure replay did not reproduce the deterministic failure.
    #[error("parallel bootstrap failure at replication {replication} did not reproduce serially")]
    ParallelFailureReplayMismatch {
        /// First failed replication index.
        replication: usize,
    },
}

/// Complete random and geometric-block identity for one stationary-bootstrap request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationaryBootstrapSpec {
    replications: u32,
    mean_block_length: f64,
    stream: RandomStream,
    replication_path_start: u32,
}

impl StationaryBootstrapSpec {
    /// Validates the complete counter namespace and block-length parameter before allocation.
    ///
    /// Each zero-based replication occupies path lane `replication_path_start + replication` in
    /// the declared Philox stream. The origin sampler uses unbiased 64-bit multiply-high rejection.
    /// Block lengths use the documented open-52-bit-midpoint inverse geometric CDF.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a zero or overflowing replication range, or a nonfinite mean block
    /// length below one. No parameter is clamped or substituted.
    pub fn try_new(
        replications: usize,
        mean_block_length: f64,
        stream: RandomStream,
        replication_path_start: u32,
    ) -> Result<Self, BootstrapError> {
        if replications == 0 {
            return Err(BootstrapError::ZeroReplications);
        }
        let replications =
            u32::try_from(replications).map_err(|_| BootstrapError::ReplicationCountOverflow {
                observed: replications,
                maximum: u32::MAX,
            })?;
        if replication_path_start
            .checked_add(replications - 1)
            .is_none()
        {
            return Err(BootstrapError::ReplicationPathRangeOverflow {
                start: replication_path_start,
                replications,
            });
        }
        if !mean_block_length.is_finite() || mean_block_length < 1.0 {
            return Err(BootstrapError::InvalidMeanBlockLength {
                observed: mean_block_length,
            });
        }
        Ok(Self {
            replications,
            mean_block_length,
            stream,
            replication_path_start,
        })
    }

    /// Returns the exact number of resampled panel sequences.
    #[must_use]
    pub const fn replications(self) -> u32 {
        self.replications
    }

    /// Returns the geometric mean block length.
    #[must_use]
    pub const fn mean_block_length(self) -> f64 {
        self.mean_block_length
    }

    /// Returns the Philox key and counter domain.
    #[must_use]
    pub const fn stream(self) -> RandomStream {
        self.stream
    }

    /// Returns the first replication's Philox path lane.
    #[must_use]
    pub const fn replication_path_start(self) -> u32 {
        self.replication_path_start
    }
}

/// Borrowed row-major sufficient statistics for ordered observed panels.
#[derive(Clone, Copy, Debug)]
pub struct AdditivePanelStatistics<'a> {
    values: &'a [f64],
    panels: usize,
    statistics: usize,
}

#[derive(Clone, Copy)]
enum PanelResamplingLayout<'a> {
    Whole,
    Strata(&'a [usize]),
}

impl<'a> AdditivePanelStatistics<'a> {
    /// Validates a finite row-major `panels x statistics` matrix without copying it.
    ///
    /// # Errors
    ///
    /// Returns the first shape or finite-domain violation. Empty dimensions and panel counts above
    /// the exact Float64 integer range are rejected before any output allocation.
    pub fn try_new(
        values: &'a [f64],
        panels: usize,
        statistics: usize,
    ) -> Result<Self, BootstrapError> {
        if panels == 0 {
            return Err(BootstrapError::EmptyPanels);
        }
        if statistics == 0 {
            return Err(BootstrapError::EmptyStatistics);
        }
        if u64::try_from(panels).map_or(true, |value| value > MAX_EXACT_F64_INTEGER) {
            return Err(BootstrapError::PanelCountExceedsExactF64Maximum {
                observed: panels,
                maximum: MAX_EXACT_F64_INTEGER,
            });
        }
        let expected = panels
            .checked_mul(statistics)
            .ok_or(BootstrapError::ShapeOverflow { panels, statistics })?;
        if values.len() != expected {
            return Err(BootstrapError::ShapeMismatch {
                observed: values.len(),
                expected,
                panels,
                statistics,
            });
        }
        if let Some((index, &observed)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BootstrapError::NonFiniteInput {
                panel: index / statistics,
                statistic: index % statistics,
                observed,
            });
        }
        Ok(Self {
            values,
            panels,
            statistics,
        })
    }

    /// Returns the ordered observed panel count.
    #[must_use]
    pub const fn panels(self) -> usize {
        self.panels
    }

    /// Returns the number of additive sufficient statistics per panel.
    #[must_use]
    pub const fn statistics(self) -> usize {
        self.statistics
    }

    /// Resamples complete circular panel blocks and returns one additive-sum row per replication.
    ///
    /// A prefix-sum kernel evaluates each geometric block in `O(statistics)` rather than revisiting
    /// every panel. Every output still represents exactly `panels` resampled positions. Replication
    /// streams are path-addressed, so output is invariant to executor thread count on one numerical
    /// build.
    ///
    /// # Errors
    ///
    /// Returns before sampling on allocation or prefix overflow. Any nonfinite replication sum is
    /// reported at the first replication and statistic; no partial result is returned.
    pub fn stationary_bootstrap_sums(
        self,
        spec: StationaryBootstrapSpec,
        executor: &CpuExecutor,
    ) -> Result<PanelBootstrapSums, BootstrapError> {
        let prefix = self.build_prefix()?;
        self.bootstrap_sums_from_prefix(&prefix, PanelResamplingLayout::Whole, spec, executor)
    }

    /// Resamples circular geometric blocks independently inside contiguous source strata.
    ///
    /// `stratum_offsets` must be a strictly increasing partition `[0, ..., panels]`. Every
    /// replication preserves the exact number of panel positions from every stratum, while block
    /// origins wrap only inside their own stratum. One Philox path owns the complete fixed-order
    /// stratum sequence, so output remains invariant to executor width.
    ///
    /// # Errors
    ///
    /// Returns before sampling when offsets do not define a complete nonempty partition, or on the
    /// same allocation, prefix, and finite-domain failures as [`Self::stationary_bootstrap_sums`].
    pub fn stationary_bootstrap_stratified_sums(
        self,
        stratum_offsets: &[u64],
        spec: StationaryBootstrapSpec,
        executor: &CpuExecutor,
    ) -> Result<PanelBootstrapSums, BootstrapError> {
        let strata = self.validate_strata(stratum_offsets)?;
        let prefix = self.build_prefix()?;
        self.bootstrap_sums_from_prefix(
            &prefix,
            PanelResamplingLayout::Strata(&strata),
            spec,
            executor,
        )
    }

    fn bootstrap_sums_from_prefix(
        self,
        prefix: &[f64],
        layout: PanelResamplingLayout<'_>,
        spec: StationaryBootstrapSpec,
        executor: &CpuExecutor,
    ) -> Result<PanelBootstrapSums, BootstrapError> {
        let replications = spec.replications as usize;
        let elements = replications.checked_mul(self.statistics).ok_or(
            BootstrapError::ResultAllocationFailed {
                elements: usize::MAX,
            },
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| BootstrapError::ResultAllocationFailed { elements })?;
        values.resize(elements, 0.0);
        executor.dispatch_mut(
            &mut values,
            |output| self.fill_serial(prefix, layout, spec, output),
            |output| self.fill_parallel(prefix, layout, spec, output),
        )?;
        Ok(PanelBootstrapSums {
            values,
            replications,
            statistics: self.statistics,
        })
    }

    fn validate_strata(self, offsets: &[u64]) -> Result<Box<[usize]>, BootstrapError> {
        if offsets.len() < 2 {
            return Err(BootstrapError::StratumOffsetCount {
                observed: offsets.len(),
            });
        }
        if offsets[0] != 0 {
            return Err(BootstrapError::StratumOffsetsDoNotStartAtZero {
                observed: offsets[0],
            });
        }
        for (stratum, pair) in offsets.windows(2).enumerate() {
            if pair[0] >= pair[1] {
                return Err(BootstrapError::InvalidStratumRange {
                    stratum,
                    start: pair[0],
                    end: pair[1],
                });
            }
        }
        let expected = u64::try_from(self.panels).expect("validated panel count fits u64");
        let observed = *offsets.last().expect("validated nonempty offsets");
        if observed != expected {
            return Err(BootstrapError::StratumOffsetEndMismatch { observed, expected });
        }
        Ok(offsets
            .iter()
            .map(|&offset| usize::try_from(offset).expect("validated offset fits usize"))
            .collect())
    }

    fn build_prefix(self) -> Result<Vec<f64>, BootstrapError> {
        let rows = self
            .panels
            .checked_add(1)
            .ok_or(BootstrapError::PrefixAllocationFailed {
                elements: usize::MAX,
            })?;
        let elements =
            rows.checked_mul(self.statistics)
                .ok_or(BootstrapError::PrefixAllocationFailed {
                    elements: usize::MAX,
                })?;
        let mut prefix = Vec::new();
        prefix
            .try_reserve_exact(elements)
            .map_err(|_| BootstrapError::PrefixAllocationFailed { elements })?;
        prefix.resize(elements, 0.0);
        for panel in 0..self.panels {
            let source_start = panel * self.statistics;
            let previous_start = panel * self.statistics;
            let next_start = (panel + 1) * self.statistics;
            for statistic in 0..self.statistics {
                let value =
                    prefix[previous_start + statistic] + self.values[source_start + statistic];
                if !value.is_finite() {
                    return Err(BootstrapError::NonFinitePrefix {
                        panel,
                        statistic,
                        observed: value,
                    });
                }
                prefix[next_start + statistic] = value;
            }
        }
        Ok(prefix)
    }

    fn fill_serial(
        self,
        prefix: &[f64],
        layout: PanelResamplingLayout<'_>,
        spec: StationaryBootstrapSpec,
        values: &mut [f64],
    ) -> Result<(), BootstrapError> {
        for (replication, output) in values.chunks_mut(self.statistics).enumerate() {
            self.fill_replication(prefix, layout, spec, replication, output)?;
        }
        Ok(())
    }

    fn fill_parallel(
        self,
        prefix: &[f64],
        layout: PanelResamplingLayout<'_>,
        spec: StationaryBootstrapSpec,
        values: &mut [f64],
    ) -> Result<(), BootstrapError> {
        let first_failure = AtomicUsize::new(usize::MAX);
        values
            .par_chunks_mut(self.statistics)
            .enumerate()
            .for_each(|(replication, output)| {
                if replication > first_failure.load(Ordering::Relaxed) {
                    return;
                }
                if self
                    .fill_replication(prefix, layout, spec, replication, output)
                    .is_err()
                {
                    first_failure.fetch_min(replication, Ordering::Relaxed);
                }
            });
        let failed = first_failure.load(Ordering::Relaxed);
        if failed == usize::MAX {
            return Ok(());
        }
        let start = failed * self.statistics;
        match self.fill_replication(
            prefix,
            layout,
            spec,
            failed,
            &mut values[start..start + self.statistics],
        ) {
            Err(error) => Err(error),
            Ok(()) => Err(BootstrapError::ParallelFailureReplayMismatch {
                replication: failed,
            }),
        }
    }

    fn fill_replication(
        self,
        prefix: &[f64],
        layout: PanelResamplingLayout<'_>,
        spec: StationaryBootstrapSpec,
        replication: usize,
        output: &mut [f64],
    ) -> Result<(), BootstrapError> {
        output.fill(0.0);
        let replication_u32 = u32::try_from(replication).expect("validated replication index");
        let path = spec
            .replication_path_start
            .checked_add(replication_u32)
            .expect("validated replication path range");
        let mut stream = PhiloxPathStream::new(spec.stream.seed(), spec.stream.domain(), path, 0);
        match layout {
            PanelResamplingLayout::Whole => self.fill_range(
                prefix,
                0,
                self.panels,
                spec.mean_block_length,
                &mut stream,
                output,
            ),
            PanelResamplingLayout::Strata(strata) => {
                for pair in strata.windows(2) {
                    self.fill_range(
                        prefix,
                        pair[0],
                        pair[1],
                        spec.mean_block_length,
                        &mut stream,
                        output,
                    );
                }
            }
        }
        if let Some((statistic, &observed)) = output
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BootstrapError::NonFiniteReplicationSum {
                replication,
                statistic,
                observed,
            });
        }
        Ok(())
    }

    fn fill_range(
        self,
        prefix: &[f64],
        range_start: usize,
        range_end: usize,
        mean_block_length: f64,
        stream: &mut PhiloxPathStream,
        output: &mut [f64],
    ) {
        let panels = range_end - range_start;
        let mut remaining = panels;
        while remaining != 0 {
            let origin = usize::try_from(sample_bounded_u64(
                stream,
                u64::try_from(panels).expect("range panel count fits u64"),
            ))
            .expect("sampled origin fits usize");
            let length = sample_geometric_length(stream, mean_block_length, remaining);
            self.add_circular_range_block(prefix, range_start, range_end, origin, length, output);
            remaining -= length;
        }
    }

    fn add_circular_range_block(
        self,
        prefix: &[f64],
        range_start: usize,
        range_end: usize,
        relative_origin: usize,
        length: usize,
        output: &mut [f64],
    ) {
        let panels = range_end - range_start;
        debug_assert!(range_start < range_end && range_end <= self.panels);
        debug_assert!(relative_origin < panels);
        debug_assert!((1..=panels).contains(&length));
        debug_assert_eq!(output.len(), self.statistics);
        let origin = range_start + relative_origin;
        let end = origin + length;
        if end <= range_end {
            let origin_start = origin * self.statistics;
            let end_start = end * self.statistics;
            for statistic in 0..self.statistics {
                output[statistic] +=
                    prefix[end_start + statistic] - prefix[origin_start + statistic];
            }
            return;
        }
        let origin_start = origin * self.statistics;
        let range_end_start = range_end * self.statistics;
        let wrapped_end = range_start + end - range_end;
        let range_start_index = range_start * self.statistics;
        let wrapped_end_start = wrapped_end * self.statistics;
        for statistic in 0..self.statistics {
            let tail = prefix[range_end_start + statistic] - prefix[origin_start + statistic];
            let head =
                prefix[wrapped_end_start + statistic] - prefix[range_start_index + statistic];
            output[statistic] += tail + head;
        }
    }
}

/// Owned row-major bootstrap sums with shape `replications x statistics`.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelBootstrapSums {
    values: Vec<f64>,
    replications: usize,
    statistics: usize,
}

impl PanelBootstrapSums {
    /// Returns the row-major additive sums.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Consumes the result and returns its row-major allocation.
    #[must_use]
    pub fn into_vec(self) -> Vec<f64> {
        self.values
    }

    /// Returns the output row count.
    #[must_use]
    pub const fn replications(&self) -> usize {
        self.replications
    }

    /// Returns the output column count.
    #[must_use]
    pub const fn statistics(&self) -> usize {
        self.statistics
    }
}

#[inline]
fn sample_bounded_u64(stream: &mut PhiloxPathStream, bound: u64) -> u64 {
    debug_assert_ne!(bound, 0);
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let word = stream.next_u64();
        let product = u128::from(word) * u128::from(bound);
        let low =
            u64::try_from(product & u128::from(u64::MAX)).expect("masked multiply-low fits u64");
        if low >= threshold {
            return u64::try_from(product >> 64).expect("multiply-high fits u64");
        }
    }
}

#[inline]
fn sample_geometric_length(
    stream: &mut PhiloxPathStream,
    mean_block_length: f64,
    remaining: usize,
) -> usize {
    debug_assert!(mean_block_length.is_finite() && mean_block_length >= 1.0);
    debug_assert_ne!(remaining, 0);
    if mean_block_length <= 1.0 {
        return 1;
    }
    let uniform = open_unit_midpoint_52(stream.next_u64());
    let log_survival = libm::log1p(-uniform);
    let log_continuation = libm::log1p(-mean_block_length.recip());
    let length = libm::floor(log_survival / log_continuation) + 1.0;
    let remaining_f64 = exact_u64_to_f64(u64::try_from(remaining).expect("panel count fits u64"));
    if length >= remaining_f64 {
        remaining
    } else {
        exact_positive_integer_f64_to_usize(length)
    }
}

#[inline]
fn exact_u64_to_f64(value: u64) -> f64 {
    debug_assert!(value <= (1_u64 << 53));
    let high = u32::try_from(value >> 32).expect("high word fits u32");
    let low = u32::try_from(value & u64::from(u32::MAX)).expect("low word fits u32");
    f64::from(high).mul_add(TWO_POW_32_F64, f64::from(low))
}

#[inline]
fn open_unit_midpoint_52(word: u64) -> f64 {
    let bits = word >> 12;
    (exact_u64_to_f64(bits) + 0.5) / OPEN_F64_DENOMINATOR
}

#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn exact_positive_integer_f64_to_usize(value: f64) -> usize {
    // The inverse CDF was floored, and the caller proved 1 <= value < remaining <= 2^53.
    debug_assert!(value.is_finite() && (1.0..=MAX_EXACT_F64_INTEGER_F64).contains(&value));
    value as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(replications: usize, mean_block_length: f64) -> StationaryBootstrapSpec {
        StationaryBootstrapSpec::try_new(
            replications,
            mean_block_length,
            RandomStream::new(0x1319_8a2e_0370_7344, 2_000_000),
            0,
        )
        .unwrap()
    }

    #[test]
    fn rejects_incomplete_shapes_and_wrapping_streams() {
        assert_eq!(
            AdditivePanelStatistics::try_new(&[], 0, 1).unwrap_err(),
            BootstrapError::EmptyPanels
        );
        assert_eq!(
            AdditivePanelStatistics::try_new(&[], 1, 0).unwrap_err(),
            BootstrapError::EmptyStatistics
        );
        assert_eq!(
            AdditivePanelStatistics::try_new(&[1.0], 1, 2).unwrap_err(),
            BootstrapError::ShapeMismatch {
                observed: 1,
                expected: 2,
                panels: 1,
                statistics: 2,
            }
        );
        assert!(matches!(
            StationaryBootstrapSpec::try_new(2, 20.0, RandomStream::new(1, 2), u32::MAX),
            Err(BootstrapError::ReplicationPathRangeOverflow { .. })
        ));
        assert!(matches!(
            StationaryBootstrapSpec::try_new(1, 0.999, RandomStream::new(1, 2), 0),
            Err(BootstrapError::InvalidMeanBlockLength { .. })
        ));
    }

    #[test]
    fn block_uniform_mapping_is_strictly_inside_the_unit_interval() {
        let smallest = open_unit_midpoint_52(0);
        let largest = open_unit_midpoint_52(u64::MAX);
        assert!(smallest > 0.0);
        assert!(largest < 1.0);
        assert!(smallest < largest);
    }

    #[test]
    fn every_replication_contains_exactly_the_observed_panel_count() {
        let panels = 17;
        let values = vec![1.0; panels];
        let statistics = AdditivePanelStatistics::try_new(&values, panels, 1).unwrap();
        for mean in [1.0, 5.0, 20.0, 60.0] {
            let result = statistics
                .stationary_bootstrap_sums(spec(257, mean), &CpuExecutor::try_new(4).unwrap())
                .unwrap();
            assert!(
                result
                    .as_slice()
                    .iter()
                    .all(|&value| value.to_bits() == 17.0_f64.to_bits())
            );
        }
    }

    #[test]
    fn pins_complete_philox_origin_and_geometric_mapping_sequence() {
        let values = [1.0, 2.0, 4.0, 8.0, 16.0];
        let statistics = AdditivePanelStatistics::try_new(&values, 5, 1).unwrap();
        let result = statistics
            .stationary_bootstrap_sums(spec(8, 3.0), &CpuExecutor::try_new(1).unwrap())
            .unwrap();
        assert_eq!(
            result.as_slice(),
            &[36.0, 31.0, 31.0, 31.0, 19.0, 31.0, 17.0, 22.0]
        );
    }

    #[test]
    fn output_is_bitwise_invariant_to_executor_width() {
        let values = [
            1.0, -2.0, 0.25, 3.0, 1.5, -7.0, 2.0, 9.0, -4.0, 8.5, 6.0, 0.125,
        ];
        let statistics = AdditivePanelStatistics::try_new(&values, 6, 2).unwrap();
        let request = spec(1_003, 5.0);
        let serial = statistics
            .stationary_bootstrap_sums(request, &CpuExecutor::try_new(1).unwrap())
            .unwrap();
        let parallel = statistics
            .stationary_bootstrap_sums(request, &CpuExecutor::try_new(4).unwrap())
            .unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn stratified_bootstrap_preserves_every_stratum_and_executor_identity() {
        let values = [
            1.0, 10.0, 1.0, 1.0, 10.0, 2.0, 1.0, 100.0, 4.0, 1.0, 100.0, 8.0, 1.0, 100.0, 16.0,
            1.0, 100.0, 32.0,
        ];
        let statistics = AdditivePanelStatistics::try_new(&values, 6, 3).unwrap();
        let request = spec(1_003, 3.0);
        let serial = statistics
            .stationary_bootstrap_stratified_sums(
                &[0, 2, 6],
                request,
                &CpuExecutor::try_new(1).unwrap(),
            )
            .unwrap();
        let parallel = statistics
            .stationary_bootstrap_stratified_sums(
                &[0, 2, 6],
                request,
                &CpuExecutor::try_new(4).unwrap(),
            )
            .unwrap();
        assert_eq!(serial, parallel);
        assert!(serial.as_slice().chunks(3).all(|row| {
            row[0].to_bits() == 6.0_f64.to_bits() && row[1].to_bits() == 420.0_f64.to_bits()
        }));
        assert!(
            serial
                .as_slice()
                .chunks(3)
                .map(|row| row[2].to_bits())
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1
        );
    }

    #[test]
    fn stratified_bootstrap_rejects_incomplete_or_empty_partitions() {
        let values = [1.0, 2.0, 3.0];
        let statistics = AdditivePanelStatistics::try_new(&values, 3, 1).unwrap();
        let executor = CpuExecutor::try_new(1).unwrap();
        assert!(matches!(
            statistics.stationary_bootstrap_stratified_sums(&[0], spec(1, 2.0), &executor),
            Err(BootstrapError::StratumOffsetCount { observed: 1 })
        ));
        assert!(matches!(
            statistics.stationary_bootstrap_stratified_sums(&[1, 3], spec(1, 2.0), &executor),
            Err(BootstrapError::StratumOffsetsDoNotStartAtZero { observed: 1 })
        ));
        assert!(matches!(
            statistics
                .stationary_bootstrap_stratified_sums(&[0, 2, 2, 3], spec(1, 2.0), &executor,),
            Err(BootstrapError::InvalidStratumRange { stratum: 1, .. })
        ));
        assert!(matches!(
            statistics.stationary_bootstrap_stratified_sums(&[0, 2], spec(1, 2.0), &executor),
            Err(BootstrapError::StratumOffsetEndMismatch {
                observed: 2,
                expected: 3,
            })
        ));
    }

    #[test]
    fn prefix_kernel_matches_a_naive_panel_walk() {
        let values = [1.0, 10.0, 2.0, 20.0, 4.0, 40.0, 8.0, 80.0, 16.0, 160.0];
        let statistics = AdditivePanelStatistics::try_new(&values, 5, 2).unwrap();
        let request = spec(101, 3.0);
        let fast = statistics
            .stationary_bootstrap_sums(request, &CpuExecutor::try_new(1).unwrap())
            .unwrap();
        let mut naive = vec![0.0; fast.as_slice().len()];
        for (replication, output) in naive.chunks_mut(2).enumerate() {
            let mut stream = PhiloxPathStream::new(
                request.stream.seed(),
                request.stream.domain(),
                u32::try_from(replication).unwrap(),
                0,
            );
            let mut remaining = 5;
            while remaining != 0 {
                let origin = usize::try_from(sample_bounded_u64(&mut stream, 5)).unwrap();
                let length =
                    sample_geometric_length(&mut stream, request.mean_block_length, remaining);
                for offset in 0..length {
                    let panel = (origin + offset) % 5;
                    output[0] += values[panel * 2];
                    output[1] += values[panel * 2 + 1];
                }
                remaining -= length;
            }
        }
        assert_eq!(fast.as_slice(), naive);
    }

    #[test]
    fn stratified_prefix_kernel_matches_a_naive_within_stratum_walk() {
        let values = [1.0, 10.0, 2.0, 20.0, 4.0, 40.0, 8.0, 80.0, 16.0, 160.0];
        let statistics = AdditivePanelStatistics::try_new(&values, 5, 2).unwrap();
        let request = spec(101, 3.0);
        let offsets = [0_u64, 2, 5];
        let fast = statistics
            .stationary_bootstrap_stratified_sums(
                &offsets,
                request,
                &CpuExecutor::try_new(1).unwrap(),
            )
            .unwrap();
        let mut naive = vec![0.0; fast.as_slice().len()];
        for (replication, output) in naive.chunks_mut(2).enumerate() {
            let mut stream = PhiloxPathStream::new(
                request.stream.seed(),
                request.stream.domain(),
                u32::try_from(replication).unwrap(),
                0,
            );
            for pair in offsets.windows(2) {
                let start = usize::try_from(pair[0]).unwrap();
                let panels = usize::try_from(pair[1] - pair[0]).unwrap();
                let mut remaining = panels;
                while remaining != 0 {
                    let origin = usize::try_from(sample_bounded_u64(
                        &mut stream,
                        u64::try_from(panels).unwrap(),
                    ))
                    .unwrap();
                    let length =
                        sample_geometric_length(&mut stream, request.mean_block_length, remaining);
                    for offset in 0..length {
                        let panel = start + (origin + offset) % panels;
                        output[0] += values[panel * 2];
                        output[1] += values[panel * 2 + 1];
                    }
                    remaining -= length;
                }
            }
        }
        assert_eq!(fast.as_slice(), naive);
    }
}
