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
            |output| self.fill_serial(&prefix, spec, output),
            |output| self.fill_parallel(&prefix, spec, output),
        )?;
        Ok(PanelBootstrapSums {
            values,
            replications,
            statistics: self.statistics,
        })
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
        spec: StationaryBootstrapSpec,
        values: &mut [f64],
    ) -> Result<(), BootstrapError> {
        for (replication, output) in values.chunks_mut(self.statistics).enumerate() {
            self.fill_replication(prefix, spec, replication, output)?;
        }
        Ok(())
    }

    fn fill_parallel(
        self,
        prefix: &[f64],
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
                    .fill_replication(prefix, spec, replication, output)
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
        let mut remaining = self.panels;
        while remaining != 0 {
            let origin = usize::try_from(sample_bounded_u64(
                &mut stream,
                u64::try_from(self.panels).expect("panel count fits u64"),
            ))
            .expect("sampled origin fits usize");
            let length = sample_geometric_length(&mut stream, spec.mean_block_length, remaining);
            self.add_circular_block(prefix, origin, length, output);
            remaining -= length;
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

    fn add_circular_block(self, prefix: &[f64], origin: usize, length: usize, output: &mut [f64]) {
        debug_assert!(origin < self.panels);
        debug_assert!((1..=self.panels).contains(&length));
        debug_assert_eq!(output.len(), self.statistics);
        let end = origin + length;
        if end <= self.panels {
            let origin_start = origin * self.statistics;
            let end_start = end * self.statistics;
            for statistic in 0..self.statistics {
                output[statistic] +=
                    prefix[end_start + statistic] - prefix[origin_start + statistic];
            }
            return;
        }
        let origin_start = origin * self.statistics;
        let panel_end_start = self.panels * self.statistics;
        let wrapped_end_start = (end - self.panels) * self.statistics;
        for statistic in 0..self.statistics {
            let tail = prefix[panel_end_start + statistic] - prefix[origin_start + statistic];
            output[statistic] += tail + prefix[wrapped_end_start + statistic];
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
}
