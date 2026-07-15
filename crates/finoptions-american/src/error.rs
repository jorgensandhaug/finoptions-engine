//! Typed failures for deterministic American-option methods.

use thiserror::Error;

/// A deterministic American-option request or computation is invalid.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum AmericanOptionError {
    /// The fixed-point boundary iteration count is outside the admitted implementation range.
    #[error("Andersen-Lake iterations must be in 1..={maximum}; observed {observed}")]
    InvalidIterations {
        /// Largest admitted fixed iteration count.
        maximum: usize,
        /// Rejected count.
        observed: usize,
    },

    /// The Chebyshev boundary collocation count is outside the admitted implementation range.
    #[error(
        "Andersen-Lake collocation nodes must be in {minimum}..={maximum}; observed {observed}"
    )]
    InvalidCollocationNodes {
        /// Smallest admitted positive-time node count.
        minimum: usize,
        /// Largest admitted positive-time node count.
        maximum: usize,
        /// Rejected count.
        observed: usize,
    },

    /// A requested Gauss-Legendre rule is not one of the version-pinned rules.
    #[error("Gauss-Legendre order must be one of {supported}; observed {observed}")]
    UnsupportedLegendreOrder {
        /// Rejected quadrature order.
        observed: usize,
        /// Stable list of supported orders.
        supported: &'static str,
    },

    /// A checked-in quadrature rule failed its construction-time invariants.
    #[error("invalid checked-in Gauss-Legendre rule of order {order}: {reason}")]
    InvalidLegendreRule {
        /// Rule order.
        order: usize,
        /// Stable invariant description.
        reason: &'static str,
    },

    /// The caller's strict boundary residual tolerance is invalid.
    #[error(
        "maximum relative boundary residual must be finite and strictly positive; observed {observed}"
    )]
    InvalidBoundaryResidualTolerance {
        /// Rejected tolerance.
        observed: f64,
    },

    /// A caller's explicit normalized obstacle-projection bound is invalid.
    #[error(
        "maximum relative obstacle projection must be finite and nonnegative; observed {observed}"
    )]
    InvalidObstacleProjectionTolerance {
        /// Rejected tolerance.
        observed: f64,
    },

    /// Group offsets do not contain at least a start and end position.
    #[error("grouped American-call inputs require at least two offsets; observed {observed}")]
    InsufficientOffsets {
        /// Offset count.
        observed: usize,
    },

    /// One offset is negative.
    #[error("group offset {index} is negative: {observed}")]
    NegativeOffset {
        /// Offset position.
        index: usize,
        /// Rejected value.
        observed: i64,
    },

    /// Group offsets decrease.
    #[error(
        "group offsets decrease from position {previous_index} ({previous}) to {index} ({observed})"
    )]
    DecreasingOffset {
        /// Previous position.
        previous_index: usize,
        /// Previous offset.
        previous: i64,
        /// Current position.
        index: usize,
        /// Rejected current offset.
        observed: i64,
    },

    /// One offset cannot be represented as a platform index.
    #[error("group offset {index} cannot be represented as usize: {observed}")]
    OffsetIndexOverflow {
        /// Offset position.
        index: usize,
        /// Rejected value.
        observed: i64,
    },

    /// Offsets span a different number of strikes than the supplied child slice.
    #[error(
        "group offsets span {offset_strikes} strikes from {first} to {last}, but the strike buffer has {buffer_strikes}"
    )]
    OffsetSpanMismatch {
        /// First offset.
        first: i64,
        /// Last offset.
        last: i64,
        /// Child count implied by offsets.
        offset_strikes: usize,
        /// Supplied child count.
        buffer_strikes: usize,
    },

    /// Group-level parameter columns do not have one common length.
    #[error("group column {column} has length {observed}; expected {expected} from the offsets")]
    GroupColumnLengthMismatch {
        /// Stable column name.
        column: &'static str,
        /// Required group count.
        expected: usize,
        /// Supplied count.
        observed: usize,
    },

    /// One group has no strikes.
    #[error("American-call group {group} has no strikes")]
    EmptyStrikeGroup {
        /// Rejected group.
        group: usize,
    },

    /// One initial spot is invalid.
    #[error("American-call group {group} spot must be finite and positive; observed {observed}")]
    InvalidSpot {
        /// Rejected group.
        group: usize,
        /// Rejected value.
        observed: f64,
    },

    /// One maturity is invalid.
    #[error(
        "American-call group {group} maturity must be finite and positive; observed {observed}"
    )]
    InvalidMaturity {
        /// Rejected group.
        group: usize,
        /// Rejected value.
        observed: f64,
    },

    /// One risk-free rate is invalid.
    #[error("American-call group {group} rate must be finite; observed {observed}")]
    InvalidRate {
        /// Rejected group.
        group: usize,
        /// Rejected value.
        observed: f64,
    },

    /// One proportional dividend yield is outside this solver's proven call domain.
    #[error(
        "American-call group {group} dividend yield must be finite and nonnegative; observed {observed}"
    )]
    InvalidDividendYield {
        /// Rejected group.
        group: usize,
        /// Rejected value.
        observed: f64,
    },

    /// One volatility is invalid.
    #[error(
        "American-call group {group} volatility must be finite and positive; observed {observed}"
    )]
    InvalidVolatility {
        /// Rejected group.
        group: usize,
        /// Rejected value.
        observed: f64,
    },

    /// One strike is invalid.
    #[error(
        "American-call group {group} strike {strike} must be finite and positive; observed {observed}"
    )]
    InvalidStrike {
        /// Rejected group.
        group: usize,
        /// Within-group strike position.
        strike: usize,
        /// Rejected value.
        observed: f64,
    },

    /// A caller-sized buffer could not be allocated.
    #[error("could not reserve {elements} elements for American-option buffer {buffer}")]
    Allocation {
        /// Stable buffer identity.
        buffer: &'static str,
        /// Requested element count.
        elements: usize,
    },

    /// The fixed-point boundary left its financially admissible interpolation domain.
    #[error(
        "Andersen-Lake boundary is invalid in group {group}, iteration {iteration}, node {node}: {observed} (origin limit {origin_limit})"
    )]
    InvalidBoundary {
        /// Failing group.
        group: usize,
        /// Zero-based completed iteration count.
        iteration: usize,
        /// Positive-time collocation node.
        node: usize,
        /// Rejected normalized boundary.
        observed: f64,
        /// Exact short-expiry right limit used by the interpolation transform.
        origin_limit: f64,
    },

    /// Boundary interpolation produced an invalid transformed value.
    #[error(
        "Andersen-Lake boundary interpolation produced H={observed} in group {group} at time {time}"
    )]
    InvalidBoundaryInterpolation {
        /// Failing group.
        group: usize,
        /// Requested time to maturity.
        time: f64,
        /// Invalid transformed boundary value.
        observed: f64,
    },

    /// Fixed iterations did not satisfy the caller-declared residual gate.
    #[error(
        "Andersen-Lake boundary residual {observed} exceeds tolerance {tolerance} in group {group}"
    )]
    BoundaryResidualExceeded {
        /// Failing group.
        group: usize,
        /// Maximum nodewise residual normalized by put strike.
        observed: f64,
        /// Caller-declared maximum.
        tolerance: f64,
    },

    /// Restoring the exact stopping obstacle would require more than the declared numerical bound.
    #[error(
        "Andersen-Lake origin obstacle projection {observed} exceeds tolerance {tolerance} in group {group}, strike {strike}"
    )]
    ObstacleProjectionExceeded {
        /// Failing group.
        group: usize,
        /// Within-group strike position.
        strike: usize,
        /// Required projection divided by original call spot.
        observed: f64,
        /// Caller-declared maximum.
        tolerance: f64,
    },

    /// Valid finite inputs generated an invalid price or financial bound violation.
    #[error(
        "Andersen-Lake produced invalid price {observed} in group {group}, strike {strike}; required [{lower_bound}, {upper_bound}]"
    )]
    InvalidPrice {
        /// Failing group.
        group: usize,
        /// Within-group strike position.
        strike: usize,
        /// Invalid call price.
        observed: f64,
        /// American intrinsic lower bound.
        lower_bound: f64,
        /// Stock upper bound.
        upper_bound: f64,
    },

    /// Internal parallel sharding failed to preserve exact output shape.
    #[error("American-call shard output has {observed} {field} elements; expected {expected}")]
    ShardShapeMismatch {
        /// Stable output field.
        field: &'static str,
        /// Required count.
        expected: usize,
        /// Produced count.
        observed: usize,
    },
}
