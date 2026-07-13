//! Typed construction and execution failures for simulation kernels.

use finoptions_core::InputError;
use thiserror::Error;

/// A failure to construct or execute an exact simulation plan.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SimulationError {
    /// Financial input validation failed.
    #[error(transparent)]
    Input(#[from] InputError),

    /// A requested transition skipped or reversed model nodes.
    #[error("GBM transition must join adjacent increasing nodes; observed {from} -> {to}")]
    NonAdjacentNodes {
        /// Start node ID.
        from: u32,
        /// End node ID.
        to: u32,
    },

    /// Curve and event schedule were validated against different time grids.
    #[error("discount curve and event schedule must use the identical time grid")]
    TimeGridMismatch,

    /// A simulation requested no paths.
    #[error("simulation path count must be positive")]
    EmptyPathSet,

    /// A CPU executor requested no worker threads.
    #[error("CPU executor thread count must be positive")]
    EmptyThreadPool,

    /// A dedicated CPU worker pool could not be created.
    #[error("failed to build CPU executor with {threads} threads: {reason}")]
    ThreadPoolBuild {
        /// Explicit requested worker count.
        threads: usize,
        /// Stable upstream failure message.
        reason: String,
    },

    /// A requested buffer shape overflowed the platform index domain.
    #[error("simulation shape overflow for {rows} rows and {paths} paths")]
    ShapeOverflow {
        /// Number of time/observation rows.
        rows: usize,
        /// Number of paths.
        paths: usize,
    },

    /// A caller-provided buffer has the wrong exact length.
    #[error("{buffer} buffer length mismatch: expected {expected}, observed {observed}")]
    BufferLengthMismatch {
        /// Stable buffer name.
        buffer: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        observed: usize,
    },

    /// The initial pre-event stock state is invalid.
    #[error("initial spot must be finite, nonnegative, and canonical; observed {observed:?}")]
    InvalidInitialSpot {
        /// Invalid state.
        observed: f64,
    },

    /// A post-event stock state is negative, NaN, or infinite.
    #[error("post-event spot must be finite and nonnegative; observed {observed:?}")]
    InvalidPostEventSpot {
        /// Invalid state.
        observed: f64,
    },

    /// A supplied standard-normal variate is NaN or infinite.
    #[error("standard-normal variate must be finite; observed {observed:?}")]
    NonFiniteNormalVariate {
        /// Invalid variate.
        observed: f64,
    },

    /// A normal buffer contains an invalid element.
    #[error(
        "standard-normal buffer contains a non-finite value at interval {interval}, path {path}: {observed:?}"
    )]
    NonFiniteNormalBuffer {
        /// Zero-based adjacent interval.
        interval: usize,
        /// Zero-based path.
        path: usize,
        /// Invalid variate.
        observed: f64,
    },

    /// A scalar transition overflowed or otherwise produced a non-finite stock state.
    #[error(
        "GBM transition from node {from} to {to} produced non-finite spot from state {spot:?} and variate {normal:?}"
    )]
    NonFiniteTransition {
        /// Start node.
        from: u32,
        /// End node.
        to: u32,
        /// Input post-event spot.
        spot: f64,
        /// Input standard-normal variate.
        normal: f64,
    },

    /// A batched transition overflowed or otherwise produced a non-finite stock state.
    #[error(
        "GBM path transition from node {from} to {to} at path {path} produced non-finite spot from state {spot:?} and variate {normal:?}"
    )]
    NonFinitePathTransition {
        /// Start node.
        from: u32,
        /// End node.
        to: u32,
        /// Zero-based path.
        path: usize,
        /// Input post-event spot.
        spot: f64,
        /// Input standard-normal variate.
        normal: f64,
    },
}
