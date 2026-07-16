//! Typed construction and execution failures for simulation kernels.

use finoptions_core::InputError;
use finoptions_runtime::CpuMathError;
use thiserror::Error;

/// A failure to construct or execute an exact simulation plan.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SimulationError {
    /// Financial input validation failed.
    #[error(transparent)]
    Input(#[from] InputError),

    /// The requested named CPU math implementation could not honor the kernel contract.
    #[error(transparent)]
    CpuMath(#[from] CpuMathError),

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

    /// Continuous dividend-yield curve and event schedule use different time grids.
    #[error("continuous dividend-yield curve and event schedule must use the identical time grid")]
    DividendYieldTimeGridMismatch,

    /// A simulation requested no paths.
    #[error("simulation path count must be positive")]
    EmptyPathSet,

    /// A seeded simulation exceeded the explicit 32-bit path counter lane.
    #[error(
        "seeded simulation supports at most {maximum} paths in one random domain; observed {paths}"
    )]
    RandomPathNamespaceOverflow {
        /// Requested path count.
        paths: usize,
        /// Maximum path count supported by the counter layout.
        maximum: u32,
    },

    /// A deterministic diagnostic replay did not reproduce a parallel worker failure.
    #[error(
        "parallel seeded simulation reported path {path} as failed, but deterministic diagnostic replay succeeded"
    )]
    ParallelFailureReplayMismatch {
        /// Lowest path reported as failing by the parallel kernel.
        path: usize,
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

    /// The deterministic cash component exhausts or exceeds the observed stock value.
    #[error(
        "escrowed-dividend prepaid forward must be finite and strictly positive; initial spot is {initial_spot:?} and origin prepaid cash is {prepaid_cash:?}"
    )]
    NonPositiveEscrowedPrepaidForward {
        /// Observed cum-dividend stock value at valuation origin.
        initial_spot: f64,
        /// Deterministic present value of the future cash schedule.
        prepaid_cash: f64,
    },

    /// Backward valuation of the deterministic cash component overflowed.
    #[error(
        "escrowed-dividend prepaid cash is not finite at node {node}; discounted later cash is {discounted_future:?} and same-node cash is {same_node_cash:?}"
    )]
    NonFiniteEscrowedPrepaidCash {
        /// Grid node where valuation failed.
        node: u32,
        /// Later cash discounted back from the adjacent node.
        discounted_future: f64,
        /// Cash paid at this node after the pre-event observation.
        same_node_cash: f64,
    },

    /// Adding deterministic prepaid cash to the risky state overflowed.
    #[error(
        "escrowed-dividend observation at node {node}, path {path} is not finite; prepaid-forward state is {prepaid_forward:?} and prepaid cash is {prepaid_cash:?}"
    )]
    NonFiniteEscrowedObservation {
        /// Observation node.
        node: u32,
        /// Zero-based path.
        path: usize,
        /// Simulated risky prepaid-forward state.
        prepaid_forward: f64,
        /// Deterministic cash component immediately before a same-node event.
        prepaid_cash: f64,
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

    /// A seeded path requested more normals than its exact interval budget.
    #[error(
        "random-normal budget exhausted at interval {interval}, path {path}; this indicates an internal estimator-contract violation"
    )]
    RandomNormalBudgetExhausted {
        /// Zero-based adjacent interval.
        interval: usize,
        /// Zero-based path.
        path: usize,
    },

    /// Bounded ZIGNOR generation did not accept a variate within its declared limit.
    #[error(
        "random-normal rejection limit reached at interval {interval}, path {path}; no variate or fallback was produced"
    )]
    RandomNormalRejectionLimit {
        /// Zero-based adjacent interval.
        interval: usize,
        /// Zero-based path.
        path: usize,
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
