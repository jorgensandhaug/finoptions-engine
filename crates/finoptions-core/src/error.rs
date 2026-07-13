//! Typed input failures at the financial core boundary.

use thiserror::Error;

/// A violation of the explicitly configured financial input contract.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum InputError {
    /// A time grid has no valuation node.
    #[error("time grid must contain at least the valuation node")]
    EmptyTimeGrid,

    /// The first year fraction is not exactly the valuation origin.
    #[error("time grid must begin at exactly 0.0; observed {observed:?}")]
    InvalidTimeOrigin {
        /// Observed first year fraction.
        observed: f64,
    },

    /// A year fraction is NaN or infinite.
    #[error("time at node {node} must be finite; observed {observed:?}")]
    NonFiniteTime {
        /// Zero-based node index.
        node: usize,
        /// Invalid year fraction.
        observed: f64,
    },

    /// Adjacent node times are not strictly increasing.
    #[error(
        "time grid must be strictly increasing; node {previous_node} is {previous:?} and node {node} is {observed:?}"
    )]
    NonIncreasingTime {
        /// Previous zero-based node index.
        previous_node: usize,
        /// Previous year fraction.
        previous: f64,
        /// Current zero-based node index.
        node: usize,
        /// Current year fraction.
        observed: f64,
    },

    /// A node index cannot be represented by the canonical node ID.
    #[error("time grid contains {nodes} nodes, exceeding the supported node-ID domain")]
    TooManyNodes {
        /// Requested number of nodes.
        nodes: usize,
    },

    /// A node ID is outside a particular grid.
    #[error("node {node} is outside a time grid containing {nodes} nodes")]
    NodeOutOfBounds {
        /// Requested node ID.
        node: u32,
        /// Number of nodes in the grid.
        nodes: usize,
    },

    /// Curve and time-grid node counts differ.
    #[error("log-discount curve has {observed} nodes; time grid requires {expected}")]
    CurveLengthMismatch {
        /// Required node count.
        expected: usize,
        /// Supplied node count.
        observed: usize,
    },

    /// The curve does not have the exact valuation-origin discount factor.
    #[error("log discount at node 0 must be exactly 0.0; observed {observed:?}")]
    InvalidDiscountOrigin {
        /// Observed log discount factor.
        observed: f64,
    },

    /// A log discount is NaN or infinite.
    #[error("log discount at node {node} must be finite; observed {observed:?}")]
    NonFiniteLogDiscount {
        /// Zero-based node index.
        node: usize,
        /// Invalid log discount factor.
        observed: f64,
    },

    /// An interval discount factor is not representable as finite positive `f64`.
    #[error(
        "discount factor from node {from_node} to {to_node} is not finite and positive; interval log discount is {interval_log_discount:?}"
    )]
    UnrepresentableIntervalDiscount {
        /// Start node.
        from_node: usize,
        /// End node.
        to_node: usize,
        /// `log(P(0,t_to) / P(0,t_from))`.
        interval_log_discount: f64,
    },

    /// A cash dividend amount is negative, NaN, or infinite.
    #[error("cash dividend at node {node} must be finite and nonnegative; observed {observed:?}")]
    InvalidCashDividend {
        /// Dividend node.
        node: u32,
        /// Invalid cash amount.
        observed: f64,
    },

    /// A pre-event stock state is negative, NaN, or infinite.
    #[error(
        "spot before event at node {node} must be finite and nonnegative; observed {observed:?}"
    )]
    InvalidSpotBeforeEvent {
        /// Event node.
        node: u32,
        /// Invalid stock state.
        observed: f64,
    },

    /// The terminal expiry node is absent, is time zero, or is not the final node.
    #[error("expiry must be the final non-origin node {expected}; observed node {observed}")]
    InvalidExpiryNode {
        /// Required final node ID.
        expected: u32,
        /// Supplied expiry node ID.
        observed: u32,
    },

    /// More than one cash dividend was attached to the same event node.
    #[error("cash-dividend schedule repeats node {node}")]
    DuplicateDividendNode {
        /// Repeated node ID.
        node: u32,
    },

    /// Cash-dividend nodes are not supplied in strictly increasing order.
    #[error(
        "cash-dividend nodes must be strictly increasing; observed node {node} after {previous}"
    )]
    UnorderedDividendNode {
        /// Previous node ID.
        previous: u32,
        /// Current out-of-order node ID.
        node: u32,
    },

    /// A cash dividend is attached at valuation or at/after expiry.
    #[error(
        "cash dividend at node {node} must occur strictly after valuation and before expiry node {expiry}"
    )]
    InvalidDividendNode {
        /// Invalid dividend node.
        node: u32,
        /// Terminal expiry node.
        expiry: u32,
    },

    /// More than one exercise opportunity was attached to the same event node.
    #[error("exercise schedule repeats node {node}")]
    DuplicateExerciseNode {
        /// Repeated node ID.
        node: u32,
    },

    /// Exercise nodes are not supplied in strictly increasing order.
    #[error("exercise nodes must be strictly increasing; observed node {node} after {previous}")]
    UnorderedExerciseNode {
        /// Previous node ID.
        previous: u32,
        /// Current out-of-order node ID.
        node: u32,
    },

    /// An exercise decision is attached at/after terminal expiry.
    #[error("exercise opportunity at node {node} must occur before expiry node {expiry}")]
    InvalidExerciseNode {
        /// Invalid exercise node.
        node: u32,
        /// Terminal expiry node.
        expiry: u32,
    },

    /// Parallel schedule columns have different lengths.
    #[error("event schedule has {nodes} dividend nodes but {amounts} dividend amounts")]
    DividendColumnLengthMismatch {
        /// Number of dividend node IDs.
        nodes: usize,
        /// Number of dividend cash amounts.
        amounts: usize,
    },
}
