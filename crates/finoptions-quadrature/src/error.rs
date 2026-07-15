//! Typed construction and valuation failures for direct quadrature.

use finoptions_core::InputError;
use finoptions_runtime::CpuMathError;
use finoptions_sim::SimulationError;
use thiserror::Error;

/// An event-aligned quadrature request cannot preserve its declared semantics.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum QuadratureError {
    /// A shared financial value type rejected an input.
    #[error(transparent)]
    FinancialInput(#[from] InputError),

    /// The schedule and discount curve do not own the same exact grid.
    #[error("event schedule and log-discount curve use different time grids")]
    DiscountTimeGridMismatch,

    /// The schedule and continuous-yield curve do not own the same exact grid.
    #[error("event schedule and cumulative dividend-yield curve use different time grids")]
    DividendYieldTimeGridMismatch,

    /// The initial stock state is outside the supported nonnegative finite domain.
    #[error("initial spot must be finite, nonnegative, and canonical at zero; observed {observed}")]
    InvalidInitialSpot {
        /// Rejected stock state.
        observed: f64,
    },

    /// The call strike is not finite and strictly positive.
    #[error("call strike must be finite and strictly positive; observed {observed}")]
    InvalidStrike {
        /// Rejected strike.
        observed: f64,
    },

    /// The declared upper boundary is not finite and strictly positive.
    #[error("upper stock boundary must be finite and strictly positive; observed {observed}")]
    InvalidUpperStockBoundary {
        /// Rejected boundary.
        observed: f64,
    },

    /// The declared upper boundary does not contain the requested spot and strike.
    #[error(
        "upper stock boundary {upper} must be strictly above initial spot {initial_spot} and strike {strike}"
    )]
    UpperStockBoundaryTooLow {
        /// Declared upper boundary.
        upper: f64,
        /// Initial spot.
        initial_spot: f64,
        /// Call strike.
        strike: f64,
    },

    /// The uniform stock grid is too small for this reference method.
    #[error("stock grid requires at least {minimum} points; observed {observed}")]
    TooFewStockPoints {
        /// Minimum admitted point count.
        minimum: usize,
        /// Rejected point count.
        observed: usize,
    },

    /// The stock grid exceeds the exact integer-to-f64 coordinate domain.
    #[error("stock grid supports at most {maximum} points; observed {observed}")]
    TooManyStockPoints {
        /// Largest admitted point count.
        maximum: usize,
        /// Rejected point count.
        observed: usize,
    },

    /// The requested normal-quadrature order is not one of the version-pinned rules.
    #[error("standard-normal Gauss-Hermite order must be one of {supported}; observed {observed}")]
    UnsupportedQuadratureOrder {
        /// Rejected order.
        observed: usize,
        /// Stable list of admitted orders.
        supported: &'static str,
    },

    /// Golub-Welsch produced an invalid standard-normal rule.
    #[error("invalid standard-normal quadrature rule at order {order}: {reason}")]
    InvalidNormalRule {
        /// Requested quadrature order.
        order: usize,
        /// Stable validation failure.
        reason: &'static str,
    },

    /// A caller-sized work buffer could not be reserved.
    #[error("could not reserve {elements} f64 elements for quadrature buffer {buffer}")]
    Allocation {
        /// Stable buffer identity.
        buffer: &'static str,
        /// Requested element count.
        elements: usize,
    },

    /// Exact GBM transition construction failed.
    #[error(transparent)]
    Simulation(#[from] SimulationError),

    /// The explicitly requested exponential implementation failed.
    #[error(transparent)]
    CpuMath(#[from] CpuMathError),

    /// A finite exponent or stock state overflowed during one quadrature transition.
    #[error(
        "non-finite quadrature transition at interval {interval}, stock-grid point {stock_point}, normal node {normal_node}"
    )]
    NonFiniteTransition {
        /// Zero-based time interval.
        interval: usize,
        /// Zero-based stock-grid point.
        stock_point: usize,
        /// Zero-based normal-quadrature node.
        normal_node: usize,
    },

    /// Backward expectation arithmetic produced a non-finite value.
    #[error("non-finite quadrature value at interval {interval}, stock-grid point {stock_point}")]
    NonFiniteValue {
        /// Zero-based time interval.
        interval: usize,
        /// Zero-based stock-grid point.
        stock_point: usize,
    },

    /// The explicit affine upper boundary is already negative at the declared boundary.
    #[error(
        "affine upper-boundary value is negative or non-finite at node {node}: upper {upper}, value {value}"
    )]
    InvalidAffineBoundary {
        /// Event node at which the boundary was checked.
        node: u32,
        /// Declared upper stock boundary.
        upper: f64,
        /// Rejected affine value.
        value: f64,
    },

    /// The final interpolation produced an invalid option value.
    #[error("event-aligned quadrature produced invalid call value {observed}")]
    InvalidResult {
        /// Rejected result.
        observed: f64,
    },
}
