//! Typed failures for regression and stopping-policy construction.

use thiserror::Error;

use finoptions_sim::SimulationError;

/// A fail-fast violation of the declared polynomial regression estimator.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum RegressionError {
    /// The requested polynomial degree has not passed the production evidence gate.
    #[error(
        "polynomial degree {observed} is unsupported; the validated maximum degree is {maximum}"
    )]
    UnsupportedDegree {
        /// Requested degree.
        observed: usize,
        /// Largest validated degree.
        maximum: usize,
    },

    /// The configured numerical-rank tolerance is invalid.
    #[error("rank tolerance must be finite and strictly between 0 and 1; observed {observed:?}")]
    InvalidRankTolerance {
        /// Invalid tolerance.
        observed: f64,
    },

    /// The configured condition-number limit is invalid or inconsistent with rank tolerance.
    #[error(
        "condition limit must be finite, at least 1, and no greater than 1/rank_tolerance ({maximum:?}); observed {observed:?}"
    )]
    InvalidConditionLimit {
        /// Invalid limit.
        observed: f64,
        /// Largest limit consistent with the rank tolerance.
        maximum: f64,
    },

    /// Predictor and response columns have different lengths.
    #[error("regression has {predictors} predictors but {responses} responses")]
    LengthMismatch {
        /// Predictor count.
        predictors: usize,
        /// Response count.
        responses: usize,
    },

    /// The declared basis has more columns than the regression population can support.
    #[error(
        "regression has {observed} observations but degree {degree} requires at least {required}"
    )]
    InsufficientObservations {
        /// Observed row count.
        observed: usize,
        /// Required row count.
        required: usize,
        /// Declared polynomial degree.
        degree: usize,
    },

    /// The regression population cannot be represented by the canonical path identifier.
    #[error("regression has {observed} observations, exceeding the supported maximum {maximum}")]
    TooManyObservations {
        /// Observed row count.
        observed: usize,
        /// Largest supported row count.
        maximum: u32,
    },

    /// A predictor is NaN or infinite.
    #[error("regression predictor {index} must be finite; observed {observed:?}")]
    NonFinitePredictor {
        /// Zero-based row index.
        index: usize,
        /// Invalid value.
        observed: f64,
    },

    /// A response is NaN or infinite.
    #[error("regression response {index} must be finite; observed {observed:?}")]
    NonFiniteResponse {
        /// Zero-based row index.
        index: usize,
        /// Invalid value.
        observed: f64,
    },

    /// A positive-degree fit has no representable predictor variation.
    #[error("degree {degree} regression predictor scale is degenerate; observed {scale:?}")]
    DegeneratePredictor {
        /// Declared polynomial degree.
        degree: usize,
        /// Computed population standard deviation.
        scale: f64,
    },

    /// The declared polynomial design is numerically rank deficient.
    #[error(
        "degree {degree} polynomial design has numerical rank {rank} of {columns} at rank tolerance {tolerance:?}"
    )]
    RankDeficient {
        /// Declared polynomial degree.
        degree: usize,
        /// Diagnosed numerical rank.
        rank: usize,
        /// Required basis-column count.
        columns: usize,
        /// Configured relative singular-value threshold.
        tolerance: f64,
    },

    /// A full-rank design exceeds the explicitly configured condition domain.
    #[error(
        "degree {degree} polynomial design condition number {observed:?} exceeds configured limit {limit:?}"
    )]
    ConditionLimitExceeded {
        /// Declared polynomial degree.
        degree: usize,
        /// Diagnosed two-norm condition number.
        observed: f64,
        /// Configured maximum.
        limit: f64,
    },

    /// The small one-sided Jacobi condition diagnostic did not converge.
    #[error(
        "polynomial condition diagnostic did not converge for {columns} columns after {sweeps} sweeps"
    )]
    ConditionDiagnosticDidNotConverge {
        /// Number of basis columns.
        columns: usize,
        /// Fixed sweep budget.
        sweeps: usize,
    },

    /// Fitting produced an unrepresentable intermediate or coefficient.
    #[error("polynomial regression produced non-finite arithmetic at basis degree {basis_degree}")]
    NonFiniteFit {
        /// Basis degree being processed.
        basis_degree: usize,
    },

    /// Evaluation produced an unrepresentable continuation value.
    #[error("polynomial regression produced non-finite prediction for predictor {predictor:?}")]
    NonFinitePrediction {
        /// Predictor passed to the fitted model.
        predictor: f64,
    },
}

/// A fail-fast violation of the independently fitted/evaluated LSM estimator.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum LsmError {
    /// The fixed call strike is outside the supported positive finite domain.
    #[error("call strike must be finite and strictly positive; observed {observed:?}")]
    InvalidStrike {
        /// Invalid strike.
        observed: f64,
    },

    /// The initial state does not match the simulated valuation-origin state.
    #[error("initial spot must be finite and nonnegative; observed {observed:?}")]
    InvalidInitialSpot {
        /// Invalid spot.
        observed: f64,
    },

    /// The curve used for risk-neutral path simulation differs from the LSM discount curve.
    #[error(
        "LSM discount curve must exactly match the curve used for risk-neutral path simulation"
    )]
    SimulationCurveMismatch,

    /// Evaluation dynamics differ from the complete plan used to fit the stopping policy.
    #[error("LSM evaluation paths must use the identical GBM path plan as training paths")]
    EvaluationPathPlanMismatch,

    /// Evaluation paths do not have the exact observation identity of the fitted policy.
    #[error("evaluation observation nodes differ from the fitted stopping policy")]
    ObservationNodeMismatch,

    /// A valuation-origin observation does not equal the declared initial state.
    #[error(
        "valuation-origin state at path {path} differs from declared initial spot {expected:?}; observed {observed:?}"
    )]
    InitialStateMismatch {
        /// Zero-based path index.
        path: usize,
        /// Declared initial state.
        expected: f64,
        /// Observed path state.
        observed: f64,
    },

    /// A private policy invariant was violated instead of producing a value.
    #[error("fitted policy contains an invalid origin exercise rule at row {row}, node {node}")]
    InvalidPolicyOriginRule {
        /// Zero-based observation row.
        row: usize,
        /// Observed node.
        node: u32,
    },

    /// A standalone path population exceeds the canonical 32-bit path identity domain.
    #[error(
        "LSM {population} population has {observed} paths, exceeding the supported maximum {maximum}"
    )]
    PathCountOutsideDomain {
        /// Stable population role.
        population: &'static str,
        /// Observed path count.
        observed: usize,
        /// Largest supported path count.
        maximum: u32,
    },

    /// A discount between LSM observation nodes cannot be represented.
    #[error(
        "discount factor from observation node {from} to {to} is not finite and positive; log discount is {log_discount:?}"
    )]
    UnrepresentableObservationDiscount {
        /// Earlier node.
        from: u32,
        /// Later node.
        to: u32,
        /// `log(P(0,to) / P(0,from))`.
        log_discount: f64,
    },

    /// A continuation regression failed at an exact exercise node.
    #[error("continuation regression failed at exercise node {node}: {source}")]
    RegressionAtNode {
        /// Exercise node.
        node: u32,
        /// Underlying numerical failure.
        #[source]
        source: RegressionError,
    },

    /// A fitted continuation model failed while making a path decision.
    #[error("continuation prediction failed at exercise node {node}, path {path}: {source}")]
    PredictionAtPath {
        /// Exercise node.
        node: u32,
        /// Zero-based path.
        path: usize,
        /// Underlying numerical failure.
        #[source]
        source: RegressionError,
    },

    /// Evaluation enters an ITM region absent from the training policy.
    #[error(
        "{active_in_the_money_paths} active evaluation paths are in the money at node {node}, but training had no in-the-money support"
    )]
    EvaluationOutsideTrainingSupport {
        /// Exercise node.
        node: u32,
        /// Active evaluation paths requiring a continuation model.
        active_in_the_money_paths: u32,
    },

    /// A discounted payoff or online sampling diagnostic became non-finite.
    #[error("LSM evaluation produced non-finite discounted payoff at path {path}")]
    NonFiniteDiscountedPayoff {
        /// Zero-based evaluation path.
        path: usize,
    },

    /// Fitting and evaluation were assigned the same random domain.
    #[error("LSM training and evaluation random domains must differ; both use domain {domain}")]
    RandomDomainCollision {
        /// Colliding counter-domain lane.
        domain: u32,
    },

    /// Training and evaluation use an identical seeded stream or external path matrix.
    #[error("LSM training and evaluation populations are identical, not independent")]
    TrainingEvaluationPopulationCollision,

    /// Seeded and external path identities cannot be compared without an explicit population ID.
    #[error(
        "cannot independently evaluate a policy fitted on {training} paths using {evaluation} paths"
    )]
    IncomparablePopulationIdentity {
        /// Training provenance class.
        training: &'static str,
        /// Evaluation provenance class.
        evaluation: &'static str,
    },

    /// Training-path simulation failed before policy fitting.
    #[error("LSM training simulation failed: {source}")]
    TrainingSimulation {
        /// Underlying simulation failure.
        #[source]
        source: SimulationError,
    },

    /// Evaluation-path simulation failed after policy fitting.
    #[error("LSM evaluation simulation failed: {source}")]
    EvaluationSimulation {
        /// Underlying simulation failure.
        #[source]
        source: SimulationError,
    },
}
