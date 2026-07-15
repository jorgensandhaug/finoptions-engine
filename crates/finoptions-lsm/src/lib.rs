//! Independently fitted and evaluated least-squares Monte Carlo.

mod batch;
mod error;
mod policy;
mod regression;

pub use error::{LsmError, RegressionError};
pub use policy::{
    ExerciseFitDiagnostics, ExerciseNodeDiagnostics, ExerciseTieRule, FittedLsmCallPolicy,
    InsufficientExerciseSupportPolicy, LsmCallEstimator, LsmEvaluation, LsmRun,
};
pub use regression::{
    MAX_POLYNOMIAL_DEGREE, PolynomialDegree, PolynomialRegressionSpec, RegressionDiagnostics,
};

pub use batch::{
    BatchDividendModel, BatchExerciseMesh, BatchLsmCallInputs, BatchLsmCallMethod,
    BatchLsmCallResult, BatchLsmCallRowResult, BatchLsmError, BatchRandomStreams,
    CashDividendExerciseMesh, EscrowedVolatilityConvention, GroupedBatchLsmCallInputs,
    OriginExercise, UniformExerciseMesh,
};
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub use regression::{RegressionBenchmark, RegressionPredictionBenchmark};
