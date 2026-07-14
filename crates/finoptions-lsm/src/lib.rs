//! Independently fitted and evaluated least-squares Monte Carlo.

mod batch;
mod error;
mod policy;
mod regression;

pub use error::{LsmError, RegressionError};
pub use policy::{
    ExerciseFitDiagnostics, ExerciseNodeDiagnostics, ExerciseTieRule, FittedLsmCallPolicy,
    LsmCallEstimator, LsmEvaluation, LsmRun,
};
pub use regression::{
    MAX_POLYNOMIAL_DEGREE, PolynomialDegree, PolynomialRegressionSpec, RegressionDiagnostics,
};

pub use batch::{
    BatchDividendModel, BatchLsmCallInputs, BatchLsmCallMethod, BatchLsmCallResult,
    BatchLsmCallRowResult, BatchLsmError, BatchRandomStreams, OriginExercise, UniformExerciseMesh,
};
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub use regression::RegressionBenchmark;
