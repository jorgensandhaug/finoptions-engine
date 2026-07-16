//! Deterministic, domain-explicit American-option methods.

mod andersen_lake;
mod error;
mod legendre;

pub use andersen_lake::{
    AndersenLake, AndersenLakeBatchResult, AndersenLakeGroupDiagnostics, AndersenLakeSpec,
    GroupedContinuousYieldCalls,
};
pub use error::AmericanOptionError;
