//! Exact event-aligned simulation for explicitly supported market models.

mod error;
mod executor;
mod path;
mod step;

pub use error::SimulationError;
pub use executor::CpuExecutor;
pub use path::{GbmPathPlan, SimulatedPaths};
pub use step::GbmStep;
