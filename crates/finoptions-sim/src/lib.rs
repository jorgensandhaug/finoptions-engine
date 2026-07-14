//! Exact event-aligned simulation for explicitly supported market models.

mod error;
mod executor;
mod path;
mod random;
mod step;

pub use error::SimulationError;
pub use finoptions_runtime::{CpuExecutor, CpuMathBackend, ExecutionError};
pub use path::{EscrowedGbmPathPlan, GbmPathPlan, SimulatedPaths};
pub use random::RandomStream;
pub use step::GbmStep;
