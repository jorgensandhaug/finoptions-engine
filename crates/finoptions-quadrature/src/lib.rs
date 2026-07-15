//! Deterministic event-aligned direct integration for explicit finite exercise schedules.

mod call;
mod error;
mod rule;

pub use call::{EventAlignedQuadrature, EventAlignedQuadratureSpec, QuadratureCallResult};
pub use error::QuadratureError;
