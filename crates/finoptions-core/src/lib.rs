//! Financial value types and validation shared by every engine backend.

mod curve;
mod dividend;
mod error;
mod event;
mod market;
mod time;

pub use curve::LogDiscountCurve;
pub use dividend::{CashDividend, DividendPolicy};
pub use error::InputError;
pub use event::{EventSchedule, NodePhases};
pub use market::ConstantVolatility;
pub use time::{NodeId, TimeGrid};
