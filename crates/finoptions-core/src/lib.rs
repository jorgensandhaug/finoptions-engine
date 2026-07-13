//! Financial value types and validation shared by every engine backend.

mod curve;
mod dividend;
mod error;
mod time;

pub use curve::LogDiscountCurve;
pub use dividend::{CashDividend, DividendPolicy};
pub use error::InputError;
pub use time::{NodeId, TimeGrid};
