//! Non-production independent financial references and fixtures.

mod interpolated_crr;
mod one_dividend;

pub use interpolated_crr::{InterpolatedCrrError, interpolated_crr_call};
pub use one_dividend::{OneDividendReferenceError, one_dividend_american_call};

/// Identifies the testkit without exposing it from the Python package.
pub const TESTKIT_REVISION: u32 = 2;
