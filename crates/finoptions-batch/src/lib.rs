//! Validated ragged financial-event batches and consumer-specific tensor kernels.
//!
//! Exact source schedules remain borrowed `i64` offsets plus `f64` values. Dense float32 outputs
//! are created only for a named model consumer and only after financial-domain, shape, and
//! precision validation.

mod error;
mod tensor;
mod view;

pub use error::BatchError;
pub use tensor::{
    BinnedDividendBatch, CashAmountScale, DividendBinSpec, MaturityBoundary, PaddedDividendBatch,
    PaddedSequenceSpec, TimeBinClosure, bin_dividends_f32, pack_padded_dividends_f32,
};
pub use view::{CashDividendRow, RaggedCashDividends};
