//! Typed failures for ragged cash-dividend batches and derived model tensors.

use thiserror::Error;

/// A structural, financial-domain, shape, or precision violation in a batch request.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum BatchError {
    /// Offsets cannot describe even one row.
    #[error("ragged offsets must contain at least two values; observed {observed}")]
    InsufficientOffsets {
        /// Supplied offset count.
        observed: usize,
    },

    /// Event-time and cash-amount buffers have different lengths.
    #[error("event-time buffer has {times} values but cash-amount buffer has {amounts}")]
    EventColumnLengthMismatch {
        /// Supplied time count.
        times: usize,
        /// Supplied amount count.
        amounts: usize,
    },

    /// One Arrow-style offset is negative.
    #[error("ragged offset {index} must be nonnegative; observed {observed}")]
    NegativeOffset {
        /// Zero-based offset index.
        index: usize,
        /// Invalid offset.
        observed: i64,
    },

    /// Adjacent offsets are decreasing.
    #[error(
        "ragged offsets must be nondecreasing; offset {previous_index} is {previous} and offset {index} is {observed}"
    )]
    DecreasingOffset {
        /// Previous zero-based offset index.
        previous_index: usize,
        /// Previous offset.
        previous: i64,
        /// Current zero-based offset index.
        index: usize,
        /// Invalid current offset.
        observed: i64,
    },

    /// The supplied child slice does not match the first-to-last offset span.
    #[error(
        "ragged offset span [{first}, {last}) contains {offset_events} events but child buffers contain {buffer_events}"
    )]
    OffsetSpanMismatch {
        /// First Arrow-style offset.
        first: i64,
        /// Final Arrow-style offset.
        last: i64,
        /// Event count implied by the offsets.
        offset_events: usize,
        /// Supplied child-buffer event count.
        buffer_events: usize,
    },

    /// An offset cannot be represented in this process's index domain.
    #[error("ragged offset {index} cannot be represented as a native index; observed {observed}")]
    OffsetIndexOverflow {
        /// Zero-based offset index.
        index: usize,
        /// Unrepresentable offset.
        observed: i64,
    },

    /// A future dividend time is non-finite, non-positive, or negative zero.
    #[error(
        "cash-dividend time at row {row}, event {event} must be finite and strictly positive; observed {observed:?}"
    )]
    InvalidEventTime {
        /// Zero-based batch row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Invalid year fraction.
        observed: f64,
    },

    /// Event times decrease within one row.
    #[error(
        "cash-dividend times must be nondecreasing; row {row}, event {event} is {observed:?} after {previous:?}"
    )]
    DecreasingEventTime {
        /// Zero-based batch row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Previous year fraction.
        previous: f64,
        /// Invalid current year fraction.
        observed: f64,
    },

    /// A deterministic cash amount is negative, negative zero, or non-finite.
    #[error(
        "cash-dividend amount at row {row}, event {event} must be finite, nonnegative, and canonical; observed {observed:?}"
    )]
    InvalidCashAmount {
        /// Zero-based batch row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Invalid cash amount.
        observed: f64,
    },

    /// A row-valued input has the wrong exact length.
    #[error("{field} has {observed} rows; dividend batch requires {expected}")]
    RowCountMismatch {
        /// Stable input name.
        field: &'static str,
        /// Required row count.
        expected: usize,
        /// Supplied row count.
        observed: usize,
    },

    /// An option maturity is not a finite positive year fraction.
    #[error("maturity at row {row} must be finite and strictly positive; observed {observed:?}")]
    InvalidMaturity {
        /// Zero-based batch row.
        row: usize,
        /// Invalid year fraction.
        observed: f64,
    },

    /// An event lies beyond the option maturity.
    #[error(
        "cash-dividend time at row {row}, event {event} is {event_time:?}, after maturity {maturity:?}"
    )]
    EventAfterMaturity {
        /// Zero-based batch row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Event year fraction.
        event_time: f64,
        /// Row maturity.
        maturity: f64,
    },

    /// An event equals maturity under a strict-before policy.
    #[error(
        "cash-dividend time at row {row}, event {event} equals maturity {maturity:?}, but the request requires events strictly before maturity"
    )]
    EventAtExcludedMaturity {
        /// Zero-based batch row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Row maturity and event year fraction.
        maturity: f64,
    },

    /// A required per-row spot is invalid.
    #[error("spot at row {row} must be finite and strictly positive; observed {observed:?}")]
    InvalidSpot {
        /// Zero-based batch row.
        row: usize,
        /// Invalid spot.
        observed: f64,
    },

    /// A padded-sequence request has no slots.
    #[error("padded dividend capacity must be positive")]
    EmptySequenceCapacity,

    /// A requested sequence width cannot be represented by the output length dtype.
    #[error("padded dividend capacity exceeds the unsigned 32-bit domain; observed {observed}")]
    SequenceCapacityOverflow {
        /// Invalid requested capacity.
        observed: usize,
    },

    /// A row contains more events than the exact requested capacity.
    #[error("row {row} contains {events} cash dividends, exceeding requested capacity {capacity}")]
    SequenceCapacityExceeded {
        /// Zero-based batch row.
        row: usize,
        /// Event count in this row.
        events: usize,
        /// Requested slots per row.
        capacity: usize,
    },

    /// A row event count cannot be represented in the output length dtype.
    #[error("row {row} contains {events} events, exceeding the unsigned 32-bit length domain")]
    EventCountOverflow {
        /// Zero-based batch row.
        row: usize,
        /// Event count in this row.
        events: usize,
    },

    /// A fixed-width time-bin request has no bins.
    #[error("dividend bin count must be positive")]
    EmptyBinCount,

    /// A requested bin count cannot be represented exactly by the kernel's index arithmetic.
    #[error("dividend bin count exceeds the unsigned 32-bit domain; observed {observed}")]
    BinCountOverflow {
        /// Invalid requested bin count.
        observed: usize,
    },

    /// A binning horizon is not a finite positive year fraction.
    #[error("dividend binning horizon must be finite and strictly positive; observed {observed:?}")]
    InvalidHorizon {
        /// Invalid year fraction.
        observed: f64,
    },

    /// An option maturity exceeds the fixed binning horizon.
    #[error("maturity at row {row} is {maturity:?}, beyond binning horizon {horizon:?}")]
    MaturityAfterHorizon {
        /// Zero-based batch row.
        row: usize,
        /// Row maturity.
        maturity: f64,
        /// Configured fixed horizon.
        horizon: f64,
    },

    /// An output shape exceeds the platform index domain.
    #[error("{output} shape overflows for {rows} rows and width {width}")]
    ShapeOverflow {
        /// Stable output name.
        output: &'static str,
        /// Row count.
        rows: usize,
        /// Per-row output width.
        width: usize,
    },

    /// A requested owned output buffer could not reserve memory.
    #[error("failed to reserve {elements} elements for {output}")]
    AllocationFailure {
        /// Stable output name.
        output: &'static str,
        /// Exact requested element count.
        elements: usize,
    },

    /// An exact f64 source value is outside the declared float32 model-tensor contract.
    #[error(
        "{field} at row {row}, element {element} cannot be safely represented as float32; observed {observed:?}"
    )]
    UnsafeFloat32Cast {
        /// Stable value name.
        field: &'static str,
        /// Zero-based batch row.
        row: usize,
        /// Zero-based element within the row.
        element: usize,
        /// Rejected exact value.
        observed: f64,
    },

    /// Float32 accumulation overflowed despite valid source values.
    #[error("float32 dividend-bin sum overflowed at row {row}, bin {bin}")]
    BinSumOverflow {
        /// Zero-based batch row.
        row: usize,
        /// Zero-based chronological bin.
        bin: usize,
    },

    /// Deterministic replay could not reproduce a parallel row failure.
    #[error(
        "parallel dividend-tensor kernel reported row {row} as failed, but deterministic replay succeeded"
    )]
    ParallelFailureReplayMismatch {
        /// Lowest row reported as failing.
        row: usize,
    },

    /// A native amount-scaling request has inconsistent arguments.
    #[error("amount scaling policy {policy:?} requires {requirement}")]
    InvalidScaleArguments {
        /// Stable scaling policy name.
        policy: String,
        /// Stable argument requirement.
        requirement: &'static str,
    },
}
