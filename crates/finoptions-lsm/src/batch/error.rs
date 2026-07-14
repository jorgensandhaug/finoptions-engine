//! Typed failures for contract-parallel LSM batches.

use finoptions_core::InputError;
use finoptions_sim::SimulationError;
use thiserror::Error;

use crate::LsmError;

/// A fail-fast violation of one constant-parameter call batch or its execution.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum BatchLsmError {
    /// A batch contains no option contracts.
    #[error("LSM call batch must contain at least one contract")]
    EmptyBatch,

    /// A required per-contract column has the wrong row count.
    #[error("LSM batch field {field} has {observed} rows; expected {expected}")]
    RowCountMismatch {
        /// Stable field name.
        field: &'static str,
        /// Required row count.
        expected: usize,
        /// Supplied row count.
        observed: usize,
    },

    /// Group offsets do not contain one boundary per simulation group plus the terminal row.
    #[error("LSM grouped batch offsets contain {observed} values; expected {expected}")]
    GroupOffsetCountMismatch {
        /// Required number of offset values.
        expected: usize,
        /// Supplied number of offset values.
        observed: usize,
    },

    /// The first grouped contract offset is not zero.
    #[error("LSM grouped batch offsets must start at zero; observed {observed}")]
    GroupOffsetsDoNotStartAtZero {
        /// Supplied first boundary.
        observed: u32,
    },

    /// A grouped batch contains an empty or reversed contract range.
    #[error(
        "LSM grouped batch range {group} must be nonempty and increasing; observed [{start}, {end})"
    )]
    InvalidGroupRange {
        /// Zero-based simulation group.
        group: usize,
        /// Inclusive supplied row boundary.
        start: u32,
        /// Exclusive supplied row boundary.
        end: u32,
    },

    /// The final grouped offset does not equal the strike-row count.
    #[error("LSM grouped batch terminal offset is {observed}; expected {expected} contract rows")]
    GroupOffsetEndMismatch {
        /// Exact strike-row count.
        expected: usize,
        /// Supplied terminal boundary.
        observed: u32,
    },

    /// A per-contract scalar lies outside its declared financial domain.
    #[error("LSM batch field {field} is invalid at row {row}: observed {observed:?}")]
    InvalidRowValue {
        /// Stable field name.
        field: &'static str,
        /// Zero-based contract row.
        row: usize,
        /// Invalid scalar.
        observed: f64,
    },

    /// The uniform base time mesh contains no interval.
    #[error("LSM uniform exercise mesh must contain at least one base interval")]
    EmptyBaseIntervals,

    /// Independent evaluation cannot report sampling dispersion from one path.
    #[error("batched LSM evaluation requires at least two paths; observed {observed}")]
    InsufficientEvaluationPaths {
        /// Requested independent evaluation paths.
        observed: usize,
    },

    /// One path population exceeds the seeded simulator's path lane.
    #[error("batched LSM {population} path count {observed} exceeds maximum {maximum}")]
    PathCountOutsideDomain {
        /// Stable population role.
        population: &'static str,
        /// Requested paths.
        observed: usize,
        /// Largest supported count.
        maximum: u32,
    },

    /// A row-addressed random-domain interval exceeds the 32-bit lane.
    #[error(
        "batched LSM {population} random domains starting at {start} cannot address {rows} rows"
    )]
    RandomDomainRangeOverflow {
        /// Stable population role.
        population: &'static str,
        /// First requested domain.
        start: u32,
        /// Batch rows requiring unique domains.
        rows: usize,
    },

    /// Training and evaluation random-domain intervals overlap.
    #[error(
        "batched LSM training domains [{training_start}, {training_end}) overlap evaluation domains [{evaluation_start}, {evaluation_end})"
    )]
    RandomDomainRangeCollision {
        /// Inclusive first training domain.
        training_start: u64,
        /// Exclusive training domain end.
        training_end: u64,
        /// Inclusive first evaluation domain.
        evaluation_start: u64,
        /// Exclusive evaluation domain end.
        evaluation_end: u64,
    },

    /// A cash event is not strictly before its contract expiry.
    #[error(
        "cash dividend at batch row {row}, event {event}, time {time:?} must occur strictly before maturity {maturity:?}"
    )]
    CashDividendOutsideContract {
        /// Zero-based contract row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Event year fraction.
        time: f64,
        /// Contract maturity year fraction.
        maturity: f64,
    },

    /// Two deterministic cash declarations share one exact event time.
    #[error(
        "cash-dividend batch row {row} repeats exact event time {time:?} at event {event}; aggregation policy is not declared"
    )]
    DuplicateCashDividendTime {
        /// Zero-based contract row.
        row: usize,
        /// Second event index.
        event: usize,
        /// Repeated exact year fraction.
        time: f64,
    },

    /// Discounting an escrowed cash schedule overflowed.
    #[error(
        "escrowed prepaid cash is not finite at batch row {row}, event {event}: time={time:?}, amount={amount:?}, rate={rate:?}"
    )]
    NonFiniteEscrowedPrepaidCash {
        /// Zero-based contract row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Event year fraction.
        time: f64,
        /// Deterministic cash amount.
        amount: f64,
        /// Constant continuously compounded rate.
        rate: f64,
    },

    /// The observed stock does not leave a positive risky prepaid-forward component.
    #[error(
        "escrowed prepaid forward must be strictly positive at batch row {row}; initial spot is {initial_spot:?} and origin prepaid cash is {prepaid_cash:?}"
    )]
    NonPositiveEscrowedPrepaidForward {
        /// Zero-based contract row.
        row: usize,
        /// Observed cum-dividend stock value.
        initial_spot: f64,
        /// Present value of future deterministic cash dividends.
        prepaid_cash: f64,
    },

    /// Origin dollar-volatility matching produced an invalid prepaid-forward volatility.
    #[error(
        "escrowed spot-matched volatility is invalid at batch row {row}: supplied={supplied_volatility:?}, initial_spot={initial_spot:?}, prepaid_forward={prepaid_forward:?}, transformed={transformed:?}"
    )]
    InvalidEscrowedPrepaidForwardVolatility {
        /// Zero-based contract row.
        row: usize,
        /// Supplied spot volatility.
        supplied_volatility: f64,
        /// Observed stock value at origin.
        initial_spot: f64,
        /// Risky prepaid-forward value at origin.
        prepaid_forward: f64,
        /// Attempted transformed volatility.
        transformed: f64,
    },

    /// A row-specific union mesh exceeds the supported node domain.
    #[error(
        "LSM batch mesh shape overflow at row {row} for {base_intervals} base intervals and {cash_events} cash events"
    )]
    ContractMeshShapeOverflow {
        /// Zero-based contract row.
        row: usize,
        /// Requested uniform base intervals.
        base_intervals: u32,
        /// Exact cash-event count.
        cash_events: usize,
    },

    /// A row-specific exact union mesh could not reserve its declared memory.
    #[error("LSM batch could not allocate {nodes} mesh nodes at row {row}")]
    ContractMeshAllocationFailed {
        /// Zero-based contract row.
        row: usize,
        /// Maximum requested node count.
        nodes: usize,
    },

    /// The row-ordered result buffer could not reserve its declared memory.
    #[error("LSM batch could not allocate output storage for {rows} rows")]
    ResultAllocationFailed {
        /// Requested result rows.
        rows: usize,
    },

    /// The parallel grouped scheduler could not allocate its disjoint slice descriptors.
    #[error("LSM batch could not allocate parallel partition metadata for {groups} groups")]
    ParallelGroupPartitionAllocationFailed {
        /// Requested simulation groups.
        groups: usize,
    },

    /// An exact cash-event time was unexpectedly absent from its constructed union mesh.
    #[error(
        "cash dividend at batch row {row}, event {event}, time {time:?} is absent from its exact union mesh"
    )]
    CashDividendNodeMissing {
        /// Zero-based contract row.
        row: usize,
        /// Zero-based event within the row.
        event: usize,
        /// Exact event year fraction.
        time: f64,
    },

    /// Building an exact row-specific financial value failed.
    #[error("LSM batch financial input failed at row {row}: {source}")]
    FinancialInputAtRow {
        /// Zero-based contract row.
        row: usize,
        /// Underlying typed input failure.
        #[source]
        source: InputError,
    },

    /// Building an exact row-specific simulation plan failed.
    #[error("LSM batch simulation plan failed at row {row}: {source}")]
    SimulationPlanAtRow {
        /// Zero-based contract row.
        row: usize,
        /// Underlying typed simulation failure.
        #[source]
        source: SimulationError,
    },

    /// One independently fitted/evaluated row failed.
    #[error("LSM batch estimator failed at row {row}: {source}")]
    EstimatorAtRow {
        /// Zero-based contract row.
        row: usize,
        /// Underlying estimator failure.
        #[source]
        source: LsmError,
    },

    /// A diagnostic count could not be represented by its declared output type.
    #[error("LSM batch diagnostic {field} overflowed at row {row}")]
    DiagnosticOverflow {
        /// Stable diagnostic field.
        field: &'static str,
        /// Zero-based contract row.
        row: usize,
    },

    /// Independent evaluation omitted dispersion despite a multi-path request.
    #[error("LSM batch independent evaluation omitted {field} at row {row}")]
    MissingSamplingDiagnostic {
        /// Stable missing diagnostic.
        field: &'static str,
        /// Zero-based contract row.
        row: usize,
    },

    /// Stopping counts do not partition the independent evaluation population.
    #[error(
        "LSM batch stopping counts total {observed} at row {row}; evaluation contains {expected} paths"
    )]
    StoppingCountMismatch {
        /// Zero-based contract row.
        row: usize,
        /// Expected independent path count.
        expected: u32,
        /// Observed sum across exercise and expiry nodes.
        observed: u64,
    },

    /// Parallel failure replay did not reproduce the reported deterministic row failure.
    #[error(
        "parallel LSM batch reported row {row} as failed, but deterministic serial replay succeeded"
    )]
    ParallelFailureReplayMismatch {
        /// Lowest row reported as failing.
        row: usize,
    },
}
