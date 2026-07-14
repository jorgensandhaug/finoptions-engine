//! Borrowed, validated inputs and immutable method identity for call batches.

use finoptions_batch::RaggedCashDividends;
use finoptions_core::DividendPolicy;

use crate::{ExerciseTieRule, PolynomialRegressionSpec};

use super::BatchLsmError;

/// Exact treatment of the valuation-origin exercise opportunity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum OriginExercise {
    /// The first possible exercise decision is strictly after valuation.
    Excluded,
    /// Valuation is an explicit common exercise decision before the first transition.
    Included,
}

/// Interpretation of the volatility column for an escrowed cash-dividend approximation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum EscrowedVolatilityConvention {
    /// Values are already volatilities of the risky prepaid-forward component.
    PrepaidForward,
    /// Values are origin spot volatilities, rescaled to match initial dollar volatility.
    SpotMatchedAtOrigin,
}

/// Uniform Bermudan base mesh augmented by every exact deterministic cash-event time.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UniformExerciseMesh {
    base_intervals: u32,
    origin_exercise: OriginExercise,
}

impl UniformExerciseMesh {
    /// Validates an exact base interval count and origin convention.
    ///
    /// # Errors
    ///
    /// Returns [`BatchLsmError::EmptyBaseIntervals`] when `base_intervals == 0`.
    pub fn try_new(
        base_intervals: u32,
        origin_exercise: OriginExercise,
    ) -> Result<Self, BatchLsmError> {
        if base_intervals == 0 {
            return Err(BatchLsmError::EmptyBaseIntervals);
        }
        Ok(Self {
            base_intervals,
            origin_exercise,
        })
    }

    /// Returns the exact number of uniform intervals before cash-event insertion.
    #[must_use]
    pub const fn base_intervals(self) -> u32 {
        self.base_intervals
    }

    /// Returns the declared valuation-origin exercise convention.
    #[must_use]
    pub const fn origin_exercise(self) -> OriginExercise {
        self.origin_exercise
    }
}

/// Row-addressed independent random namespaces for fitting and evaluation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BatchRandomStreams {
    training_seed: u64,
    evaluation_seed: u64,
    training_domain_start: u32,
    evaluation_domain_start: u32,
}

impl BatchRandomStreams {
    /// Declares two contiguous, row-addressed Philox domain intervals.
    #[must_use]
    pub const fn new(
        training_seed: u64,
        evaluation_seed: u64,
        training_domain_start: u32,
        evaluation_domain_start: u32,
    ) -> Self {
        Self {
            training_seed,
            evaluation_seed,
            training_domain_start,
            evaluation_domain_start,
        }
    }

    /// Returns the root fitting seed.
    #[must_use]
    pub const fn training_seed(self) -> u64 {
        self.training_seed
    }

    /// Returns the root independent-evaluation seed.
    #[must_use]
    pub const fn evaluation_seed(self) -> u64 {
        self.evaluation_seed
    }

    /// Returns the first fitting domain.
    #[must_use]
    pub const fn training_domain_start(self) -> u32 {
        self.training_domain_start
    }

    /// Returns the first independent-evaluation domain.
    #[must_use]
    pub const fn evaluation_domain_start(self) -> u32 {
        self.evaluation_domain_start
    }

    pub(super) fn validate(self, rows: usize) -> Result<(), BatchLsmError> {
        let training_end = domain_end("training", self.training_domain_start, rows)?;
        let evaluation_end = domain_end("evaluation", self.evaluation_domain_start, rows)?;
        let training_start = u64::from(self.training_domain_start);
        let evaluation_start = u64::from(self.evaluation_domain_start);
        if training_start < evaluation_end && evaluation_start < training_end {
            return Err(BatchLsmError::RandomDomainRangeCollision {
                training_start,
                training_end,
                evaluation_start,
                evaluation_end,
            });
        }
        Ok(())
    }

    pub(super) fn training_domain(self, row: usize) -> Result<u32, BatchLsmError> {
        domain_at("training", self.training_domain_start, row)
    }

    pub(super) fn evaluation_domain(self, row: usize) -> Result<u32, BatchLsmError> {
        domain_at("evaluation", self.evaluation_domain_start, row)
    }
}

fn domain_at(population: &'static str, start: u32, row: usize) -> Result<u32, BatchLsmError> {
    let row_u64 = u64::try_from(row).map_err(|_| BatchLsmError::RandomDomainRangeOverflow {
        population,
        start,
        rows: row.saturating_add(1),
    })?;
    let domain = u64::from(start) + row_u64;
    u32::try_from(domain).map_err(|_| BatchLsmError::RandomDomainRangeOverflow {
        population,
        start,
        rows: row.saturating_add(1),
    })
}

fn domain_end(population: &'static str, start: u32, rows: usize) -> Result<u64, BatchLsmError> {
    let rows_u64 = u64::try_from(rows).map_err(|_| BatchLsmError::RandomDomainRangeOverflow {
        population,
        start,
        rows,
    })?;
    let end = u64::from(start) + rows_u64;
    if end > u64::from(u32::MAX) + 1 {
        return Err(BatchLsmError::RandomDomainRangeOverflow {
            population,
            start,
            rows,
        });
    }
    Ok(end)
}

/// Complete numerical and random identity for one independently evaluated LSM call batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchLsmCallMethod {
    mesh: UniformExerciseMesh,
    training_paths: usize,
    evaluation_paths: usize,
    regression: PolynomialRegressionSpec,
    tie_rule: ExerciseTieRule,
    random_streams: BatchRandomStreams,
}

impl BatchLsmCallMethod {
    /// Validates path domains while preserving every estimator-defining input.
    ///
    /// # Errors
    ///
    /// Returns a typed path-count failure. No path count, mesh, regression, tie convention, or
    /// random namespace is changed.
    pub fn try_new(
        mesh: UniformExerciseMesh,
        training_paths: usize,
        evaluation_paths: usize,
        regression: PolynomialRegressionSpec,
        tie_rule: ExerciseTieRule,
        random_streams: BatchRandomStreams,
    ) -> Result<Self, BatchLsmError> {
        validate_path_count("training", training_paths)?;
        validate_path_count("evaluation", evaluation_paths)?;
        if evaluation_paths < 2 {
            return Err(BatchLsmError::InsufficientEvaluationPaths {
                observed: evaluation_paths,
            });
        }
        Ok(Self {
            mesh,
            training_paths,
            evaluation_paths,
            regression,
            tie_rule,
            random_streams,
        })
    }

    /// Returns the exact base exercise mesh.
    #[must_use]
    pub const fn mesh(self) -> UniformExerciseMesh {
        self.mesh
    }

    /// Returns the fitting population size per contract.
    #[must_use]
    pub const fn training_paths(self) -> usize {
        self.training_paths
    }

    /// Returns the independent evaluation population size per contract.
    #[must_use]
    pub const fn evaluation_paths(self) -> usize {
        self.evaluation_paths
    }

    /// Returns the exact polynomial regression specification.
    #[must_use]
    pub const fn regression(self) -> PolynomialRegressionSpec {
        self.regression
    }

    /// Returns the exact exercise tie convention.
    #[must_use]
    pub const fn tie_rule(self) -> ExerciseTieRule {
        self.tie_rule
    }

    /// Returns the row-addressed fitting/evaluation random identity.
    #[must_use]
    pub const fn random_streams(self) -> BatchRandomStreams {
        self.random_streams
    }
}

fn validate_path_count(population: &'static str, observed: usize) -> Result<(), BatchLsmError> {
    if observed == 0 || u32::try_from(observed).is_err() {
        return Err(BatchLsmError::PathCountOutsideDomain {
            population,
            observed,
            maximum: u32::MAX,
        });
    }
    Ok(())
}

/// Exactly one supported deterministic dividend representation for a complete batch.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum BatchDividendModel<'a> {
    /// One nonnegative constant annualized decimal proportional-dividend yield per contract.
    ContinuousYield {
        /// Row-aligned constant yields `q` used as `Q(t)=q*t`.
        annualized_decimals: &'a [f64],
    },
    /// Exact deterministic cash-event schedules under one declared limited-liability policy.
    CashDividends {
        /// Row-aligned future cash schedules.
        schedules: RaggedCashDividends<'a>,
        /// Exact cash-jump boundary convention.
        policy: DividendPolicy,
    },
    /// Escrowed approximation: risky prepaid forward plus deterministic prepaid cash.
    EscrowedCashDividends {
        /// Row-aligned future cash schedules.
        schedules: RaggedCashDividends<'a>,
        /// Explicit meaning and any transformation of the volatility input column.
        volatility_convention: EscrowedVolatilityConvention,
    },
}

impl BatchDividendModel<'_> {
    pub(super) fn rows(self) -> usize {
        match self {
            Self::ContinuousYield {
                annualized_decimals,
            } => annualized_decimals.len(),
            Self::CashDividends { schedules, .. }
            | Self::EscrowedCashDividends { schedules, .. } => schedules.rows(),
        }
    }
}

/// Validated zero-copy scalar columns for constant-rate, constant-volatility call contracts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchLsmCallInputs<'a> {
    pub(super) initial_spots: &'a [f64],
    pub(super) strikes: &'a [f64],
    pub(super) maturities: &'a [f64],
    pub(super) constant_continuously_compounded_rates: &'a [f64],
    pub(super) constant_volatilities: &'a [f64],
    pub(super) dividend_model: BatchDividendModel<'a>,
}

impl<'a> BatchLsmCallInputs<'a> {
    /// Validates aligned scalar columns and dividend-model rows without copying any input.
    ///
    /// Rates are constant continuously compounded annual decimals over each contract horizon.
    /// Volatilities are constant positive annual decimals. Continuous yields and exact cash jumps
    /// are different variants and cannot be mixed within one request.
    ///
    /// # Errors
    ///
    /// Returns the first row/field/event violation. Cash event times must be unique and strictly
    /// before maturity; events are never rounded, dropped, shifted, or aggregated.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        initial_spots: &'a [f64],
        strikes: &'a [f64],
        maturities: &'a [f64],
        constant_continuously_compounded_rates: &'a [f64],
        constant_volatilities: &'a [f64],
        dividend_model: BatchDividendModel<'a>,
    ) -> Result<Self, BatchLsmError> {
        let rows = initial_spots.len();
        if rows == 0 {
            return Err(BatchLsmError::EmptyBatch);
        }
        validate_length("strikes", rows, strikes.len())?;
        validate_length("maturities", rows, maturities.len())?;
        validate_length(
            "constant_continuously_compounded_rates",
            rows,
            constant_continuously_compounded_rates.len(),
        )?;
        validate_length("constant_volatilities", rows, constant_volatilities.len())?;
        validate_length("dividend_model", rows, dividend_model.rows())?;

        for row in 0..rows {
            validate_nonnegative("initial_spots", row, initial_spots[row])?;
            validate_positive("strikes", row, strikes[row])?;
            validate_positive("maturities", row, maturities[row])?;
            validate_finite(
                "constant_continuously_compounded_rates",
                row,
                constant_continuously_compounded_rates[row],
            )?;
            validate_positive("constant_volatilities", row, constant_volatilities[row])?;
            match dividend_model {
                BatchDividendModel::ContinuousYield {
                    annualized_decimals,
                } => validate_nonnegative(
                    "continuous_dividend_yields",
                    row,
                    annualized_decimals[row],
                )?,
                BatchDividendModel::CashDividends { schedules, .. }
                | BatchDividendModel::EscrowedCashDividends { schedules, .. } => {
                    let schedule = schedules.row(row).ok_or(BatchLsmError::RowCountMismatch {
                        field: "dividend_model",
                        expected: rows,
                        observed: schedules.rows(),
                    })?;
                    for (event, &time) in schedule.times().iter().enumerate() {
                        if time >= maturities[row] {
                            return Err(BatchLsmError::CashDividendOutsideContract {
                                row,
                                event,
                                time,
                                maturity: maturities[row],
                            });
                        }
                        if event > 0 && schedule.times()[event - 1].to_bits() == time.to_bits() {
                            return Err(BatchLsmError::DuplicateCashDividendTime {
                                row,
                                event,
                                time,
                            });
                        }
                    }
                    if let BatchDividendModel::EscrowedCashDividends {
                        volatility_convention,
                        ..
                    } = dividend_model
                    {
                        validate_escrowed_row(
                            row,
                            initial_spots[row],
                            constant_continuously_compounded_rates[row],
                            constant_volatilities[row],
                            schedule,
                            volatility_convention,
                        )?;
                    }
                }
            }
        }
        Ok(Self {
            initial_spots,
            strikes,
            maturities,
            constant_continuously_compounded_rates,
            constant_volatilities,
            dividend_model,
        })
    }

    /// Returns the exact contract row count.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.initial_spots.len()
    }
}

/// Compact strike groups whose contracts share one exact simulated path population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupedBatchLsmCallInputs<'a> {
    pub(super) group_offsets: &'a [u32],
    pub(super) initial_spots: &'a [f64],
    pub(super) strikes: &'a [f64],
    pub(super) maturities: &'a [f64],
    pub(super) constant_continuously_compounded_rates: &'a [f64],
    pub(super) constant_volatilities: &'a [f64],
    pub(super) dividend_model: BatchDividendModel<'a>,
}

impl<'a> GroupedBatchLsmCallInputs<'a> {
    /// Validates group-level path inputs, row-level strikes, and exact contiguous boundaries.
    ///
    /// Each group owns one spot, maturity, rate, volatility, and dividend representation. Every
    /// strike in `group_offsets[group]..group_offsets[group + 1]` is fitted and independently
    /// evaluated against the same group-addressed training and evaluation path populations. Empty
    /// groups and inferred or repaired boundaries are rejected.
    ///
    /// # Errors
    ///
    /// Returns the first scalar, schedule, or boundary violation. Cash events must be unique and
    /// strictly before their group's maturity. No contract is moved between groups.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        group_offsets: &'a [u32],
        initial_spots: &'a [f64],
        strikes: &'a [f64],
        maturities: &'a [f64],
        constant_continuously_compounded_rates: &'a [f64],
        constant_volatilities: &'a [f64],
        dividend_model: BatchDividendModel<'a>,
    ) -> Result<Self, BatchLsmError> {
        let groups = initial_spots.len();
        if groups == 0 || strikes.is_empty() {
            return Err(BatchLsmError::EmptyBatch);
        }
        validate_length("maturities", groups, maturities.len())?;
        validate_length(
            "constant_continuously_compounded_rates",
            groups,
            constant_continuously_compounded_rates.len(),
        )?;
        validate_length("constant_volatilities", groups, constant_volatilities.len())?;
        validate_length("dividend_model", groups, dividend_model.rows())?;
        validate_group_offsets(group_offsets, groups, strikes.len())?;

        for (row, &strike) in strikes.iter().enumerate() {
            validate_positive("strikes", row, strike)?;
        }
        for group in 0..groups {
            validate_nonnegative("initial_spots", group, initial_spots[group])?;
            validate_positive("maturities", group, maturities[group])?;
            validate_finite(
                "constant_continuously_compounded_rates",
                group,
                constant_continuously_compounded_rates[group],
            )?;
            validate_positive("constant_volatilities", group, constant_volatilities[group])?;
            match dividend_model {
                BatchDividendModel::ContinuousYield {
                    annualized_decimals,
                } => validate_nonnegative(
                    "continuous_dividend_yields",
                    group,
                    annualized_decimals[group],
                )?,
                BatchDividendModel::CashDividends { schedules, .. }
                | BatchDividendModel::EscrowedCashDividends { schedules, .. } => {
                    let schedule = schedules
                        .row(group)
                        .ok_or(BatchLsmError::RowCountMismatch {
                            field: "dividend_model",
                            expected: groups,
                            observed: schedules.rows(),
                        })?;
                    for (event, &time) in schedule.times().iter().enumerate() {
                        if time >= maturities[group] {
                            return Err(BatchLsmError::CashDividendOutsideContract {
                                row: group,
                                event,
                                time,
                                maturity: maturities[group],
                            });
                        }
                        if event > 0 && schedule.times()[event - 1].to_bits() == time.to_bits() {
                            return Err(BatchLsmError::DuplicateCashDividendTime {
                                row: group,
                                event,
                                time,
                            });
                        }
                    }
                    if let BatchDividendModel::EscrowedCashDividends {
                        volatility_convention,
                        ..
                    } = dividend_model
                    {
                        validate_escrowed_row(
                            group,
                            initial_spots[group],
                            constant_continuously_compounded_rates[group],
                            constant_volatilities[group],
                            schedule,
                            volatility_convention,
                        )?;
                    }
                }
            }
        }
        Ok(Self {
            group_offsets,
            initial_spots,
            strikes,
            maturities,
            constant_continuously_compounded_rates,
            constant_volatilities,
            dividend_model,
        })
    }

    /// Returns the number of independent simulated path populations.
    #[must_use]
    pub const fn groups(self) -> usize {
        self.initial_spots.len()
    }

    /// Returns the number of strike-specific contracts.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.strikes.len()
    }

    pub(super) fn group_range(self, group: usize) -> std::ops::Range<usize> {
        let start = usize::try_from(self.group_offsets[group]).expect("validated group offset");
        let end = usize::try_from(self.group_offsets[group + 1]).expect("validated group offset");
        start..end
    }
}

fn validate_group_offsets(
    offsets: &[u32],
    groups: usize,
    rows: usize,
) -> Result<(), BatchLsmError> {
    // A non-ZST `f64` slice cannot have `usize::MAX` elements, so this addition cannot overflow.
    let expected = groups + 1;
    if offsets.len() != expected {
        return Err(BatchLsmError::GroupOffsetCountMismatch {
            expected,
            observed: offsets.len(),
        });
    }
    if offsets[0] != 0 {
        return Err(BatchLsmError::GroupOffsetsDoNotStartAtZero {
            observed: offsets[0],
        });
    }
    for (group, window) in offsets.windows(2).enumerate() {
        if window[0] >= window[1] {
            return Err(BatchLsmError::InvalidGroupRange {
                group,
                start: window[0],
                end: window[1],
            });
        }
    }
    let terminal = offsets[offsets.len() - 1];
    if usize::try_from(terminal).ok() != Some(rows) {
        return Err(BatchLsmError::GroupOffsetEndMismatch {
            expected: rows,
            observed: terminal,
        });
    }
    Ok(())
}

fn validate_escrowed_row(
    row: usize,
    initial_spot: f64,
    rate: f64,
    supplied_volatility: f64,
    schedule: finoptions_batch::CashDividendRow<'_>,
    volatility_convention: EscrowedVolatilityConvention,
) -> Result<(), BatchLsmError> {
    let prepaid_cash = escrowed_origin_prepaid_cash(row, rate, schedule)?;
    let prepaid_forward = initial_spot - prepaid_cash;
    if !prepaid_forward.is_finite() || prepaid_forward <= 0.0 {
        return Err(BatchLsmError::NonPositiveEscrowedPrepaidForward {
            row,
            initial_spot,
            prepaid_cash,
        });
    }
    if volatility_convention == EscrowedVolatilityConvention::SpotMatchedAtOrigin {
        let transformed = supplied_volatility * initial_spot / prepaid_forward;
        if !transformed.is_finite() || transformed <= 0.0 {
            return Err(BatchLsmError::InvalidEscrowedPrepaidForwardVolatility {
                row,
                supplied_volatility,
                initial_spot,
                prepaid_forward,
                transformed,
            });
        }
    }
    Ok(())
}

pub(super) fn escrowed_origin_prepaid_cash(
    row: usize,
    rate: f64,
    schedule: finoptions_batch::CashDividendRow<'_>,
) -> Result<f64, BatchLsmError> {
    let mut prepaid_cash = 0.0;
    for (event, (&time, &amount)) in schedule.times().iter().zip(schedule.amounts()).enumerate() {
        let discounted = amount * (-rate * time).exp();
        prepaid_cash += discounted;
        if !discounted.is_finite() || !prepaid_cash.is_finite() {
            return Err(BatchLsmError::NonFiniteEscrowedPrepaidCash {
                row,
                event,
                time,
                amount,
                rate,
            });
        }
    }
    Ok(prepaid_cash)
}

fn validate_length(
    field: &'static str,
    expected: usize,
    observed: usize,
) -> Result<(), BatchLsmError> {
    if observed != expected {
        return Err(BatchLsmError::RowCountMismatch {
            field,
            expected,
            observed,
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, row: usize, observed: f64) -> Result<(), BatchLsmError> {
    if !observed.is_finite() {
        return Err(BatchLsmError::InvalidRowValue {
            field,
            row,
            observed,
        });
    }
    Ok(())
}

fn validate_positive(field: &'static str, row: usize, observed: f64) -> Result<(), BatchLsmError> {
    if !observed.is_finite() || observed <= 0.0 {
        return Err(BatchLsmError::InvalidRowValue {
            field,
            row,
            observed,
        });
    }
    Ok(())
}

fn validate_nonnegative(
    field: &'static str,
    row: usize,
    observed: f64,
) -> Result<(), BatchLsmError> {
    if !observed.is_finite() || observed < 0.0 || (observed == 0.0 && observed.is_sign_negative()) {
        return Err(BatchLsmError::InvalidRowValue {
            field,
            row,
            observed,
        });
    }
    Ok(())
}
