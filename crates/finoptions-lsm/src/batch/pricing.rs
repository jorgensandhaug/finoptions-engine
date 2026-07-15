//! Contract-parallel construction, simulation, policy fitting, and independent evaluation.

use std::sync::atomic::{AtomicUsize, Ordering};

use finoptions_batch::CashDividendRow;
use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve, TimeGrid,
};
use finoptions_sim::{
    CpuExecutor, CpuMathBackend, EscrowedGbmPathPlan, GbmPathPlan, RandomStream, SimulatedPaths,
};
use rayon::prelude::*;

use crate::{
    ExerciseFitDiagnostics, FittedLsmCallPolicy, LsmCallEstimator, LsmError, LsmEvaluation, LsmRun,
    RegressionDiagnostics,
};

use super::input::escrowed_origin_prepaid_cash;
use super::{
    BatchDividendModel, BatchLsmCallInputs, BatchLsmCallMethod, BatchLsmError,
    EscrowedVolatilityConvention, GroupedBatchLsmCallInputs, OriginExercise,
};

#[derive(Clone, Copy)]
struct RowDividendModel<'a> {
    cash_schedule: Option<CashDividendRow<'a>>,
    continuous_yield: f64,
    cash_policy: DividendPolicy,
    escrowed_volatility: Option<EscrowedVolatilityConvention>,
}

enum RowPathPlan {
    Spot(GbmPathPlan),
    Escrowed(EscrowedGbmPathPlan),
}

#[derive(Clone, Copy)]
enum GroupPopulation {
    Training,
    Evaluation,
}

/// Compact complete per-contract outputs for one independent fixed-policy evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BatchLsmCallRowResult {
    price: f64,
    sample_standard_deviation: f64,
    standard_error: f64,
    origin_decision_fitted: bool,
    origin_exercised: bool,
    regression_nodes: u32,
    no_training_support_nodes: u32,
    minimum_in_the_money_training_paths: u32,
    maximum_regression_condition_number: f64,
    early_exercise_paths: u32,
    expiry_paths: u32,
}

impl BatchLsmCallRowResult {
    /// Returns the independent mean discounted payoff.
    #[must_use]
    pub const fn price(self) -> f64 {
        self.price
    }

    /// Returns the independent payoff sample standard deviation.
    #[must_use]
    pub const fn sample_standard_deviation(self) -> f64 {
        self.sample_standard_deviation
    }

    /// Returns the standard error of the independent mean.
    #[must_use]
    pub const fn standard_error(self) -> f64 {
        self.standard_error
    }

    /// Returns whether valuation origin was a fitted common decision.
    #[must_use]
    pub const fn origin_decision_fitted(self) -> bool {
        self.origin_decision_fitted
    }

    /// Returns whether the fitted common origin decision was immediate exercise.
    #[must_use]
    pub const fn origin_exercised(self) -> bool {
        self.origin_exercised
    }

    /// Returns the number of accepted positive-time continuation regressions.
    #[must_use]
    pub const fn regression_nodes(self) -> u32 {
        self.regression_nodes
    }

    /// Returns the number of exercise nodes with no in-the-money training state.
    #[must_use]
    pub const fn no_training_support_nodes(self) -> u32 {
        self.no_training_support_nodes
    }

    /// Returns the minimum ITM fitting population, or zero when no regression was fitted.
    #[must_use]
    pub const fn minimum_in_the_money_training_paths(self) -> u32 {
        self.minimum_in_the_money_training_paths
    }

    /// Returns the largest accepted regression condition number, or zero with no regression.
    #[must_use]
    pub const fn maximum_regression_condition_number(self) -> f64 {
        self.maximum_regression_condition_number
    }

    /// Returns independent paths stopped before terminal expiry.
    #[must_use]
    pub const fn early_exercise_paths(self) -> u32 {
        self.early_exercise_paths
    }

    /// Returns independent paths reaching terminal expiry.
    #[must_use]
    pub const fn expiry_paths(self) -> u32 {
        self.expiry_paths
    }
}

/// Row-ordered outputs from one exact batch request.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchLsmCallResult {
    rows: Vec<BatchLsmCallRowResult>,
}

impl BatchLsmCallResult {
    /// Returns the contract count.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns whether the result contains no contracts.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Borrows compact row-ordered outputs.
    #[must_use]
    pub fn as_slice(&self) -> &[BatchLsmCallRowResult] {
        &self.rows
    }

    /// Transfers all compact row-ordered outputs without copying.
    #[must_use]
    pub fn into_rows(self) -> Vec<BatchLsmCallRowResult> {
        self.rows
    }
}

impl<'a> BatchLsmCallInputs<'a> {
    /// Prices every contract using outer contract parallelism and serial path-local kernels.
    ///
    /// Each row receives unique training and evaluation domains. A cash event is inserted at its
    /// exact year fraction and observed for exercise before its declared jump. Results are invariant
    /// to worker count on one numerical build. Parallel failures are replayed at the deterministic
    /// lowest row to return the complete typed error.
    ///
    /// # Errors
    ///
    /// Returns before allocation if random-domain intervals are invalid, or returns a row-addressed
    /// financial, simulation, regression, support, or arithmetic error. No row, event, estimator,
    /// path count, thread count, or dividend representation is changed.
    pub fn price(
        self,
        method: BatchLsmCallMethod,
        executor: &CpuExecutor,
    ) -> Result<BatchLsmCallResult, BatchLsmError> {
        method.random_streams().validate(self.rows())?;
        let math_backend = executor.math_backend();
        executor.dispatch(
            || self.price_serial(method, math_backend),
            || self.price_parallel(method, math_backend),
        )
    }

    fn price_serial(
        self,
        method: BatchLsmCallMethod,
        math_backend: CpuMathBackend,
    ) -> Result<BatchLsmCallResult, BatchLsmError> {
        let mut results = Vec::new();
        results
            .try_reserve_exact(self.rows())
            .map_err(|_| BatchLsmError::ResultAllocationFailed { rows: self.rows() })?;
        for row in 0..self.rows() {
            results.push(self.price_row(method, math_backend, row)?);
        }
        Ok(BatchLsmCallResult { rows: results })
    }

    fn price_parallel(
        self,
        method: BatchLsmCallMethod,
        math_backend: CpuMathBackend,
    ) -> Result<BatchLsmCallResult, BatchLsmError> {
        let mut results = Vec::new();
        results
            .try_reserve_exact(self.rows())
            .map_err(|_| BatchLsmError::ResultAllocationFailed { rows: self.rows() })?;
        results.resize(self.rows(), BatchLsmCallRowResult::default());
        let first_failure = AtomicUsize::new(usize::MAX);
        results
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, output)| {
                if row > first_failure.load(Ordering::Relaxed) {
                    return;
                }
                match self.price_row(method, math_backend, row) {
                    Ok(result) => *output = result,
                    Err(_) => {
                        first_failure.fetch_min(row, Ordering::Relaxed);
                    }
                }
            });

        let failed_row = first_failure.load(Ordering::Relaxed);
        if failed_row == usize::MAX {
            return Ok(BatchLsmCallResult { rows: results });
        }
        match self.price_row(method, math_backend, failed_row) {
            Err(error) => Err(error),
            Ok(_) => Err(BatchLsmError::ParallelFailureReplayMismatch { row: failed_row }),
        }
    }

    fn price_row(
        self,
        method: BatchLsmCallMethod,
        math_backend: CpuMathBackend,
        row: usize,
    ) -> Result<BatchLsmCallRowResult, BatchLsmError> {
        let (plan, curve) = self.build_plan(method, row)?;
        let estimator =
            LsmCallEstimator::try_new(self.strikes[row], method.regression(), method.tie_rule())
                .map_err(|source| BatchLsmError::EstimatorAtRow { row, source })?;
        let streams = method.random_streams();
        let training_stream =
            RandomStream::new(streams.training_seed(), streams.training_domain(row)?);
        let evaluation_stream =
            RandomStream::new(streams.evaluation_seed(), streams.evaluation_domain(row)?);
        let run = match &plan {
            RowPathPlan::Spot(plan) => estimator.price_seeded_serial_with_math_backend(
                math_backend,
                plan,
                &curve,
                self.initial_spots[row],
                method.training_paths(),
                method.evaluation_paths(),
                training_stream,
                evaluation_stream,
            ),
            RowPathPlan::Escrowed(plan) => estimator
                .price_escrowed_seeded_serial_with_math_backend(
                    math_backend,
                    plan,
                    &curve,
                    self.initial_spots[row],
                    method.training_paths(),
                    method.evaluation_paths(),
                    training_stream,
                    evaluation_stream,
                ),
        }
        .map_err(|source| BatchLsmError::EstimatorAtRow { row, source })?;
        BatchLsmCallRowResult::try_from_run(row, run)
    }

    fn build_plan(
        self,
        method: BatchLsmCallMethod,
        row: usize,
    ) -> Result<(RowPathPlan, LogDiscountCurve), BatchLsmError> {
        let dividend = self.row_dividend_model(row)?;
        build_path_plan(
            row,
            method,
            self.initial_spots[row],
            self.maturities[row],
            self.constant_continuously_compounded_rates[row],
            self.constant_volatilities[row],
            dividend,
        )
    }

    fn row_dividend_model(self, row: usize) -> Result<RowDividendModel<'a>, BatchLsmError> {
        match self.dividend_model {
            BatchDividendModel::ContinuousYield {
                annualized_decimals,
            } => Ok(RowDividendModel {
                cash_schedule: None,
                continuous_yield: annualized_decimals[row],
                cash_policy: DividendPolicy::Liquidator,
                escrowed_volatility: None,
            }),
            BatchDividendModel::CashDividends { schedules, policy } => Ok(RowDividendModel {
                cash_schedule: Some(self.cash_row(schedules, row)?),
                continuous_yield: 0.0,
                cash_policy: policy,
                escrowed_volatility: None,
            }),
            BatchDividendModel::EscrowedCashDividends {
                schedules,
                volatility_convention,
            } => Ok(RowDividendModel {
                cash_schedule: Some(self.cash_row(schedules, row)?),
                continuous_yield: 0.0,
                cash_policy: DividendPolicy::Liquidator,
                escrowed_volatility: Some(volatility_convention),
            }),
        }
    }

    fn cash_row(
        self,
        schedules: finoptions_batch::RaggedCashDividends<'a>,
        row: usize,
    ) -> Result<CashDividendRow<'a>, BatchLsmError> {
        schedules.row(row).ok_or(BatchLsmError::RowCountMismatch {
            field: "dividend_model",
            expected: self.rows(),
            observed: schedules.rows(),
        })
    }
}

impl<'a> GroupedBatchLsmCallInputs<'a> {
    /// Prices strike groups with one group-addressed path pair per simulation state.
    ///
    /// Every strike retains a separately fitted stopping policy and independent out-of-sample
    /// valuation. Contracts within a group intentionally share common random numbers because their
    /// complete simulated market state is represented once at the type boundary. Parallel results
    /// are invariant to worker count on one numerical build.
    ///
    /// # Errors
    ///
    /// Returns before allocation if group-addressed random domains are invalid, or returns the
    /// first contract-addressed financial, simulation, regression, support, or arithmetic error.
    /// No group, strike, path count, event, or estimator setting is changed.
    pub fn price(
        self,
        method: BatchLsmCallMethod,
        executor: &CpuExecutor,
    ) -> Result<BatchLsmCallResult, BatchLsmError> {
        method.random_streams().validate(self.groups())?;
        let math_backend = executor.math_backend();
        executor.dispatch(
            || self.price_serial(method, math_backend),
            || self.price_parallel(method, math_backend),
        )
    }

    fn price_serial(
        self,
        method: BatchLsmCallMethod,
        math_backend: CpuMathBackend,
    ) -> Result<BatchLsmCallResult, BatchLsmError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.rows())
            .map_err(|_| BatchLsmError::ResultAllocationFailed { rows: self.rows() })?;
        rows.resize(self.rows(), BatchLsmCallRowResult::default());
        for group in 0..self.groups() {
            let range = self.group_range(group);
            self.price_group_into(method, math_backend, group, &mut rows[range])?;
        }
        Ok(BatchLsmCallResult { rows })
    }

    fn price_parallel(
        self,
        method: BatchLsmCallMethod,
        math_backend: CpuMathBackend,
    ) -> Result<BatchLsmCallResult, BatchLsmError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.rows())
            .map_err(|_| BatchLsmError::ResultAllocationFailed { rows: self.rows() })?;
        rows.resize(self.rows(), BatchLsmCallRowResult::default());
        let mut group_outputs = Vec::new();
        group_outputs
            .try_reserve_exact(self.groups())
            .map_err(|_| BatchLsmError::ParallelGroupPartitionAllocationFailed {
                groups: self.groups(),
            })?;
        let mut remaining = rows.as_mut_slice();
        for group in 0..self.groups() {
            let group_rows = self.group_range(group).len();
            let (output, tail) = remaining.split_at_mut(group_rows);
            group_outputs.push(output);
            remaining = tail;
        }
        debug_assert!(remaining.is_empty());

        let first_failure = AtomicUsize::new(usize::MAX);
        group_outputs
            .par_iter_mut()
            .enumerate()
            .for_each(|(group, output)| {
                if group > first_failure.load(Ordering::Relaxed) {
                    return;
                }
                if self
                    .price_group_into(method, math_backend, group, output)
                    .is_err()
                {
                    first_failure.fetch_min(group, Ordering::Relaxed);
                }
            });
        drop(group_outputs);

        let failed_group = first_failure.load(Ordering::Relaxed);
        if failed_group == usize::MAX {
            return Ok(BatchLsmCallResult { rows });
        }
        let failed_range = self.group_range(failed_group);
        let failed_row = failed_range.start;
        match self.price_group_into(method, math_backend, failed_group, &mut rows[failed_range]) {
            Err(error) => Err(error),
            Ok(()) => Err(BatchLsmError::ParallelFailureReplayMismatch { row: failed_row }),
        }
    }

    fn price_group_into(
        self,
        method: BatchLsmCallMethod,
        math_backend: CpuMathBackend,
        group: usize,
        output: &mut [BatchLsmCallRowResult],
    ) -> Result<(), BatchLsmError> {
        let range = self.group_range(group);
        debug_assert_eq!(range.len(), output.len());
        let first_row = range.start;
        let dividend = self.group_dividend_model(group)?;
        let (plan, curve) = build_path_plan(
            first_row,
            method,
            self.initial_spots[group],
            self.maturities[group],
            self.constant_continuously_compounded_rates[group],
            self.constant_volatilities[group],
            dividend,
        )?;
        let streams = method.random_streams();
        let training_stream =
            RandomStream::new(streams.training_seed(), streams.training_domain(group)?);
        let evaluation_stream =
            RandomStream::new(streams.evaluation_seed(), streams.evaluation_domain(group)?);
        let training = simulate_group_population(
            &plan,
            math_backend,
            self.initial_spots[group],
            method.training_paths(),
            training_stream,
            GroupPopulation::Training,
        )
        .map_err(|source| BatchLsmError::EstimatorAtRow {
            row: first_row,
            source,
        })?;
        let mut policies = Vec::new();
        policies
            .try_reserve_exact(range.len())
            .map_err(|_| BatchLsmError::ResultAllocationFailed { rows: range.len() })?;
        for local_row in 0..range.len() {
            let row = first_row + local_row;
            let estimator = LsmCallEstimator::try_new(
                self.strikes[row],
                method.regression(),
                method.tie_rule(),
            )
            .map_err(|source| BatchLsmError::EstimatorAtRow { row, source })?;
            let policy = estimator
                .fit_with_math_backend(&training, &curve, self.initial_spots[group], math_backend)
                .map_err(|source| BatchLsmError::EstimatorAtRow { row, source })?;
            policies.push(policy);
        }
        drop(training);
        let evaluation = simulate_group_population(
            &plan,
            math_backend,
            self.initial_spots[group],
            method.evaluation_paths(),
            evaluation_stream,
            GroupPopulation::Evaluation,
        )
        .map_err(|source| BatchLsmError::EstimatorAtRow {
            row: first_row,
            source,
        })?;
        for (local_row, (policy, output)) in policies.into_iter().zip(output).enumerate() {
            let row = first_row + local_row;
            let valuation = policy
                .evaluate(&evaluation)
                .map_err(|source| BatchLsmError::EstimatorAtRow { row, source })?;
            *output = BatchLsmCallRowResult::try_from_parts(row, &policy, &valuation)?;
        }
        Ok(())
    }

    fn group_dividend_model(self, group: usize) -> Result<RowDividendModel<'a>, BatchLsmError> {
        match self.dividend_model {
            BatchDividendModel::ContinuousYield {
                annualized_decimals,
            } => Ok(RowDividendModel {
                cash_schedule: None,
                continuous_yield: annualized_decimals[group],
                cash_policy: DividendPolicy::Liquidator,
                escrowed_volatility: None,
            }),
            BatchDividendModel::CashDividends { schedules, policy } => Ok(RowDividendModel {
                cash_schedule: Some(self.group_cash_row(schedules, group)?),
                continuous_yield: 0.0,
                cash_policy: policy,
                escrowed_volatility: None,
            }),
            BatchDividendModel::EscrowedCashDividends {
                schedules,
                volatility_convention,
            } => Ok(RowDividendModel {
                cash_schedule: Some(self.group_cash_row(schedules, group)?),
                continuous_yield: 0.0,
                cash_policy: DividendPolicy::Liquidator,
                escrowed_volatility: Some(volatility_convention),
            }),
        }
    }

    fn group_cash_row(
        self,
        schedules: finoptions_batch::RaggedCashDividends<'a>,
        group: usize,
    ) -> Result<CashDividendRow<'a>, BatchLsmError> {
        schedules.row(group).ok_or(BatchLsmError::RowCountMismatch {
            field: "dividend_model",
            expected: self.groups(),
            observed: schedules.rows(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn simulate_group_population(
    plan: &RowPathPlan,
    math_backend: CpuMathBackend,
    initial_spot: f64,
    paths: usize,
    stream: RandomStream,
    population: GroupPopulation,
) -> Result<SimulatedPaths, LsmError> {
    let simulated = match plan {
        RowPathPlan::Spot(plan) => {
            plan.simulate_seeded_with_math_backend(math_backend, initial_spot, paths, stream)
        }
        RowPathPlan::Escrowed(plan) => {
            plan.simulate_seeded_with_math_backend(math_backend, initial_spot, paths, stream)
        }
    };
    simulated.map_err(|source| match population {
        GroupPopulation::Training => LsmError::TrainingSimulation { source },
        GroupPopulation::Evaluation => LsmError::EvaluationSimulation { source },
    })
}

#[allow(clippy::too_many_arguments)]
fn build_path_plan(
    row: usize,
    method: BatchLsmCallMethod,
    initial_spot: f64,
    maturity: f64,
    rate: f64,
    supplied_volatility: f64,
    dividend: RowDividendModel<'_>,
) -> Result<(RowPathPlan, LogDiscountCurve), BatchLsmError> {
    let cash_times = dividend
        .cash_schedule
        .map_or(&[][..], CashDividendRow::times);
    let times = exact_union_times(row, maturity, method.mesh().base_intervals(), cash_times)?;
    let grid = TimeGrid::try_new(times)
        .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?;
    let (curve, dividend_yield_curve) =
        build_row_curves(row, &grid, rate, dividend.continuous_yield)?;
    let schedule = build_row_schedule(
        row,
        grid,
        dividend.cash_schedule,
        method.mesh().origin_exercise(),
    )?;
    let volatility_value = path_volatility(row, initial_spot, rate, supplied_volatility, dividend)?;
    let volatility = ConstantVolatility::try_new(volatility_value)
        .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?;
    let plan = if dividend.escrowed_volatility.is_some() {
        EscrowedGbmPathPlan::try_new(&schedule, &curve, volatility).map(RowPathPlan::Escrowed)
    } else {
        GbmPathPlan::try_new(
            &schedule,
            &curve,
            &dividend_yield_curve,
            volatility,
            dividend.cash_policy,
        )
        .map(RowPathPlan::Spot)
    }
    .map_err(|source| BatchLsmError::SimulationPlanAtRow { row, source })?;
    Ok((plan, curve))
}

fn path_volatility(
    row: usize,
    initial_spot: f64,
    rate: f64,
    supplied_volatility: f64,
    dividend: RowDividendModel<'_>,
) -> Result<f64, BatchLsmError> {
    if dividend.escrowed_volatility != Some(EscrowedVolatilityConvention::SpotMatchedAtOrigin) {
        return Ok(supplied_volatility);
    }
    let schedule = dividend
        .cash_schedule
        .ok_or(BatchLsmError::RowCountMismatch {
            field: "escrowed_cash_dividends",
            expected: 1,
            observed: 0,
        })?;
    let prepaid_cash = escrowed_origin_prepaid_cash(row, rate, schedule)?;
    let prepaid_forward = initial_spot - prepaid_cash;
    Ok(supplied_volatility * initial_spot / prepaid_forward)
}

fn build_row_curves(
    row: usize,
    grid: &TimeGrid,
    rate: f64,
    continuous_yield: f64,
) -> Result<(LogDiscountCurve, CumulativeDividendYieldCurve), BatchLsmError> {
    let log_discounts = grid
        .as_slice()
        .iter()
        .enumerate()
        .map(|(node, &time)| if node == 0 { 0.0 } else { -rate * time })
        .collect::<Vec<_>>();
    let curve = LogDiscountCurve::try_new(grid, log_discounts)
        .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?;
    let cumulative_yields = grid
        .as_slice()
        .iter()
        .enumerate()
        .map(|(node, &time)| {
            if node == 0 {
                0.0
            } else {
                continuous_yield * time
            }
        })
        .collect::<Vec<_>>();
    let yield_curve = CumulativeDividendYieldCurve::try_new(grid, cumulative_yields)
        .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?;
    Ok((curve, yield_curve))
}

fn build_row_schedule(
    row: usize,
    grid: TimeGrid,
    cash_schedule: Option<CashDividendRow<'_>>,
    origin_exercise: OriginExercise,
) -> Result<EventSchedule, BatchLsmError> {
    let mut dividends = Vec::new();
    if let Some(cash) = cash_schedule {
        dividends.try_reserve_exact(cash.len()).map_err(|_| {
            BatchLsmError::ContractMeshAllocationFailed {
                row,
                nodes: cash.len(),
            }
        })?;
        for (event, (&time, &amount)) in cash.times().iter().zip(cash.amounts()).enumerate() {
            let index = grid
                .as_slice()
                .binary_search_by(|candidate| candidate.total_cmp(&time))
                .map_err(|_| BatchLsmError::CashDividendNodeMissing { row, event, time })?;
            let node = grid
                .node_id(index)
                .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?;
            dividends.push(
                CashDividend::try_new(node, amount)
                    .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?,
            );
        }
    }
    let exercise_start = usize::from(origin_exercise == OriginExercise::Excluded);
    let exercise_nodes = (exercise_start..grid.len() - 1)
        .map(|node| grid.node_id(node))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?;
    let expiry = grid
        .node_id(grid.len() - 1)
        .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })?;
    EventSchedule::try_new(grid, dividends, exercise_nodes, expiry)
        .map_err(|source| BatchLsmError::FinancialInputAtRow { row, source })
}

fn exact_union_times(
    row: usize,
    maturity: f64,
    base_intervals: u32,
    cash_times: &[f64],
) -> Result<Vec<f64>, BatchLsmError> {
    let base_intervals_usize =
        usize::try_from(base_intervals).map_err(|_| BatchLsmError::ContractMeshShapeOverflow {
            row,
            base_intervals,
            cash_events: cash_times.len(),
        })?;
    let base_nodes =
        base_intervals_usize
            .checked_add(1)
            .ok_or(BatchLsmError::ContractMeshShapeOverflow {
                row,
                base_intervals,
                cash_events: cash_times.len(),
            })?;
    let capacity = base_nodes.checked_add(cash_times.len()).ok_or(
        BatchLsmError::ContractMeshShapeOverflow {
            row,
            base_intervals,
            cash_events: cash_times.len(),
        },
    )?;
    if capacity.saturating_sub(1) > u32::MAX as usize {
        return Err(BatchLsmError::ContractMeshShapeOverflow {
            row,
            base_intervals,
            cash_events: cash_times.len(),
        });
    }
    let mut times = Vec::new();
    times
        .try_reserve_exact(capacity)
        .map_err(|_| BatchLsmError::ContractMeshAllocationFailed {
            row,
            nodes: capacity,
        })?;
    let mut base_node = Some(0_u32);
    let mut event = 0_usize;
    while base_node.is_some() || event < cash_times.len() {
        let base_time = base_node.map(|node| {
            if node == base_intervals {
                maturity
            } else {
                maturity * (f64::from(node) / f64::from(base_intervals))
            }
        });
        let event_time = cash_times.get(event).copied();
        match (base_time, event_time) {
            (Some(base), Some(cash)) => match base.total_cmp(&cash) {
                std::cmp::Ordering::Less => {
                    times.push(base);
                    base_node = base_node.and_then(|node| next_base_node(node, base_intervals));
                }
                std::cmp::Ordering::Equal => {
                    times.push(base);
                    base_node = base_node.and_then(|node| next_base_node(node, base_intervals));
                    event += 1;
                }
                std::cmp::Ordering::Greater => {
                    times.push(cash);
                    event += 1;
                }
            },
            (Some(base), None) => {
                times.push(base);
                base_node = base_node.and_then(|node| next_base_node(node, base_intervals));
            }
            (None, Some(cash)) => {
                times.push(cash);
                event += 1;
            }
            (None, None) => break,
        }
    }
    Ok(times)
}

fn next_base_node(node: u32, final_node: u32) -> Option<u32> {
    if node == final_node {
        None
    } else {
        node.checked_add(1)
    }
}

impl BatchLsmCallRowResult {
    fn try_from_run(row: usize, run: LsmRun) -> Result<Self, BatchLsmError> {
        let (policy, evaluation) = run.into_parts();
        Self::try_from_parts(row, &policy, &evaluation)
    }

    fn try_from_parts(
        row: usize,
        policy: &FittedLsmCallPolicy,
        evaluation: &LsmEvaluation,
    ) -> Result<Self, BatchLsmError> {
        let mut origin_decision_fitted = false;
        let mut origin_exercised = false;
        let mut regression_nodes = 0_usize;
        let mut no_training_support_nodes = 0_usize;
        let mut minimum_in_the_money_training_paths = u32::MAX;
        let mut maximum_regression_condition_number = 0.0_f64;
        for diagnostic in policy.diagnostics() {
            match diagnostic.fit() {
                ExerciseFitDiagnostics::OriginDecision { exercise, .. } => {
                    origin_decision_fitted = true;
                    origin_exercised = exercise;
                }
                ExerciseFitDiagnostics::NoInTheMoneyPaths => {
                    no_training_support_nodes += 1;
                }
                ExerciseFitDiagnostics::Regression {
                    in_the_money_paths,
                    diagnostics,
                } => {
                    regression_nodes += 1;
                    minimum_in_the_money_training_paths =
                        minimum_in_the_money_training_paths.min(in_the_money_paths);
                    maximum_regression_condition_number = maximum_regression_condition_number
                        .max(RegressionDiagnostics::condition_number(diagnostics));
                }
            }
        }
        if regression_nodes == 0 {
            minimum_in_the_money_training_paths = 0;
        }
        let counts = evaluation.exercise_counts();
        let expiry_paths = *counts.last().ok_or(BatchLsmError::StoppingCountMismatch {
            row,
            expected: evaluation.paths(),
            observed: 0,
        })?;
        let early_exercise_total = counts[..counts.len() - 1]
            .iter()
            .map(|&count| u64::from(count))
            .sum::<u64>();
        let observed_total = early_exercise_total + u64::from(expiry_paths);
        if observed_total != u64::from(evaluation.paths()) {
            return Err(BatchLsmError::StoppingCountMismatch {
                row,
                expected: evaluation.paths(),
                observed: observed_total,
            });
        }
        Ok(Self {
            price: evaluation.price(),
            sample_standard_deviation: evaluation.sample_standard_deviation().ok_or(
                BatchLsmError::MissingSamplingDiagnostic {
                    field: "sample_standard_deviation",
                    row,
                },
            )?,
            standard_error: evaluation.standard_error().ok_or(
                BatchLsmError::MissingSamplingDiagnostic {
                    field: "standard_error",
                    row,
                },
            )?,
            origin_decision_fitted,
            origin_exercised,
            regression_nodes: u32::try_from(regression_nodes).map_err(|_| {
                BatchLsmError::DiagnosticOverflow {
                    field: "regression_nodes",
                    row,
                }
            })?,
            no_training_support_nodes: u32::try_from(no_training_support_nodes).map_err(|_| {
                BatchLsmError::DiagnosticOverflow {
                    field: "no_training_support_nodes",
                    row,
                }
            })?,
            minimum_in_the_money_training_paths,
            maximum_regression_condition_number,
            early_exercise_paths: u32::try_from(early_exercise_total).map_err(|_| {
                BatchLsmError::DiagnosticOverflow {
                    field: "early_exercise_paths",
                    row,
                }
            })?,
            expiry_paths,
        })
    }
}

#[cfg(test)]
mod tests {
    use finoptions_batch::RaggedCashDividends;

    use super::*;
    use crate::{ExerciseTieRule, PolynomialDegree, PolynomialRegressionSpec};

    fn regression() -> PolynomialRegressionSpec {
        PolynomialRegressionSpec::try_new(PolynomialDegree::try_new(4).unwrap(), 1.0e-12, 1.0e8)
            .unwrap()
    }

    fn method(origin: OriginExercise) -> BatchLsmCallMethod {
        BatchLsmCallMethod::try_new(
            super::super::UniformExerciseMesh::try_new(8, origin).unwrap(),
            2_048,
            2_048,
            regression(),
            ExerciseTieRule::StrictlyGreater,
            super::super::BatchRandomStreams::new(7, 11, 0, 10_000),
        )
        .unwrap()
    }

    fn valid_continuous_inputs<'a>(
        spots: &'a [f64],
        strikes: &'a [f64],
        maturities: &'a [f64],
        rates: &'a [f64],
        volatilities: &'a [f64],
        yields: &'a [f64],
    ) -> BatchLsmCallInputs<'a> {
        BatchLsmCallInputs::try_new(
            spots,
            strikes,
            maturities,
            rates,
            volatilities,
            BatchDividendModel::ContinuousYield {
                annualized_decimals: yields,
            },
        )
        .unwrap()
    }

    fn valid_grouped_continuous_inputs<'a>(
        group_offsets: &'a [u32],
        spots: &'a [f64],
        strikes: &'a [f64],
        maturities: &'a [f64],
        rates: &'a [f64],
        volatilities: &'a [f64],
        yields: &'a [f64],
    ) -> GroupedBatchLsmCallInputs<'a> {
        GroupedBatchLsmCallInputs::try_new(
            group_offsets,
            spots,
            strikes,
            maturities,
            rates,
            volatilities,
            BatchDividendModel::ContinuousYield {
                annualized_decimals: yields,
            },
        )
        .unwrap()
    }

    #[test]
    fn exact_union_mesh_inserts_cash_times_without_rounding_or_duplication() {
        assert_eq!(
            exact_union_times(0, 1.0, 4, &[0.1, 0.5, 0.9]).unwrap(),
            [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        );

        let immediately_before_half = f64::from_bits(0.5_f64.to_bits() - 1);
        assert_eq!(
            exact_union_times(0, 1.0, 2, &[immediately_before_half]).unwrap(),
            [0.0, immediately_before_half, 0.5, 1.0]
        );
        assert_eq!(next_base_node(u32::MAX - 1, u32::MAX), Some(u32::MAX));
        assert_eq!(next_base_node(u32::MAX, u32::MAX), None);

        let maximum_maturity_mesh = exact_union_times(0, f64::MAX, 4, &[]).unwrap();
        assert_eq!(maximum_maturity_mesh.last(), Some(&f64::MAX));
        assert!(maximum_maturity_mesh.iter().all(|time| time.is_finite()));
    }

    #[test]
    fn rejects_empty_and_misaligned_batches() {
        let spots = [100.0, 120.0];
        let strikes = [100.0, 110.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];

        assert_eq!(
            BatchLsmCallInputs::try_new(
                &[],
                &[],
                &[],
                &[],
                &[],
                BatchDividendModel::ContinuousYield {
                    annualized_decimals: &[],
                },
            )
            .unwrap_err(),
            BatchLsmError::EmptyBatch
        );
        assert!(matches!(
            BatchLsmCallInputs::try_new(
                &spots,
                &strikes[..1],
                &maturities,
                &rates,
                &volatilities,
                BatchDividendModel::ContinuousYield {
                    annualized_decimals: &yields,
                },
            ),
            Err(BatchLsmError::RowCountMismatch {
                field: "strikes",
                expected: 2,
                observed: 1,
            })
        ));
    }

    #[test]
    fn grouped_boundaries_and_compact_axes_fail_before_simulation() {
        let spots = [100.0, 120.0];
        let strikes = [90.0, 100.0, 110.0, 120.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];
        let dividends = BatchDividendModel::ContinuousYield {
            annualized_decimals: &yields,
        };
        assert!(matches!(
            GroupedBatchLsmCallInputs::try_new(
                &[0, 2],
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                dividends,
            ),
            Err(BatchLsmError::GroupOffsetCountMismatch {
                expected: 3,
                observed: 2,
            })
        ));
        assert!(matches!(
            GroupedBatchLsmCallInputs::try_new(
                &[1, 2, 4],
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                dividends,
            ),
            Err(BatchLsmError::GroupOffsetsDoNotStartAtZero { observed: 1 })
        ));
        assert!(matches!(
            GroupedBatchLsmCallInputs::try_new(
                &[0, 0, 4],
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                dividends,
            ),
            Err(BatchLsmError::InvalidGroupRange {
                group: 0,
                start: 0,
                end: 0,
            })
        ));
        assert!(matches!(
            GroupedBatchLsmCallInputs::try_new(
                &[0, 2, 3],
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                dividends,
            ),
            Err(BatchLsmError::GroupOffsetEndMismatch {
                expected: 4,
                observed: 3,
            })
        ));
    }

    #[test]
    fn one_contract_groups_match_row_addressed_batches_for_all_dividend_models_bitwise() {
        let group_offsets = [0, 1, 2];
        let spots = [100.0, 120.0];
        let strikes = [100.0, 110.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];
        let pricing_method = method(OriginExercise::Excluded);
        let executor = CpuExecutor::try_new(1).unwrap();

        let row_continuous = valid_continuous_inputs(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        )
        .price(pricing_method, &executor)
        .unwrap();
        let grouped_continuous = valid_grouped_continuous_inputs(
            &group_offsets,
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        )
        .price(pricing_method, &executor)
        .unwrap();
        assert_eq!(grouped_continuous, row_continuous);

        let dividends =
            RaggedCashDividends::try_new(&[0, 2, 3], &[0.2, 0.6, 0.4], &[1.0, 1.0, 2.0]).unwrap();
        for dividend_model in [
            BatchDividendModel::CashDividends {
                schedules: dividends,
                policy: DividendPolicy::Survivor,
            },
            BatchDividendModel::EscrowedCashDividends {
                schedules: dividends,
                volatility_convention: EscrowedVolatilityConvention::SpotMatchedAtOrigin,
            },
        ] {
            let row = BatchLsmCallInputs::try_new(
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                dividend_model,
            )
            .unwrap()
            .price(pricing_method, &executor)
            .unwrap();
            let grouped = GroupedBatchLsmCallInputs::try_new(
                &group_offsets,
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                dividend_model,
            )
            .unwrap()
            .price(pricing_method, &executor)
            .unwrap();
            assert_eq!(grouped, row);
        }
    }

    #[test]
    fn shared_group_uses_one_population_without_changing_strike_specific_results() {
        let group_offsets = [0, 3];
        let spots = [100.0];
        let strikes = [95.0, 100.0, 100.0];
        let maturities = [1.0];
        let rates = [0.05];
        let volatilities = [0.25];
        let yields = [0.02];
        let results = valid_grouped_continuous_inputs(
            &group_offsets,
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        )
        .price(
            method(OriginExercise::Excluded),
            &CpuExecutor::try_new(1).unwrap(),
        )
        .unwrap();

        assert_ne!(results.as_slice()[0], results.as_slice()[1]);
        assert_eq!(results.as_slice()[1], results.as_slice()[2]);
    }

    #[test]
    fn grouped_serial_and_parallel_results_are_bitwise_identical_for_all_dividend_models() {
        const GROUPS: usize = 10;
        let group_offsets = (0..=GROUPS)
            .map(|group| u32::try_from(group * 2).unwrap())
            .collect::<Vec<_>>();
        let spots = [100.0; GROUPS];
        let strikes = [95.0, 105.0].repeat(GROUPS);
        let maturities = [1.0; GROUPS];
        let rates = [0.05; GROUPS];
        let volatilities = [0.25; GROUPS];
        let yields = [0.02; GROUPS];
        let cash_offsets = (0..=GROUPS)
            .map(|group| i64::try_from(group).unwrap())
            .collect::<Vec<_>>();
        let cash_times = [0.3].repeat(GROUPS);
        let cash_amounts = [2.0].repeat(GROUPS);
        let dividends =
            RaggedCashDividends::try_new(&cash_offsets, &cash_times, &cash_amounts).unwrap();
        let pricing_method = method(OriginExercise::Excluded);
        let serial_executor = CpuExecutor::try_new(1).unwrap();
        let parallel_executor = CpuExecutor::try_new(4).unwrap();

        for dividend_model in [
            BatchDividendModel::ContinuousYield {
                annualized_decimals: &yields,
            },
            BatchDividendModel::CashDividends {
                schedules: dividends,
                policy: DividendPolicy::Survivor,
            },
            BatchDividendModel::EscrowedCashDividends {
                schedules: dividends,
                volatility_convention: EscrowedVolatilityConvention::SpotMatchedAtOrigin,
            },
        ] {
            let inputs = GroupedBatchLsmCallInputs::try_new(
                &group_offsets,
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                dividend_model,
            )
            .unwrap();
            let serial = inputs.price(pricing_method, &serial_executor).unwrap();
            let parallel = inputs.price(pricing_method, &parallel_executor).unwrap();
            assert_eq!(parallel, serial);
            assert_eq!(serial.len(), strikes.len());
        }
    }

    #[test]
    fn indexed_group_scheduler_completes_at_project_scale_group_count() {
        const GROUPS: usize = 74_215;
        let group_offsets = (0..=GROUPS)
            .map(|group| u32::try_from(group).unwrap())
            .collect::<Vec<_>>();
        let spots = vec![100.0; GROUPS];
        let strikes = vec![100.0; GROUPS];
        let maturities = vec![1.0; GROUPS];
        let rates = vec![0.05; GROUPS];
        let volatilities = vec![0.25; GROUPS];
        let yields = vec![0.02; GROUPS];
        let light_regression = PolynomialRegressionSpec::try_new(
            PolynomialDegree::try_new(1).unwrap(),
            1.0e-12,
            1.0e8,
        )
        .unwrap();
        let light_method = BatchLsmCallMethod::try_new(
            super::super::UniformExerciseMesh::try_new(1, OriginExercise::Excluded).unwrap(),
            2,
            2,
            light_regression,
            ExerciseTieRule::StrictlyGreater,
            super::super::BatchRandomStreams::new(17, 19, 0, 1_000_000),
        )
        .unwrap();

        let result = valid_grouped_continuous_inputs(
            &group_offsets,
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        )
        .price(light_method, &CpuExecutor::try_new(4).unwrap())
        .unwrap();

        assert_eq!(result.len(), GROUPS);
        assert!(
            result
                .as_slice()
                .iter()
                .all(|row| row.early_exercise_paths() == 0 && row.expiry_paths() == 2)
        );
    }

    #[test]
    fn grouped_parallel_failure_replays_the_lowest_contract_row() {
        let group_offsets = [0, 2, 4, 6];
        let spots = [100.0; 3];
        let strikes = [95.0, 105.0].repeat(3);
        let maturities = [1.0; 3];
        let rates = [0.05, 10_000.0, 10_000.0];
        let volatilities = [0.25; 3];
        let yields = [0.02; 3];
        let inputs = valid_grouped_continuous_inputs(
            &group_offsets,
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        );
        let pricing_method = method(OriginExercise::Excluded);
        let serial_error = inputs
            .price(pricing_method, &CpuExecutor::try_new(1).unwrap())
            .unwrap_err();
        let parallel_error = inputs
            .price(pricing_method, &CpuExecutor::try_new(4).unwrap())
            .unwrap_err();

        assert_eq!(parallel_error, serial_error);
        assert!(matches!(
            parallel_error,
            BatchLsmError::FinancialInputAtRow { row: 2, .. }
        ));
    }

    #[test]
    fn validates_every_row_scalar_domain_without_substitution() {
        let spots = [100.0, 120.0];
        let strikes = [100.0, 110.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];
        for (
            field,
            invalid_spots,
            invalid_strikes,
            invalid_maturities,
            invalid_rates,
            invalid_vols,
            invalid_yields,
        ) in [
            (
                "initial_spots",
                [-0.0, 120.0],
                strikes,
                maturities,
                rates,
                volatilities,
                yields,
            ),
            (
                "strikes",
                spots,
                [0.0, 110.0],
                maturities,
                rates,
                volatilities,
                yields,
            ),
            (
                "maturities",
                spots,
                strikes,
                [f64::NAN, 0.75],
                rates,
                volatilities,
                yields,
            ),
            (
                "constant_continuously_compounded_rates",
                spots,
                strikes,
                maturities,
                [f64::INFINITY, 0.03],
                volatilities,
                yields,
            ),
            (
                "constant_volatilities",
                spots,
                strikes,
                maturities,
                rates,
                [0.0, 0.30],
                yields,
            ),
            (
                "continuous_dividend_yields",
                spots,
                strikes,
                maturities,
                rates,
                volatilities,
                [-0.0, 0.01],
            ),
        ] {
            assert!(matches!(
                BatchLsmCallInputs::try_new(
                    &invalid_spots,
                    &invalid_strikes,
                    &invalid_maturities,
                    &invalid_rates,
                    &invalid_vols,
                    BatchDividendModel::ContinuousYield {
                        annualized_decimals: &invalid_yields,
                    },
                ),
                Err(BatchLsmError::InvalidRowValue { field: observed_field, row: 0, .. })
                    if observed_field == field
            ));
        }
    }

    #[test]
    fn validates_all_batch_method_boundaries_without_substitution() {
        assert_eq!(
            super::super::UniformExerciseMesh::try_new(0, OriginExercise::Excluded).unwrap_err(),
            BatchLsmError::EmptyBaseIntervals
        );
        assert!(matches!(
            BatchLsmCallMethod::try_new(
                super::super::UniformExerciseMesh::try_new(8, OriginExercise::Excluded).unwrap(),
                0,
                2,
                regression(),
                ExerciseTieRule::StrictlyGreater,
                super::super::BatchRandomStreams::new(1, 2, 0, 10),
            ),
            Err(BatchLsmError::PathCountOutsideDomain {
                population: "training",
                observed: 0,
                ..
            })
        ));
        assert!(matches!(
            BatchLsmCallMethod::try_new(
                super::super::UniformExerciseMesh::try_new(8, OriginExercise::Excluded).unwrap(),
                2,
                1,
                regression(),
                ExerciseTieRule::StrictlyGreater,
                super::super::BatchRandomStreams::new(1, 2, 0, 10),
            ),
            Err(BatchLsmError::InsufficientEvaluationPaths { observed: 1 })
        ));
    }

    #[test]
    fn rejects_ambiguous_or_expiry_cash_events() {
        let spots = [100.0];
        let strikes = [100.0];
        let maturities = [1.0];
        let rates = [0.05];
        let volatilities = [0.25];

        let duplicate = RaggedCashDividends::try_new(&[0, 2], &[0.25, 0.25], &[1.0, 2.0]).unwrap();
        assert!(matches!(
            BatchLsmCallInputs::try_new(
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                BatchDividendModel::CashDividends {
                    schedules: duplicate,
                    policy: DividendPolicy::Survivor,
                },
            ),
            Err(BatchLsmError::DuplicateCashDividendTime {
                row: 0,
                event: 1,
                time: 0.25,
            })
        ));

        let at_expiry = RaggedCashDividends::try_new(&[0, 1], &[1.0], &[1.0]).unwrap();
        assert!(matches!(
            BatchLsmCallInputs::try_new(
                &spots,
                &strikes,
                &maturities,
                &rates,
                &volatilities,
                BatchDividendModel::CashDividends {
                    schedules: at_expiry,
                    policy: DividendPolicy::Survivor,
                },
            ),
            Err(BatchLsmError::CashDividendOutsideContract {
                row: 0,
                event: 0,
                time: 1.0,
                maturity: 1.0,
            })
        ));
    }

    #[test]
    fn rejects_overflowing_or_colliding_row_random_domains_before_execution() {
        let spots = [100.0, 120.0];
        let strikes = [100.0, 110.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];
        let inputs = valid_continuous_inputs(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        );
        let base = method(OriginExercise::Excluded);
        let overflow = BatchLsmCallMethod::try_new(
            base.mesh(),
            base.training_paths(),
            base.evaluation_paths(),
            base.regression(),
            base.tie_rule(),
            super::super::BatchRandomStreams::new(7, 11, u32::MAX, 10_000),
        )
        .unwrap();
        assert!(matches!(
            inputs.price(overflow, &CpuExecutor::try_new(1).unwrap()),
            Err(BatchLsmError::RandomDomainRangeOverflow {
                population: "training",
                start: u32::MAX,
                rows: 2,
            })
        ));

        let collision = BatchLsmCallMethod::try_new(
            base.mesh(),
            base.training_paths(),
            base.evaluation_paths(),
            base.regression(),
            base.tie_rule(),
            super::super::BatchRandomStreams::new(7, 11, 100, 101),
        )
        .unwrap();
        assert!(matches!(
            inputs.price(collision, &CpuExecutor::try_new(1).unwrap()),
            Err(BatchLsmError::RandomDomainRangeCollision {
                training_start: 100,
                training_end: 102,
                evaluation_start: 101,
                evaluation_end: 103,
            })
        ));
    }

    #[test]
    fn parallel_failure_is_the_same_typed_lowest_row_as_serial_execution() {
        let spots = [100.0; 4];
        let strikes = [100.0; 4];
        let maturities = [1.0; 4];
        let rates = [0.05, 10_000.0, 0.03, 10_000.0];
        let volatilities = [0.25; 4];
        let yields = [0.02; 4];
        let inputs = valid_continuous_inputs(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        );
        let serial_error = inputs
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(1).unwrap(),
            )
            .unwrap_err();
        let parallel_error = inputs
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(4).unwrap(),
            )
            .unwrap_err();
        assert_eq!(parallel_error, serial_error);
        assert!(matches!(
            parallel_error,
            BatchLsmError::FinancialInputAtRow { row: 1, .. }
        ));
    }

    #[test]
    fn one_row_batch_matches_an_independently_constructed_scalar_plan_bitwise() {
        let spots = [100.0];
        let strikes = [100.0];
        let maturities = [1.0];
        let rates = [0.05];
        let volatilities = [0.25];
        let yields = [0.02];
        let batch_inputs = valid_continuous_inputs(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        );
        let batch_method = method(OriginExercise::Excluded);
        let batch_result = batch_inputs
            .price(batch_method, &CpuExecutor::try_new(1).unwrap())
            .unwrap();

        let times = (0_u32..=8)
            .map(|node| f64::from(node) / 8.0)
            .collect::<Vec<_>>();
        let grid = TimeGrid::try_new(times).unwrap();
        let log_discounts = grid
            .as_slice()
            .iter()
            .enumerate()
            .map(|(node, &time)| if node == 0 { 0.0 } else { -rates[0] * time })
            .collect::<Vec<_>>();
        let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
        let cumulative_yields = grid
            .as_slice()
            .iter()
            .enumerate()
            .map(|(node, &time)| if node == 0 { 0.0 } else { yields[0] * time })
            .collect::<Vec<_>>();
        let yield_curve = CumulativeDividendYieldCurve::try_new(&grid, cumulative_yields).unwrap();
        let exercise_nodes = (1..grid.len() - 1)
            .map(|node| grid.node_id(node).unwrap())
            .collect::<Vec<_>>();
        let expiry = grid.node_id(grid.len() - 1).unwrap();
        let schedule = EventSchedule::try_new(grid, [], exercise_nodes, expiry).unwrap();
        let plan = GbmPathPlan::try_new(
            &schedule,
            &curve,
            &yield_curve,
            ConstantVolatility::try_new(volatilities[0]).unwrap(),
            DividendPolicy::Liquidator,
        )
        .unwrap();
        let scalar_run = LsmCallEstimator::try_new(
            strikes[0],
            batch_method.regression(),
            batch_method.tie_rule(),
        )
        .unwrap()
        .price_seeded_serial(
            &plan,
            &curve,
            spots[0],
            batch_method.training_paths(),
            batch_method.evaluation_paths(),
            RandomStream::new(7, 0),
            RandomStream::new(11, 10_000),
        )
        .unwrap();
        let scalar_result = BatchLsmCallRowResult::try_from_run(0, scalar_run).unwrap();
        assert_eq!(batch_result.as_slice(), &[scalar_result]);
    }

    #[test]
    fn off_grid_cash_batch_matches_an_independently_constructed_scalar_plan_bitwise() {
        let spots = [100.0];
        let strikes = [100.0];
        let maturities = [1.0];
        let rates = [0.05];
        let volatilities = [0.25];
        let dividends = RaggedCashDividends::try_new(&[0, 1], &[0.3], &[2.0]).unwrap();
        let batch_inputs = BatchLsmCallInputs::try_new(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            BatchDividendModel::CashDividends {
                schedules: dividends,
                policy: DividendPolicy::Survivor,
            },
        )
        .unwrap();
        let batch_method = method(OriginExercise::Excluded);
        let batch_result = batch_inputs
            .price(batch_method, &CpuExecutor::try_new(1).unwrap())
            .unwrap();

        let grid = TimeGrid::try_new(vec![
            0.0, 0.125, 0.25, 0.3, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
        ])
        .unwrap();
        let log_discounts = grid
            .as_slice()
            .iter()
            .enumerate()
            .map(|(node, &time)| if node == 0 { 0.0 } else { -rates[0] * time })
            .collect::<Vec<_>>();
        let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
        let yield_curve =
            CumulativeDividendYieldCurve::try_new(&grid, vec![0.0; grid.len()]).unwrap();
        let cash = CashDividend::try_new(grid.node_id(3).unwrap(), 2.0).unwrap();
        let exercise_nodes = (1..grid.len() - 1)
            .map(|node| grid.node_id(node).unwrap())
            .collect::<Vec<_>>();
        let expiry = grid.node_id(grid.len() - 1).unwrap();
        let schedule = EventSchedule::try_new(grid, [cash], exercise_nodes, expiry).unwrap();
        let plan = GbmPathPlan::try_new(
            &schedule,
            &curve,
            &yield_curve,
            ConstantVolatility::try_new(volatilities[0]).unwrap(),
            DividendPolicy::Survivor,
        )
        .unwrap();
        let scalar_run = LsmCallEstimator::try_new(
            strikes[0],
            batch_method.regression(),
            batch_method.tie_rule(),
        )
        .unwrap()
        .price_seeded_serial(
            &plan,
            &curve,
            spots[0],
            batch_method.training_paths(),
            batch_method.evaluation_paths(),
            RandomStream::new(7, 0),
            RandomStream::new(11, 10_000),
        )
        .unwrap();
        let scalar_result = BatchLsmCallRowResult::try_from_run(0, scalar_run).unwrap();
        assert_eq!(batch_result.as_slice(), &[scalar_result]);
    }

    #[test]
    fn escrowed_batch_matches_an_independently_constructed_scalar_plan_bitwise() {
        let spots = [100.0];
        let strikes = [100.0];
        let maturities = [1.0];
        let rates = [0.05];
        let spot_volatilities = [0.25];
        let dividends = RaggedCashDividends::try_new(&[0, 1], &[0.3], &[2.0]).unwrap();
        let batch_inputs = BatchLsmCallInputs::try_new(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &spot_volatilities,
            BatchDividendModel::EscrowedCashDividends {
                schedules: dividends,
                volatility_convention: EscrowedVolatilityConvention::SpotMatchedAtOrigin,
            },
        )
        .unwrap();
        let batch_method = method(OriginExercise::Excluded);
        let batch_result = batch_inputs
            .price(batch_method, &CpuExecutor::try_new(1).unwrap())
            .unwrap();

        let grid = TimeGrid::try_new(vec![
            0.0, 0.125, 0.25, 0.3, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
        ])
        .unwrap();
        let log_discounts = grid
            .as_slice()
            .iter()
            .enumerate()
            .map(|(node, &time)| if node == 0 { 0.0 } else { -rates[0] * time })
            .collect::<Vec<_>>();
        let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
        let cash = CashDividend::try_new(grid.node_id(3).unwrap(), 2.0).unwrap();
        let exercise_nodes = (1..grid.len() - 1)
            .map(|node| grid.node_id(node).unwrap())
            .collect::<Vec<_>>();
        let expiry = grid.node_id(grid.len() - 1).unwrap();
        let schedule = EventSchedule::try_new(grid, [cash], exercise_nodes, expiry).unwrap();
        let prepaid_cash = 2.0 * (-rates[0] * 0.3_f64).exp();
        let prepaid_forward = spots[0] - prepaid_cash;
        let prepaid_forward_volatility = spot_volatilities[0] * spots[0] / prepaid_forward;
        let plan = EscrowedGbmPathPlan::try_new(
            &schedule,
            &curve,
            ConstantVolatility::try_new(prepaid_forward_volatility).unwrap(),
        )
        .unwrap();
        let scalar_run = LsmCallEstimator::try_new(
            strikes[0],
            batch_method.regression(),
            batch_method.tie_rule(),
        )
        .unwrap()
        .price_escrowed_seeded_serial(
            &plan,
            &curve,
            spots[0],
            batch_method.training_paths(),
            batch_method.evaluation_paths(),
            RandomStream::new(7, 0),
            RandomStream::new(11, 10_000),
        )
        .unwrap();
        let scalar_result = BatchLsmCallRowResult::try_from_run(0, scalar_run).unwrap();
        assert_eq!(batch_result.as_slice(), &[scalar_result]);
    }

    #[test]
    fn serial_and_parallel_batches_are_bitwise_identical_for_all_dividend_models() {
        let spots = [100.0, 120.0];
        let strikes = [100.0, 110.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];
        let continuous = valid_continuous_inputs(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        );
        let serial = continuous
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(1).unwrap(),
            )
            .unwrap();
        let parallel = continuous
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(4).unwrap(),
            )
            .unwrap();
        assert_eq!(serial, parallel);

        let dividends =
            RaggedCashDividends::try_new(&[0, 2, 3], &[0.2, 0.6, 0.4], &[1.0, 1.0, 2.0]).unwrap();
        let cash = BatchLsmCallInputs::try_new(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            BatchDividendModel::CashDividends {
                schedules: dividends,
                policy: DividendPolicy::Survivor,
            },
        )
        .unwrap();
        let serial = cash
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(1).unwrap(),
            )
            .unwrap();
        let parallel = cash
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(4).unwrap(),
            )
            .unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(serial.len(), spots.len());
        for result in serial.as_slice() {
            assert!(result.price().is_finite());
            assert!(result.standard_error() > 0.0);
            assert_eq!(result.early_exercise_paths() + result.expiry_paths(), 2_048);
        }

        let escrowed = BatchLsmCallInputs::try_new(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            BatchDividendModel::EscrowedCashDividends {
                schedules: dividends,
                volatility_convention: EscrowedVolatilityConvention::SpotMatchedAtOrigin,
            },
        )
        .unwrap();
        let serial = escrowed
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(1).unwrap(),
            )
            .unwrap();
        let parallel = escrowed
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(4).unwrap(),
            )
            .unwrap();
        assert_eq!(serial, parallel);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apple_vector_batch_is_thread_invariant_and_respects_call_bounds() {
        let spots = [100.0, 120.0];
        let strikes = [100.0, 110.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];
        let inputs = valid_continuous_inputs(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        );
        let serial_executor =
            CpuExecutor::try_new_with_math_backend(1, CpuMathBackend::AppleAccelerate).unwrap();
        let parallel_executor =
            CpuExecutor::try_new_with_math_backend(4, CpuMathBackend::AppleAccelerate).unwrap();
        let serial = inputs
            .price(method(OriginExercise::Excluded), &serial_executor)
            .unwrap();
        let parallel = inputs
            .price(method(OriginExercise::Excluded), &parallel_executor)
            .unwrap();
        assert_eq!(serial, parallel);
        for (&spot, result) in spots.iter().zip(serial.as_slice()) {
            assert!(result.price() >= 0.0);
            assert!(result.price() <= spot);
            assert!(result.standard_error().is_finite());
            assert_eq!(result.early_exercise_paths() + result.expiry_paths(), 2_048);
        }
    }

    #[test]
    fn row_addressing_matches_independent_single_row_batches() {
        let spots = [100.0, 120.0];
        let strikes = [100.0, 110.0];
        let maturities = [1.0, 0.75];
        let rates = [0.05, 0.03];
        let volatilities = [0.25, 0.30];
        let yields = [0.02, 0.01];
        let inputs = valid_continuous_inputs(
            &spots,
            &strikes,
            &maturities,
            &rates,
            &volatilities,
            &yields,
        );
        let combined = inputs
            .price(
                method(OriginExercise::Excluded),
                &CpuExecutor::try_new(2).unwrap(),
            )
            .unwrap();

        let single = valid_continuous_inputs(
            &spots[1..],
            &strikes[1..],
            &maturities[1..],
            &rates[1..],
            &volatilities[1..],
            &yields[1..],
        );
        let base = method(OriginExercise::Excluded);
        let shifted = BatchLsmCallMethod::try_new(
            base.mesh(),
            base.training_paths(),
            base.evaluation_paths(),
            base.regression(),
            base.tie_rule(),
            super::super::BatchRandomStreams::new(7, 11, 1, 10_001),
        )
        .unwrap();
        let independent = single
            .price(shifted, &CpuExecutor::try_new(1).unwrap())
            .unwrap();
        assert_eq!(combined.as_slice()[1], independent.as_slice()[0]);
    }
}
