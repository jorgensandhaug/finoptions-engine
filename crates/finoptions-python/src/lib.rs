//! Private Python extension. The stable public surface lives in `python/finoptions`.

use finoptions_batch::{
    BatchError as CoreBatchError, CashAmountScale as CoreCashAmountScale,
    DividendBinSpec as CoreDividendBinSpec, MaturityBoundary as CoreMaturityBoundary,
    PaddedSequenceSpec as CorePaddedSequenceSpec, RaggedCashDividends,
    TimeBinClosure as CoreTimeBinClosure, bin_dividends_f32 as core_bin_dividends_f32,
    pack_padded_dividends_f32 as core_pack_padded_dividends_f32,
};
use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    InputError as CoreInputError, LogDiscountCurve, TimeGrid,
};
use finoptions_lsm::{
    ExerciseFitDiagnostics as CoreExerciseFitDiagnostics, ExerciseTieRule as CoreExerciseTieRule,
    LsmCallEstimator as CoreLsmCallEstimator, LsmError as CoreLsmError, LsmRun as CoreLsmRun,
    PolynomialDegree as CorePolynomialDegree,
    PolynomialRegressionSpec as CorePolynomialRegressionSpec,
};
use finoptions_runtime::{CpuExecutor as CoreCpuExecutor, ExecutionError as CoreExecutionError};
use finoptions_sim::{
    GbmPathPlan, RandomStream as CoreRandomStream, SimulatedPaths,
    SimulationError as CoreSimulationError,
};
use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

create_exception!(_native, InputError, PyValueError);
create_exception!(_native, BatchError, PyValueError);
create_exception!(_native, ExecutionError, PyRuntimeError);
create_exception!(_native, SimulationError, PyRuntimeError);
create_exception!(_native, LsmError, PyRuntimeError);

/// Dedicated native CPU worker pool owned by the public Python execution specification.
#[pyclass(name = "CpuExecutor", frozen, module = "finoptions._native")]
struct NativeCpuExecutor {
    inner: CoreCpuExecutor,
}

#[pymethods]
impl NativeCpuExecutor {
    #[new]
    fn new(threads: usize) -> PyResult<Self> {
        CoreCpuExecutor::try_new(threads)
            .map(|inner| Self { inner })
            .map_err(CoreExecutionError::into_py_execution_error)
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }
}

trait IntoPyExecutionError {
    fn into_py_execution_error(self) -> PyErr;
}

impl IntoPyExecutionError for CoreExecutionError {
    fn into_py_execution_error(self) -> PyErr {
        let message = self.to_string();
        match self {
            CoreExecutionError::EmptyThreadPool => InputError::new_err(message),
            _ => ExecutionError::new_err(message),
        }
    }
}

trait IntoPyBatchError {
    fn into_py_batch_error(self) -> PyErr;
}

impl IntoPyBatchError for CoreBatchError {
    fn into_py_batch_error(self) -> PyErr {
        BatchError::new_err(self.to_string())
    }
}

trait IntoPyInputError {
    fn into_py_input_error(self) -> PyErr;
}

impl IntoPyInputError for CoreInputError {
    fn into_py_input_error(self) -> PyErr {
        InputError::new_err(self.to_string())
    }
}

trait IntoPySimulationError {
    fn into_py_simulation_error(self) -> PyErr;
}

impl IntoPySimulationError for CoreSimulationError {
    fn into_py_simulation_error(self) -> PyErr {
        let message = self.to_string();
        match self {
            CoreSimulationError::NonFiniteTransition { .. }
            | CoreSimulationError::NonFinitePathTransition { .. }
            | CoreSimulationError::RandomNormalBudgetExhausted { .. }
            | CoreSimulationError::RandomNormalRejectionLimit { .. }
            | CoreSimulationError::ParallelFailureReplayMismatch { .. } => {
                SimulationError::new_err(message)
            }
            _ => InputError::new_err(message),
        }
    }
}

trait IntoPyLsmError {
    fn into_py_lsm_error(self) -> PyErr;
}

impl IntoPyLsmError for CoreLsmError {
    fn into_py_lsm_error(self) -> PyErr {
        LsmError::new_err(self.to_string())
    }
}

/// Returns the native package and compiler versions for immutable run manifests.
#[pyfunction]
fn build_info() -> (&'static str, &'static str) {
    (env!("CARGO_PKG_VERSION"), env!("RUSTC_VERSION", "unknown"))
}

/// Validates one cash-dividend amount independently of a schedule.
#[pyfunction]
fn validate_cash_dividend_amount(amount: f64) -> PyResult<()> {
    let grid = TimeGrid::try_new([0.0, 1.0]).map_err(CoreInputError::into_py_input_error)?;
    let synthetic_node = grid
        .node_id(1)
        .map_err(CoreInputError::into_py_input_error)?;
    CashDividend::try_new(synthetic_node, amount)
        .map(|_| ())
        .map_err(CoreInputError::into_py_input_error)
}

/// Validates annualized decimal constant volatility.
#[pyfunction]
fn validate_constant_volatility(annualized_decimal: f64) -> PyResult<()> {
    ConstantVolatility::try_new(annualized_decimal)
        .map(|_| ())
        .map_err(CoreInputError::into_py_input_error)
}

/// Validates an exact event schedule without returning a mutable native handle.
#[pyfunction]
fn validate_event_schedule(
    times: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
) -> PyResult<()> {
    parse_event_schedule(
        times,
        dividend_nodes,
        dividend_amounts,
        exercise_nodes,
        expiry_node,
    )
    .map(|_| ())
    .map_err(CoreInputError::into_py_input_error)
}

/// Validates exact Arrow-style ragged cash-dividend buffers without copying them.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn validate_ragged_cash_dividends_f64(
    offsets: PyReadonlyArray1<'_, i64>,
    times: PyReadonlyArray1<'_, f64>,
    amounts: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let offsets = offsets.as_slice()?;
    let times = times.as_slice()?;
    let amounts = amounts.as_slice()?;
    RaggedCashDividends::try_new(offsets, times, amounts)
        .map(|_| ())
        .map_err(CoreBatchError::into_py_batch_error)
}

fn parse_maturity_boundary(value: &str) -> PyResult<CoreMaturityBoundary> {
    match value {
        "strictly_before" => Ok(CoreMaturityBoundary::StrictlyBefore),
        "at_or_before" => Ok(CoreMaturityBoundary::AtOrBefore),
        _ => Err(BatchError::new_err(format!(
            "unsupported maturity boundary {value:?}"
        ))),
    }
}

fn parse_time_bin_closure(value: &str) -> PyResult<CoreTimeBinClosure> {
    match value {
        "left_closed_right_open" => Ok(CoreTimeBinClosure::LeftClosedRightOpen),
        "left_open_right_closed" => Ok(CoreTimeBinClosure::LeftOpenRightClosed),
        _ => Err(BatchError::new_err(format!(
            "unsupported time-bin closure {value:?}"
        ))),
    }
}

fn parse_cash_amount_scale<'a>(
    value: &str,
    spots: Option<&'a [f64]>,
) -> PyResult<CoreCashAmountScale<'a>> {
    match (value, spots) {
        ("absolute", None) => Ok(CoreCashAmountScale::Absolute),
        ("divide_by_spot", Some(values)) => Ok(CoreCashAmountScale::DivideBySpot(values)),
        ("absolute", Some(_)) => Err(BatchError::new_err(
            "absolute cash-amount scale requires spots=None",
        )),
        ("divide_by_spot", None) => Err(BatchError::new_err(
            "divide_by_spot cash-amount scale requires a float64 spot vector",
        )),
        _ => Err(BatchError::new_err(format!(
            "unsupported cash-amount scale {value:?}"
        ))),
    }
}

type NativePaddedDividendArrays<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray1<u32>>,
);

/// Derives chronological right-padded float32 cash-dividend tensors.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn pack_padded_dividends_f32<'py>(
    py: Python<'py>,
    offsets: PyReadonlyArray1<'py, i64>,
    times: PyReadonlyArray1<'py, f64>,
    amounts: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    amount_scale: &str,
    capacity: usize,
    maturity_boundary: &str,
    executor: PyRef<'py, NativeCpuExecutor>,
    spots: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<NativePaddedDividendArrays<'py>> {
    let offsets = offsets.as_slice()?;
    let times = times.as_slice()?;
    let amounts = amounts.as_slice()?;
    let maturities = maturities.as_slice()?;
    let spot_values = spots.as_ref().map(PyReadonlyArray1::as_slice).transpose()?;
    let scale = parse_cash_amount_scale(amount_scale, spot_values)?;
    let spec =
        CorePaddedSequenceSpec::try_new(capacity, parse_maturity_boundary(maturity_boundary)?)
            .map_err(CoreBatchError::into_py_batch_error)?;
    let core_executor = &executor.inner;
    let output = py
        .detach(move || {
            let schedules =
                RaggedCashDividends::try_new_with_executor(offsets, times, amounts, core_executor)?;
            core_pack_padded_dividends_f32(schedules, maturities, scale, spec, core_executor)
        })
        .map_err(CoreBatchError::into_py_batch_error)?;
    let rows = output.rows();
    let width = output.capacity();
    let (times, amounts, mask, lengths) = output.into_parts();
    let times = Array2::from_shape_vec((rows, width), times).map_err(|error| {
        PyRuntimeError::new_err(format!("internal time shape failure: {error}"))
    })?;
    let amounts = Array2::from_shape_vec((rows, width), amounts).map_err(|error| {
        PyRuntimeError::new_err(format!("internal amount shape failure: {error}"))
    })?;
    let mask = Array2::from_shape_vec((rows, width), mask).map_err(|error| {
        PyRuntimeError::new_err(format!("internal mask shape failure: {error}"))
    })?;
    Ok((
        times.into_pyarray(py),
        amounts.into_pyarray(py),
        mask.into_pyarray(py),
        lengths.into_pyarray(py),
    ))
}

/// Derives chronological fixed-width float32 cash-dividend amount bins.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn bin_dividends_f32<'py>(
    py: Python<'py>,
    offsets: PyReadonlyArray1<'py, i64>,
    times: PyReadonlyArray1<'py, f64>,
    amounts: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    amount_scale: &str,
    bins: usize,
    horizon: f64,
    closure: &str,
    maturity_boundary: &str,
    executor: PyRef<'py, NativeCpuExecutor>,
    spots: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let offsets = offsets.as_slice()?;
    let times = times.as_slice()?;
    let amounts = amounts.as_slice()?;
    let maturities = maturities.as_slice()?;
    let spot_values = spots.as_ref().map(PyReadonlyArray1::as_slice).transpose()?;
    let scale = parse_cash_amount_scale(amount_scale, spot_values)?;
    let spec = CoreDividendBinSpec::try_new(
        bins,
        horizon,
        parse_time_bin_closure(closure)?,
        parse_maturity_boundary(maturity_boundary)?,
    )
    .map_err(CoreBatchError::into_py_batch_error)?;
    let core_executor = &executor.inner;
    let output = py
        .detach(move || {
            let schedules =
                RaggedCashDividends::try_new_with_executor(offsets, times, amounts, core_executor)?;
            core_bin_dividends_f32(schedules, maturities, scale, spec, core_executor)
        })
        .map_err(CoreBatchError::into_py_batch_error)?;
    let rows = output.rows();
    let bins = output.bins();
    let amounts = Array2::from_shape_vec((rows, bins), output.into_amounts()).map_err(|error| {
        PyRuntimeError::new_err(format!("internal binned-amount shape failure: {error}"))
    })?;
    Ok(amounts.into_pyarray(py))
}

fn parse_event_schedule(
    times: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
) -> Result<EventSchedule, CoreInputError> {
    if dividend_nodes.len() != dividend_amounts.len() {
        return Err(CoreInputError::DividendColumnLengthMismatch {
            nodes: dividend_nodes.len(),
            amounts: dividend_amounts.len(),
        });
    }
    let grid = TimeGrid::try_new(times)?;
    let dividends = dividend_nodes
        .into_iter()
        .zip(dividend_amounts)
        .map(|(node, amount)| {
            let node = grid.node_id(node as usize)?;
            CashDividend::try_new(node, amount)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exercise_nodes = exercise_nodes
        .into_iter()
        .map(|node| grid.node_id(node as usize))
        .collect::<Result<Vec<_>, _>>()?;
    let expiry = grid.node_id(expiry_node as usize)?;
    EventSchedule::try_new(grid, dividends, exercise_nodes, expiry)
}

/// Validates node-aligned log discount factors.
#[pyfunction]
fn validate_log_discount_curve(times: Vec<f64>, log_discounts: Vec<f64>) -> PyResult<()> {
    let grid = TimeGrid::try_new(times).map_err(CoreInputError::into_py_input_error)?;
    LogDiscountCurve::try_new(&grid, log_discounts)
        .map(|_| ())
        .map_err(CoreInputError::into_py_input_error)
}

/// Validates node-aligned cumulative continuous proportional-dividend yields.
#[pyfunction]
fn validate_cumulative_dividend_yield_curve(
    times: Vec<f64>,
    cumulative_yields: Vec<f64>,
) -> PyResult<()> {
    let grid = TimeGrid::try_new(times).map_err(CoreInputError::into_py_input_error)?;
    CumulativeDividendYieldCurve::try_new(&grid, cumulative_yields)
        .map(|_| ())
        .map_err(CoreInputError::into_py_input_error)
}

#[allow(clippy::too_many_arguments)]
fn parse_gbm_plan(
    schedule_times: Vec<f64>,
    curve_times: Vec<f64>,
    log_discounts: Vec<f64>,
    dividend_yield_times: Vec<f64>,
    cumulative_dividend_yields: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
    annualized_volatility: f64,
    dividend_policy: &str,
) -> PyResult<GbmPathPlan> {
    parse_gbm_model(
        schedule_times,
        curve_times,
        log_discounts,
        dividend_yield_times,
        cumulative_dividend_yields,
        dividend_nodes,
        dividend_amounts,
        exercise_nodes,
        expiry_node,
        annualized_volatility,
        dividend_policy,
    )
    .map(|(plan, _)| plan)
}

#[allow(clippy::too_many_arguments)]
fn parse_gbm_model(
    schedule_times: Vec<f64>,
    curve_times: Vec<f64>,
    log_discounts: Vec<f64>,
    dividend_yield_times: Vec<f64>,
    cumulative_dividend_yields: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
    annualized_volatility: f64,
    dividend_policy: &str,
) -> PyResult<(GbmPathPlan, LogDiscountCurve)> {
    let schedule = parse_event_schedule(
        schedule_times,
        dividend_nodes,
        dividend_amounts,
        exercise_nodes,
        expiry_node,
    )
    .map_err(CoreInputError::into_py_input_error)?;
    let curve_grid = TimeGrid::try_new(curve_times).map_err(CoreInputError::into_py_input_error)?;
    let curve = LogDiscountCurve::try_new(&curve_grid, log_discounts)
        .map_err(CoreInputError::into_py_input_error)?;
    let dividend_yield_grid =
        TimeGrid::try_new(dividend_yield_times).map_err(CoreInputError::into_py_input_error)?;
    let dividend_yield_curve =
        CumulativeDividendYieldCurve::try_new(&dividend_yield_grid, cumulative_dividend_yields)
            .map_err(CoreInputError::into_py_input_error)?;
    let volatility = ConstantVolatility::try_new(annualized_volatility)
        .map_err(CoreInputError::into_py_input_error)?;
    let policy = match dividend_policy {
        "liquidator" => DividendPolicy::Liquidator,
        "survivor" => DividendPolicy::Survivor,
        observed => {
            return Err(InputError::new_err(format!(
                "unsupported dividend policy {observed:?}"
            )));
        }
    };
    let plan = GbmPathPlan::try_new(&schedule, &curve, &dividend_yield_curve, volatility, policy)
        .map_err(CoreSimulationError::into_py_simulation_error)?;
    Ok((plan, curve))
}

fn simulated_into_array(
    py: Python<'_>,
    simulated: SimulatedPaths,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let rows = simulated.observation_nodes().len();
    let paths = simulated.path_count();
    let values = simulated.into_values();
    let array = Array2::from_shape_vec((rows, paths), values)
        .map_err(|error| SimulationError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn simulate_gbm_f64<'py>(
    py: Python<'py>,
    initial_spot: f64,
    schedule_times: Vec<f64>,
    curve_times: Vec<f64>,
    log_discounts: Vec<f64>,
    dividend_yield_times: Vec<f64>,
    cumulative_dividend_yields: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
    annualized_volatility: f64,
    dividend_policy: &str,
    executor: PyRef<'py, NativeCpuExecutor>,
    normals: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let plan = parse_gbm_plan(
        schedule_times,
        curve_times,
        log_discounts,
        dividend_yield_times,
        cumulative_dividend_yields,
        dividend_nodes,
        dividend_amounts,
        exercise_nodes,
        expiry_node,
        annualized_volatility,
        dividend_policy,
    )?;

    let shape = normals.shape();
    if shape[0] != plan.intervals() {
        return Err(CoreSimulationError::BufferLengthMismatch {
            buffer: "normal interval-row",
            expected: plan.intervals(),
            observed: shape[0],
        }
        .into_py_simulation_error());
    }
    let paths = shape[1];
    let normal_values = normals.as_slice()?;
    let core_executor = &executor.inner;
    let simulated = py
        .detach(move || plan.simulate_with(core_executor, initial_spot, paths, normal_values))
        .map_err(CoreSimulationError::into_py_simulation_error)?;
    simulated_into_array(py, simulated)
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn simulate_gbm_seeded_f64<'py>(
    py: Python<'py>,
    initial_spot: f64,
    schedule_times: Vec<f64>,
    curve_times: Vec<f64>,
    log_discounts: Vec<f64>,
    dividend_yield_times: Vec<f64>,
    cumulative_dividend_yields: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
    annualized_volatility: f64,
    dividend_policy: &str,
    executor: PyRef<'py, NativeCpuExecutor>,
    paths: usize,
    seed: u64,
    domain: u32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let plan = parse_gbm_plan(
        schedule_times,
        curve_times,
        log_discounts,
        dividend_yield_times,
        cumulative_dividend_yields,
        dividend_nodes,
        dividend_amounts,
        exercise_nodes,
        expiry_node,
        annualized_volatility,
        dividend_policy,
    )?;
    let stream = CoreRandomStream::new(seed, domain);
    let core_executor = &executor.inner;
    let simulated = py
        .detach(move || plan.simulate_seeded_with(core_executor, initial_spot, paths, stream))
        .map_err(CoreSimulationError::into_py_simulation_error)?;
    simulated_into_array(py, simulated)
}

type NativeExerciseDiagnostic = (
    u32,
    &'static str,
    Option<bool>,
    Option<f64>,
    Option<u32>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);

type NativeLsmResult = (
    f64,
    u32,
    Option<f64>,
    Option<f64>,
    Vec<u32>,
    Vec<u32>,
    Vec<NativeExerciseDiagnostic>,
);

fn native_lsm_result(run: CoreLsmRun) -> NativeLsmResult {
    let (policy, evaluation) = run.into_parts();
    let diagnostics = policy
        .diagnostics()
        .iter()
        .map(|diagnostic| match diagnostic.fit() {
            CoreExerciseFitDiagnostics::OriginDecision {
                exercise,
                estimated_continuation,
            } => (
                diagnostic.node().get(),
                "origin_decision",
                Some(exercise),
                Some(estimated_continuation),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            CoreExerciseFitDiagnostics::NoInTheMoneyPaths => (
                diagnostic.node().get(),
                "no_in_the_money_paths",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            CoreExerciseFitDiagnostics::Regression {
                in_the_money_paths,
                diagnostics,
            } => (
                diagnostic.node().get(),
                "regression",
                None,
                None,
                Some(in_the_money_paths),
                Some(diagnostics.observations()),
                Some(diagnostics.columns()),
                Some(diagnostics.rank()),
                Some(diagnostics.condition_number()),
                Some(diagnostics.predictor_mean()),
                Some(diagnostics.predictor_scale()),
            ),
        })
        .collect();
    (
        evaluation.price(),
        evaluation.paths(),
        evaluation.sample_standard_deviation(),
        evaluation.standard_error(),
        policy
            .observation_nodes()
            .iter()
            .map(|node| node.get())
            .collect(),
        evaluation.exercise_counts().to_vec(),
        diagnostics,
    )
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_lsm_call_seeded_f64<'py>(
    py: Python<'py>,
    initial_spot: f64,
    strike: f64,
    schedule_times: Vec<f64>,
    curve_times: Vec<f64>,
    log_discounts: Vec<f64>,
    dividend_yield_times: Vec<f64>,
    cumulative_dividend_yields: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
    annualized_volatility: f64,
    dividend_policy: &str,
    executor: PyRef<'py, NativeCpuExecutor>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain: u32,
    evaluation_seed: u64,
    evaluation_domain: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
) -> PyResult<NativeLsmResult> {
    let (plan, curve) = parse_gbm_model(
        schedule_times,
        curve_times,
        log_discounts,
        dividend_yield_times,
        cumulative_dividend_yields,
        dividend_nodes,
        dividend_amounts,
        exercise_nodes,
        expiry_node,
        annualized_volatility,
        dividend_policy,
    )?;
    let degree = CorePolynomialDegree::try_new(polynomial_degree)
        .map_err(|error| LsmError::new_err(error.to_string()))?;
    let regression = CorePolynomialRegressionSpec::try_new(degree, rank_tolerance, condition_limit)
        .map_err(|error| LsmError::new_err(error.to_string()))?;
    let tie_rule = match tie_rule {
        "strictly_greater" => CoreExerciseTieRule::StrictlyGreater,
        "greater_or_equal" => CoreExerciseTieRule::GreaterOrEqual,
        observed => {
            return Err(InputError::new_err(format!(
                "unsupported exercise tie rule {observed:?}"
            )));
        }
    };
    let estimator = CoreLsmCallEstimator::try_new(strike, regression, tie_rule)
        .map_err(CoreLsmError::into_py_lsm_error)?;
    let training_stream = CoreRandomStream::new(training_seed, training_domain);
    let evaluation_stream = CoreRandomStream::new(evaluation_seed, evaluation_domain);
    let core_executor = &executor.inner;
    let run = py
        .detach(move || {
            estimator.price_seeded(
                core_executor,
                &plan,
                &curve,
                initial_spot,
                training_paths,
                evaluation_paths,
                training_stream,
                evaluation_stream,
            )
        })
        .map_err(CoreLsmError::into_py_lsm_error)?;
    Ok(native_lsm_result(run))
}

/// Private native module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCpuExecutor>()?;
    module.add("InputError", module.py().get_type::<InputError>())?;
    module.add("BatchError", module.py().get_type::<BatchError>())?;
    module.add("ExecutionError", module.py().get_type::<ExecutionError>())?;
    module.add("SimulationError", module.py().get_type::<SimulationError>())?;
    module.add("LsmError", module.py().get_type::<LsmError>())?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(bin_dividends_f32, module)?)?;
    module.add_function(wrap_pyfunction!(pack_padded_dividends_f32, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_gbm_f64, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_gbm_seeded_f64, module)?)?;
    module.add_function(wrap_pyfunction!(price_lsm_call_seeded_f64, module)?)?;
    module.add_function(wrap_pyfunction!(validate_cash_dividend_amount, module)?)?;
    module.add_function(wrap_pyfunction!(validate_constant_volatility, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_cumulative_dividend_yield_curve,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_event_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_log_discount_curve, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_ragged_cash_dividends_f64,
        module
    )?)?;
    Ok(())
}
