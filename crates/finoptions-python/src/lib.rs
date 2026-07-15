//! Private Python extension. The stable public surface lives in `python/finoptions`.

use finoptions_batch::{
    BatchError as CoreBatchError, CashAmountScale as CoreCashAmountScale,
    DividendBinSpec as CoreDividendBinSpec, MaturityBoundary as CoreMaturityBoundary,
    PaddedSequenceSpec as CorePaddedSequenceSpec, RaggedCashDividends, RaggedExerciseTimes,
    TimeBinClosure as CoreTimeBinClosure, bin_dividends_f32 as core_bin_dividends_f32,
    pack_padded_dividends_f32 as core_pack_padded_dividends_f32,
};
use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    InputError as CoreInputError, LogDiscountCurve, TimeGrid,
};
use finoptions_lsm::{
    BatchDividendModel as CoreBatchDividendModel, BatchLsmCallInputs as CoreBatchLsmCallInputs,
    BatchLsmCallMethod as CoreBatchLsmCallMethod, BatchLsmCallResult as CoreBatchLsmCallResult,
    BatchLsmError as CoreBatchLsmError, BatchRandomStreams as CoreBatchRandomStreams,
    CashDividendExerciseMesh as CoreCashDividendExerciseMesh,
    EscrowedVolatilityConvention as CoreEscrowedVolatilityConvention,
    ExerciseFitDiagnostics as CoreExerciseFitDiagnostics, ExerciseTieRule as CoreExerciseTieRule,
    ExplicitExerciseMesh as CoreExplicitExerciseMesh,
    GroupedBatchLsmCallInputs as CoreGroupedBatchLsmCallInputs,
    InsufficientExerciseSupportPolicy as CoreInsufficientExerciseSupportPolicy,
    LsmCallEstimator as CoreLsmCallEstimator, LsmError as CoreLsmError, LsmRun as CoreLsmRun,
    OriginExercise as CoreOriginExercise, PolynomialDegree as CorePolynomialDegree,
    PolynomialRegressionSpec as CorePolynomialRegressionSpec,
    UniformExerciseMesh as CoreUniformExerciseMesh,
};
use finoptions_quadrature::{
    EventAlignedQuadrature as CoreEventAlignedQuadrature,
    EventAlignedQuadratureSpec as CoreEventAlignedQuadratureSpec,
    QuadratureError as CoreQuadratureError,
};
use finoptions_runtime::{
    CpuExecutor as CoreCpuExecutor, CpuMathBackend as CoreCpuMathBackend,
    ExecutionError as CoreExecutionError,
};
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
create_exception!(_native, QuadratureError, PyRuntimeError);

/// Dedicated native CPU worker pool owned by the public Python execution specification.
#[pyclass(name = "CpuExecutor", frozen, module = "finoptions._native")]
struct NativeCpuExecutor {
    inner: CoreCpuExecutor,
}

/// Reusable event-aligned direct-integration rule with explicit numerical controls.
#[pyclass(name = "EventAlignedQuadrature", frozen, module = "finoptions._native")]
struct NativeEventAlignedQuadrature {
    inner: CoreEventAlignedQuadrature,
}

#[pymethods]
impl NativeEventAlignedQuadrature {
    #[new]
    fn new(upper_stock: f64, stock_points: usize, normal_order: usize) -> PyResult<Self> {
        let spec = CoreEventAlignedQuadratureSpec::try_new(upper_stock, stock_points, normal_order)
            .map_err(CoreQuadratureError::into_py_quadrature_error)?;
        CoreEventAlignedQuadrature::try_new(spec)
            .map(|inner| Self { inner })
            .map_err(CoreQuadratureError::into_py_quadrature_error)
    }

    #[getter]
    fn upper_stock(&self) -> f64 {
        self.inner.spec().upper_stock()
    }

    #[getter]
    fn stock_points(&self) -> usize {
        self.inner.spec().stock_points()
    }

    #[getter]
    fn normal_order(&self) -> usize {
        self.inner.spec().normal_order()
    }
}

#[pymethods]
impl NativeCpuExecutor {
    #[new]
    #[pyo3(signature = (threads, math_backend = "rust_std_f64"))]
    fn new(threads: usize, math_backend: &str) -> PyResult<Self> {
        CoreCpuExecutor::try_new_with_math_backend(threads, parse_cpu_math_backend(math_backend)?)
            .map(|inner| Self { inner })
            .map_err(CoreExecutionError::into_py_execution_error)
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }

    #[getter]
    fn math_backend(&self) -> &'static str {
        self.inner.math_backend().as_str()
    }
}

fn parse_cpu_math_backend(value: &str) -> PyResult<CoreCpuMathBackend> {
    match value {
        "rust_std_f64" => Ok(CoreCpuMathBackend::RustStdF64),
        "apple_accelerate" => Ok(CoreCpuMathBackend::AppleAccelerate),
        _ => Err(InputError::new_err(format!(
            "CPU math backend must be 'rust_std_f64' or 'apple_accelerate'; observed {value:?}"
        ))),
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

trait IntoPyBatchLsmDividendError {
    fn into_py_batch_lsm_dividend_error(self) -> PyErr;
}

impl IntoPyBatchLsmDividendError for CoreBatchError {
    fn into_py_batch_lsm_dividend_error(self) -> PyErr {
        let message = self.to_string();
        match self {
            CoreBatchError::ParallelFailureReplayMismatch { .. } => LsmError::new_err(message),
            _ => InputError::new_err(message),
        }
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
            | CoreSimulationError::CpuMath { .. }
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

trait IntoPyQuadratureError {
    fn into_py_quadrature_error(self) -> PyErr;
}

impl IntoPyQuadratureError for CoreQuadratureError {
    fn into_py_quadrature_error(self) -> PyErr {
        let message = self.to_string();
        match self {
            CoreQuadratureError::FinancialInput(_)
            | CoreQuadratureError::DiscountTimeGridMismatch
            | CoreQuadratureError::DividendYieldTimeGridMismatch
            | CoreQuadratureError::InvalidInitialSpot { .. }
            | CoreQuadratureError::InvalidStrike { .. }
            | CoreQuadratureError::InvalidUpperStockBoundary { .. }
            | CoreQuadratureError::UpperStockBoundaryTooLow { .. }
            | CoreQuadratureError::TooFewStockPoints { .. }
            | CoreQuadratureError::TooManyStockPoints { .. }
            | CoreQuadratureError::UnsupportedQuadratureOrder { .. } => {
                InputError::new_err(message)
            }
            _ => QuadratureError::new_err(message),
        }
    }
}

impl IntoPyLsmError for CoreLsmError {
    fn into_py_lsm_error(self) -> PyErr {
        LsmError::new_err(self.to_string())
    }
}

trait IntoPyBatchLsmError {
    fn into_py_batch_lsm_error(self) -> PyErr;
}

impl IntoPyBatchLsmError for CoreBatchLsmError {
    fn into_py_batch_lsm_error(self) -> PyErr {
        let message = self.to_string();
        match self {
            CoreBatchLsmError::EmptyBatch
            | CoreBatchLsmError::RowCountMismatch { .. }
            | CoreBatchLsmError::GroupOffsetCountMismatch { .. }
            | CoreBatchLsmError::GroupOffsetsDoNotStartAtZero { .. }
            | CoreBatchLsmError::InvalidGroupRange { .. }
            | CoreBatchLsmError::GroupOffsetEndMismatch { .. }
            | CoreBatchLsmError::InvalidRowValue { .. }
            | CoreBatchLsmError::EmptyBaseIntervals
            | CoreBatchLsmError::CashDividendExerciseMeshRequiresCashDividends
            | CoreBatchLsmError::NegativeRateForCashDividendExerciseMesh { .. }
            | CoreBatchLsmError::TerminalCashForCashDividendExerciseMesh { .. }
            | CoreBatchLsmError::ExplicitExerciseAtOrAfterMaturity { .. }
            | CoreBatchLsmError::InsufficientEvaluationPaths { .. }
            | CoreBatchLsmError::PathCountOutsideDomain { .. }
            | CoreBatchLsmError::RandomDomainRangeOverflow { .. }
            | CoreBatchLsmError::RandomDomainRangeCollision { .. }
            | CoreBatchLsmError::CashDividendAfterMaturity { .. }
            | CoreBatchLsmError::CashDividendAtExcludedMaturity { .. }
            | CoreBatchLsmError::UnsupportedCashMaturityBoundary
            | CoreBatchLsmError::DuplicateCashDividendTime { .. }
            | CoreBatchLsmError::NonFiniteEscrowedPrepaidCash { .. }
            | CoreBatchLsmError::NonPositiveEscrowedPrepaidForward { .. }
            | CoreBatchLsmError::InvalidEscrowedPrepaidForwardVolatility { .. }
            | CoreBatchLsmError::ContractMeshShapeOverflow { .. }
            | CoreBatchLsmError::ExplicitContractMeshShapeOverflow { .. }
            | CoreBatchLsmError::FinancialInputAtRow { .. }
            | CoreBatchLsmError::SimulationPlanAtRow { .. } => InputError::new_err(message),
            _ => LsmError::new_err(message),
        }
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

/// Validates exact Arrow-style ragged exercise-time buffers without copying them.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn validate_ragged_exercise_times_f64(
    offsets: PyReadonlyArray1<'_, i64>,
    times: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let offsets = offsets.as_slice()?;
    let times = times.as_slice()?;
    RaggedExerciseTimes::try_new(offsets, times)
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

fn parse_dividend_policy(value: &str) -> PyResult<DividendPolicy> {
    match value {
        "liquidator" => Ok(DividendPolicy::Liquidator),
        "survivor" => Ok(DividendPolicy::Survivor),
        observed => Err(InputError::new_err(format!(
            "unsupported dividend policy {observed:?}"
        ))),
    }
}

fn parse_exercise_tie_rule(value: &str) -> PyResult<CoreExerciseTieRule> {
    match value {
        "strictly_greater" => Ok(CoreExerciseTieRule::StrictlyGreater),
        "greater_or_equal" => Ok(CoreExerciseTieRule::GreaterOrEqual),
        observed => Err(InputError::new_err(format!(
            "unsupported exercise tie rule {observed:?}"
        ))),
    }
}

fn parse_insufficient_exercise_support(
    value: &str,
) -> PyResult<CoreInsufficientExerciseSupportPolicy> {
    match value {
        "fail" => Ok(CoreInsufficientExerciseSupportPolicy::Fail),
        "continue" => Ok(CoreInsufficientExerciseSupportPolicy::Continue),
        observed => Err(InputError::new_err(format!(
            "unsupported insufficient exercise support policy {observed:?}"
        ))),
    }
}

fn parse_origin_exercise(value: &str) -> PyResult<CoreOriginExercise> {
    match value {
        "excluded" => Ok(CoreOriginExercise::Excluded),
        "included" => Ok(CoreOriginExercise::Included),
        observed => Err(InputError::new_err(format!(
            "unsupported origin exercise convention {observed:?}"
        ))),
    }
}

fn parse_escrowed_volatility_convention(value: &str) -> PyResult<CoreEscrowedVolatilityConvention> {
    match value {
        "prepaid_forward" => Ok(CoreEscrowedVolatilityConvention::PrepaidForward),
        "spot_matched_at_origin" => Ok(CoreEscrowedVolatilityConvention::SpotMatchedAtOrigin),
        observed => Err(InputError::new_err(format!(
            "unsupported escrowed volatility convention {observed:?}"
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
    let (schedule, curve, dividend_yield_curve, volatility, policy) = parse_gbm_financial_inputs(
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
    let plan = GbmPathPlan::try_new(&schedule, &curve, &dividend_yield_curve, volatility, policy)
        .map_err(CoreSimulationError::into_py_simulation_error)?;
    Ok((plan, curve))
}

type CoreGbmFinancialInputs = (
    EventSchedule,
    LogDiscountCurve,
    CumulativeDividendYieldCurve,
    ConstantVolatility,
    DividendPolicy,
);

#[allow(clippy::too_many_arguments)]
fn parse_gbm_financial_inputs(
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
) -> PyResult<CoreGbmFinancialInputs> {
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
    let policy = parse_dividend_policy(dividend_policy)?;
    Ok((schedule, curve, dividend_yield_curve, volatility, policy))
}

/// Prices one exact event schedule with the reusable deterministic quadrature method.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_event_aligned_call_f64(
    py: Python<'_>,
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
    executor: PyRef<'_, NativeCpuExecutor>,
    method: PyRef<'_, NativeEventAlignedQuadrature>,
) -> PyResult<(f64, f64)> {
    let (schedule, curve, dividend_yield_curve, volatility, policy) = parse_gbm_financial_inputs(
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
    let core_method = &method.inner;
    let core_executor = &executor.inner;
    let result = py
        .detach(move || {
            core_method.price_call(
                core_executor,
                initial_spot,
                strike,
                &schedule,
                &curve,
                &dividend_yield_curve,
                volatility,
                policy,
            )
        })
        .map_err(CoreQuadratureError::into_py_quadrature_error)?;
    Ok((result.price(), result.maximum_upper_boundary_gap()))
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
            CoreExerciseFitDiagnostics::InsufficientExerciseSupport {
                in_the_money_paths,
                required_paths,
            } => (
                diagnostic.node().get(),
                "insufficient_exercise_support",
                None,
                None,
                Some(in_the_money_paths),
                None,
                Some(required_paths as usize),
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
    insufficient_exercise_support: &str,
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
    let tie_rule = parse_exercise_tie_rule(tie_rule)?;
    let estimator = CoreLsmCallEstimator::try_new(
        strike,
        regression,
        tie_rule,
        parse_insufficient_exercise_support(insufficient_exercise_support)?,
    )
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

#[derive(Clone, Copy)]
struct NativeBatchColumns<'a> {
    initial_spots: &'a [f64],
    strikes: &'a [f64],
    maturities: &'a [f64],
    constant_continuously_compounded_rates: &'a [f64],
    constant_volatilities: &'a [f64],
}

impl NativeBatchColumns<'_> {
    fn price(
        self,
        dividends: CoreBatchDividendModel<'_>,
        method: CoreBatchLsmCallMethod,
        executor: &CoreCpuExecutor,
    ) -> Result<CoreBatchLsmCallResult, CoreBatchLsmError> {
        CoreBatchLsmCallInputs::try_new(
            self.initial_spots,
            self.strikes,
            self.maturities,
            self.constant_continuously_compounded_rates,
            self.constant_volatilities,
            dividends,
        )?
        .price(method, executor)
    }
}

#[derive(Clone, Copy)]
struct NativeGroupedBatchColumns<'a> {
    group_offsets: &'a [u32],
    initial_spots: &'a [f64],
    strikes: &'a [f64],
    maturities: &'a [f64],
    constant_continuously_compounded_rates: &'a [f64],
    constant_volatilities: &'a [f64],
}

impl NativeGroupedBatchColumns<'_> {
    fn price(
        self,
        dividends: CoreBatchDividendModel<'_>,
        method: CoreBatchLsmCallMethod,
        executor: &CoreCpuExecutor,
    ) -> Result<CoreBatchLsmCallResult, CoreBatchLsmError> {
        CoreGroupedBatchLsmCallInputs::try_new(
            self.group_offsets,
            self.initial_spots,
            self.strikes,
            self.maturities,
            self.constant_continuously_compounded_rates,
            self.constant_volatilities,
            dividends,
        )?
        .price(method, executor)
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_batch_lsm_method<'a>(
    exercise_mesh: &str,
    base_intervals: Option<u32>,
    origin_exercise: Option<&str>,
    explicit_exercise_offsets: Option<&'a [i64]>,
    explicit_exercise_times: Option<&'a [f64]>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain_start: u32,
    evaluation_seed: u64,
    evaluation_domain_start: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
    insufficient_exercise_support: &str,
) -> PyResult<CoreBatchLsmCallMethod<'a>> {
    let degree = CorePolynomialDegree::try_new(polynomial_degree)
        .map_err(|error| LsmError::new_err(error.to_string()))?;
    let regression = CorePolynomialRegressionSpec::try_new(degree, rank_tolerance, condition_limit)
        .map_err(|error| LsmError::new_err(error.to_string()))?;
    let tie_rule = parse_exercise_tie_rule(tie_rule)?;
    let insufficient_exercise_support =
        parse_insufficient_exercise_support(insufficient_exercise_support)?;
    let random_streams = CoreBatchRandomStreams::new(
        training_seed,
        evaluation_seed,
        training_domain_start,
        evaluation_domain_start,
    );
    let method = match exercise_mesh {
        "uniform" => {
            if explicit_exercise_offsets.is_some() || explicit_exercise_times.is_some() {
                return Err(InputError::new_err(
                    "uniform exercise mesh forbids explicit exercise buffers",
                ));
            }
            let base_intervals = base_intervals.ok_or_else(|| {
                InputError::new_err("uniform exercise mesh requires base_intervals")
            })?;
            let origin_exercise = origin_exercise.ok_or_else(|| {
                InputError::new_err("uniform exercise mesh requires origin_exercise")
            })?;
            let mesh = CoreUniformExerciseMesh::try_new(
                base_intervals,
                parse_origin_exercise(origin_exercise)?,
            )
            .map_err(CoreBatchLsmError::into_py_batch_lsm_error)?;
            CoreBatchLsmCallMethod::try_new(
                mesh,
                training_paths,
                evaluation_paths,
                regression,
                tie_rule,
                insufficient_exercise_support,
                random_streams,
            )
        }
        "cash_dividend_dates" => {
            if base_intervals.is_some()
                || origin_exercise.is_some()
                || explicit_exercise_offsets.is_some()
                || explicit_exercise_times.is_some()
            {
                return Err(InputError::new_err(
                    "cash-dividend-date exercise mesh forbids uniform and explicit mesh fields",
                ));
            }
            CoreBatchLsmCallMethod::try_new_cash_dividend_dates(
                CoreCashDividendExerciseMesh::new(),
                training_paths,
                evaluation_paths,
                regression,
                tie_rule,
                insufficient_exercise_support,
                random_streams,
            )
        }
        "explicit_times" => {
            if base_intervals.is_some() {
                return Err(InputError::new_err(
                    "explicit exercise mesh forbids base_intervals",
                ));
            }
            let origin_exercise = origin_exercise.ok_or_else(|| {
                InputError::new_err("explicit exercise mesh requires origin_exercise")
            })?;
            let offsets = explicit_exercise_offsets.ok_or_else(|| {
                InputError::new_err("explicit exercise mesh requires exercise offsets")
            })?;
            let times = explicit_exercise_times.ok_or_else(|| {
                InputError::new_err("explicit exercise mesh requires exercise times")
            })?;
            let schedules = RaggedExerciseTimes::try_new(offsets, times)
                .map_err(CoreBatchError::into_py_batch_error)?;
            CoreBatchLsmCallMethod::try_new_explicit(
                CoreExplicitExerciseMesh::new(schedules, parse_origin_exercise(origin_exercise)?),
                training_paths,
                evaluation_paths,
                regression,
                tie_rule,
                insufficient_exercise_support,
                random_streams,
            )
        }
        observed => {
            return Err(InputError::new_err(format!(
                "unsupported batched LSM exercise mesh {observed:?}"
            )));
        }
    };
    method.map_err(CoreBatchLsmError::into_py_batch_lsm_error)
}

type NativeBatchLsmResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
);

fn reserve_batch_output<T>(values: &mut Vec<T>, rows: usize, field: &'static str) -> PyResult<()> {
    values.try_reserve_exact(rows).map_err(|_| {
        LsmError::new_err(format!(
            "could not allocate native batched LSM output {field} for {rows} rows"
        ))
    })
}

#[allow(clippy::too_many_lines)]
fn native_batch_lsm_result(
    py: Python<'_>,
    result: CoreBatchLsmCallResult,
) -> PyResult<NativeBatchLsmResult<'_>> {
    let rows = result.len();
    let mut prices = Vec::new();
    let mut sample_standard_deviations = Vec::new();
    let mut standard_errors = Vec::new();
    let mut origin_decisions_fitted = Vec::new();
    let mut origin_exercised = Vec::new();
    let mut regression_nodes = Vec::new();
    let mut no_training_support_nodes = Vec::new();
    let mut insufficient_exercise_support_nodes = Vec::new();
    let mut minimum_in_the_money_training_paths = Vec::new();
    let mut maximum_regression_condition_numbers = Vec::new();
    let mut early_exercise_paths = Vec::new();
    let mut expiry_paths = Vec::new();
    reserve_batch_output(&mut prices, rows, "prices")?;
    reserve_batch_output(
        &mut sample_standard_deviations,
        rows,
        "sample_standard_deviations",
    )?;
    reserve_batch_output(&mut standard_errors, rows, "standard_errors")?;
    reserve_batch_output(
        &mut origin_decisions_fitted,
        rows,
        "origin_decisions_fitted",
    )?;
    reserve_batch_output(&mut origin_exercised, rows, "origin_exercised")?;
    reserve_batch_output(&mut regression_nodes, rows, "regression_nodes")?;
    reserve_batch_output(
        &mut no_training_support_nodes,
        rows,
        "no_training_support_nodes",
    )?;
    reserve_batch_output(
        &mut insufficient_exercise_support_nodes,
        rows,
        "insufficient_exercise_support_nodes",
    )?;
    reserve_batch_output(
        &mut minimum_in_the_money_training_paths,
        rows,
        "minimum_in_the_money_training_paths",
    )?;
    reserve_batch_output(
        &mut maximum_regression_condition_numbers,
        rows,
        "maximum_regression_condition_numbers",
    )?;
    reserve_batch_output(&mut early_exercise_paths, rows, "early_exercise_paths")?;
    reserve_batch_output(&mut expiry_paths, rows, "expiry_paths")?;

    for row in result.into_rows() {
        prices.push(row.price());
        sample_standard_deviations.push(row.sample_standard_deviation());
        standard_errors.push(row.standard_error());
        origin_decisions_fitted.push(row.origin_decision_fitted());
        origin_exercised.push(row.origin_exercised());
        regression_nodes.push(row.regression_nodes());
        no_training_support_nodes.push(row.no_training_support_nodes());
        insufficient_exercise_support_nodes.push(row.insufficient_exercise_support_nodes());
        minimum_in_the_money_training_paths.push(row.minimum_in_the_money_training_paths());
        maximum_regression_condition_numbers.push(row.maximum_regression_condition_number());
        early_exercise_paths.push(row.early_exercise_paths());
        expiry_paths.push(row.expiry_paths());
    }

    Ok((
        prices.into_pyarray(py),
        sample_standard_deviations.into_pyarray(py),
        standard_errors.into_pyarray(py),
        origin_decisions_fitted.into_pyarray(py),
        origin_exercised.into_pyarray(py),
        regression_nodes.into_pyarray(py),
        no_training_support_nodes.into_pyarray(py),
        insufficient_exercise_support_nodes.into_pyarray(py),
        minimum_in_the_money_training_paths.into_pyarray(py),
        maximum_regression_condition_numbers.into_pyarray(py),
        early_exercise_paths.into_pyarray(py),
        expiry_paths.into_pyarray(py),
    ))
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_lsm_call_batch_continuous_yield_f64<'py>(
    py: Python<'py>,
    initial_spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    constant_continuously_compounded_rates: PyReadonlyArray1<'py, f64>,
    constant_volatilities: PyReadonlyArray1<'py, f64>,
    annualized_dividend_yields: PyReadonlyArray1<'py, f64>,
    exercise_mesh: &str,
    base_intervals: Option<u32>,
    origin_exercise: Option<&str>,
    explicit_exercise_offsets: Option<PyReadonlyArray1<'py, i64>>,
    explicit_exercise_times: Option<PyReadonlyArray1<'py, f64>>,
    executor: PyRef<'py, NativeCpuExecutor>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain_start: u32,
    evaluation_seed: u64,
    evaluation_domain_start: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
    insufficient_exercise_support: &str,
) -> PyResult<NativeBatchLsmResult<'py>> {
    let columns = NativeBatchColumns {
        initial_spots: initial_spots.as_slice()?,
        strikes: strikes.as_slice()?,
        maturities: maturities.as_slice()?,
        constant_continuously_compounded_rates: constant_continuously_compounded_rates
            .as_slice()?,
        constant_volatilities: constant_volatilities.as_slice()?,
    };
    let annualized_dividend_yields = annualized_dividend_yields.as_slice()?;
    let explicit_exercise_offsets = explicit_exercise_offsets
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let explicit_exercise_times = explicit_exercise_times
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let method = parse_batch_lsm_method(
        exercise_mesh,
        base_intervals,
        origin_exercise,
        explicit_exercise_offsets,
        explicit_exercise_times,
        training_paths,
        evaluation_paths,
        training_seed,
        training_domain_start,
        evaluation_seed,
        evaluation_domain_start,
        polynomial_degree,
        rank_tolerance,
        condition_limit,
        tie_rule,
        insufficient_exercise_support,
    )?;
    let core_executor = &executor.inner;
    let result = py
        .detach(move || {
            columns.price(
                CoreBatchDividendModel::ContinuousYield {
                    annualized_decimals: annualized_dividend_yields,
                },
                method,
                core_executor,
            )
        })
        .map_err(CoreBatchLsmError::into_py_batch_lsm_error)?;
    native_batch_lsm_result(py, result)
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_lsm_call_batch_cash_dividends_f64<'py>(
    py: Python<'py>,
    initial_spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    constant_continuously_compounded_rates: PyReadonlyArray1<'py, f64>,
    constant_volatilities: PyReadonlyArray1<'py, f64>,
    dividend_offsets: PyReadonlyArray1<'py, i64>,
    dividend_times: PyReadonlyArray1<'py, f64>,
    dividend_amounts: PyReadonlyArray1<'py, f64>,
    dividend_policy: &str,
    maturity_boundary: &str,
    exercise_mesh: &str,
    base_intervals: Option<u32>,
    origin_exercise: Option<&str>,
    explicit_exercise_offsets: Option<PyReadonlyArray1<'py, i64>>,
    explicit_exercise_times: Option<PyReadonlyArray1<'py, f64>>,
    executor: PyRef<'py, NativeCpuExecutor>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain_start: u32,
    evaluation_seed: u64,
    evaluation_domain_start: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
    insufficient_exercise_support: &str,
) -> PyResult<NativeBatchLsmResult<'py>> {
    let columns = NativeBatchColumns {
        initial_spots: initial_spots.as_slice()?,
        strikes: strikes.as_slice()?,
        maturities: maturities.as_slice()?,
        constant_continuously_compounded_rates: constant_continuously_compounded_rates
            .as_slice()?,
        constant_volatilities: constant_volatilities.as_slice()?,
    };
    let dividend_offsets = dividend_offsets.as_slice()?;
    let dividend_times = dividend_times.as_slice()?;
    let dividend_amounts = dividend_amounts.as_slice()?;
    let dividend_policy = parse_dividend_policy(dividend_policy)?;
    let maturity_boundary = parse_maturity_boundary(maturity_boundary)?;
    let explicit_exercise_offsets = explicit_exercise_offsets
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let explicit_exercise_times = explicit_exercise_times
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let method = parse_batch_lsm_method(
        exercise_mesh,
        base_intervals,
        origin_exercise,
        explicit_exercise_offsets,
        explicit_exercise_times,
        training_paths,
        evaluation_paths,
        training_seed,
        training_domain_start,
        evaluation_seed,
        evaluation_domain_start,
        polynomial_degree,
        rank_tolerance,
        condition_limit,
        tie_rule,
        insufficient_exercise_support,
    )?;
    let core_executor = &executor.inner;
    let result = py.detach(move || -> PyResult<CoreBatchLsmCallResult> {
        let schedules = RaggedCashDividends::try_new_with_executor(
            dividend_offsets,
            dividend_times,
            dividend_amounts,
            core_executor,
        )
        .map_err(CoreBatchError::into_py_batch_lsm_dividend_error)?;
        columns
            .price(
                CoreBatchDividendModel::CashDividends {
                    schedules,
                    policy: dividend_policy,
                    maturity_boundary,
                },
                method,
                core_executor,
            )
            .map_err(CoreBatchLsmError::into_py_batch_lsm_error)
    })?;
    native_batch_lsm_result(py, result)
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_lsm_call_batch_escrowed_cash_dividends_f64<'py>(
    py: Python<'py>,
    initial_spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    constant_continuously_compounded_rates: PyReadonlyArray1<'py, f64>,
    constant_volatilities: PyReadonlyArray1<'py, f64>,
    dividend_offsets: PyReadonlyArray1<'py, i64>,
    dividend_times: PyReadonlyArray1<'py, f64>,
    dividend_amounts: PyReadonlyArray1<'py, f64>,
    volatility_convention: &str,
    maturity_boundary: &str,
    exercise_mesh: &str,
    base_intervals: Option<u32>,
    origin_exercise: Option<&str>,
    explicit_exercise_offsets: Option<PyReadonlyArray1<'py, i64>>,
    explicit_exercise_times: Option<PyReadonlyArray1<'py, f64>>,
    executor: PyRef<'py, NativeCpuExecutor>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain_start: u32,
    evaluation_seed: u64,
    evaluation_domain_start: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
    insufficient_exercise_support: &str,
) -> PyResult<NativeBatchLsmResult<'py>> {
    let columns = NativeBatchColumns {
        initial_spots: initial_spots.as_slice()?,
        strikes: strikes.as_slice()?,
        maturities: maturities.as_slice()?,
        constant_continuously_compounded_rates: constant_continuously_compounded_rates
            .as_slice()?,
        constant_volatilities: constant_volatilities.as_slice()?,
    };
    let dividend_offsets = dividend_offsets.as_slice()?;
    let dividend_times = dividend_times.as_slice()?;
    let dividend_amounts = dividend_amounts.as_slice()?;
    let volatility_convention = parse_escrowed_volatility_convention(volatility_convention)?;
    let maturity_boundary = parse_maturity_boundary(maturity_boundary)?;
    let explicit_exercise_offsets = explicit_exercise_offsets
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let explicit_exercise_times = explicit_exercise_times
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let method = parse_batch_lsm_method(
        exercise_mesh,
        base_intervals,
        origin_exercise,
        explicit_exercise_offsets,
        explicit_exercise_times,
        training_paths,
        evaluation_paths,
        training_seed,
        training_domain_start,
        evaluation_seed,
        evaluation_domain_start,
        polynomial_degree,
        rank_tolerance,
        condition_limit,
        tie_rule,
        insufficient_exercise_support,
    )?;
    let core_executor = &executor.inner;
    let result = py.detach(move || -> PyResult<CoreBatchLsmCallResult> {
        let schedules = RaggedCashDividends::try_new_with_executor(
            dividend_offsets,
            dividend_times,
            dividend_amounts,
            core_executor,
        )
        .map_err(CoreBatchError::into_py_batch_lsm_dividend_error)?;
        columns
            .price(
                CoreBatchDividendModel::EscrowedCashDividends {
                    schedules,
                    volatility_convention,
                    maturity_boundary,
                },
                method,
                core_executor,
            )
            .map_err(CoreBatchLsmError::into_py_batch_lsm_error)
    })?;
    native_batch_lsm_result(py, result)
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_lsm_call_grouped_batch_continuous_yield_f64<'py>(
    py: Python<'py>,
    group_offsets: PyReadonlyArray1<'py, u32>,
    initial_spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    constant_continuously_compounded_rates: PyReadonlyArray1<'py, f64>,
    constant_volatilities: PyReadonlyArray1<'py, f64>,
    annualized_dividend_yields: PyReadonlyArray1<'py, f64>,
    exercise_mesh: &str,
    base_intervals: Option<u32>,
    origin_exercise: Option<&str>,
    explicit_exercise_offsets: Option<PyReadonlyArray1<'py, i64>>,
    explicit_exercise_times: Option<PyReadonlyArray1<'py, f64>>,
    executor: PyRef<'py, NativeCpuExecutor>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain_start: u32,
    evaluation_seed: u64,
    evaluation_domain_start: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
    insufficient_exercise_support: &str,
) -> PyResult<NativeBatchLsmResult<'py>> {
    let columns = NativeGroupedBatchColumns {
        group_offsets: group_offsets.as_slice()?,
        initial_spots: initial_spots.as_slice()?,
        strikes: strikes.as_slice()?,
        maturities: maturities.as_slice()?,
        constant_continuously_compounded_rates: constant_continuously_compounded_rates
            .as_slice()?,
        constant_volatilities: constant_volatilities.as_slice()?,
    };
    let annualized_dividend_yields = annualized_dividend_yields.as_slice()?;
    let explicit_exercise_offsets = explicit_exercise_offsets
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let explicit_exercise_times = explicit_exercise_times
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let method = parse_batch_lsm_method(
        exercise_mesh,
        base_intervals,
        origin_exercise,
        explicit_exercise_offsets,
        explicit_exercise_times,
        training_paths,
        evaluation_paths,
        training_seed,
        training_domain_start,
        evaluation_seed,
        evaluation_domain_start,
        polynomial_degree,
        rank_tolerance,
        condition_limit,
        tie_rule,
        insufficient_exercise_support,
    )?;
    let core_executor = &executor.inner;
    let result = py
        .detach(move || {
            columns.price(
                CoreBatchDividendModel::ContinuousYield {
                    annualized_decimals: annualized_dividend_yields,
                },
                method,
                core_executor,
            )
        })
        .map_err(CoreBatchLsmError::into_py_batch_lsm_error)?;
    native_batch_lsm_result(py, result)
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_lsm_call_grouped_batch_cash_dividends_f64<'py>(
    py: Python<'py>,
    group_offsets: PyReadonlyArray1<'py, u32>,
    initial_spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    constant_continuously_compounded_rates: PyReadonlyArray1<'py, f64>,
    constant_volatilities: PyReadonlyArray1<'py, f64>,
    dividend_offsets: PyReadonlyArray1<'py, i64>,
    dividend_times: PyReadonlyArray1<'py, f64>,
    dividend_amounts: PyReadonlyArray1<'py, f64>,
    dividend_policy: &str,
    maturity_boundary: &str,
    exercise_mesh: &str,
    base_intervals: Option<u32>,
    origin_exercise: Option<&str>,
    explicit_exercise_offsets: Option<PyReadonlyArray1<'py, i64>>,
    explicit_exercise_times: Option<PyReadonlyArray1<'py, f64>>,
    executor: PyRef<'py, NativeCpuExecutor>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain_start: u32,
    evaluation_seed: u64,
    evaluation_domain_start: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
    insufficient_exercise_support: &str,
) -> PyResult<NativeBatchLsmResult<'py>> {
    let columns = NativeGroupedBatchColumns {
        group_offsets: group_offsets.as_slice()?,
        initial_spots: initial_spots.as_slice()?,
        strikes: strikes.as_slice()?,
        maturities: maturities.as_slice()?,
        constant_continuously_compounded_rates: constant_continuously_compounded_rates
            .as_slice()?,
        constant_volatilities: constant_volatilities.as_slice()?,
    };
    let dividend_offsets = dividend_offsets.as_slice()?;
    let dividend_times = dividend_times.as_slice()?;
    let dividend_amounts = dividend_amounts.as_slice()?;
    let dividend_policy = parse_dividend_policy(dividend_policy)?;
    let maturity_boundary = parse_maturity_boundary(maturity_boundary)?;
    let explicit_exercise_offsets = explicit_exercise_offsets
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let explicit_exercise_times = explicit_exercise_times
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let method = parse_batch_lsm_method(
        exercise_mesh,
        base_intervals,
        origin_exercise,
        explicit_exercise_offsets,
        explicit_exercise_times,
        training_paths,
        evaluation_paths,
        training_seed,
        training_domain_start,
        evaluation_seed,
        evaluation_domain_start,
        polynomial_degree,
        rank_tolerance,
        condition_limit,
        tie_rule,
        insufficient_exercise_support,
    )?;
    let core_executor = &executor.inner;
    let result = py.detach(move || -> PyResult<CoreBatchLsmCallResult> {
        let schedules = RaggedCashDividends::try_new_with_executor(
            dividend_offsets,
            dividend_times,
            dividend_amounts,
            core_executor,
        )
        .map_err(CoreBatchError::into_py_batch_lsm_dividend_error)?;
        columns
            .price(
                CoreBatchDividendModel::CashDividends {
                    schedules,
                    policy: dividend_policy,
                    maturity_boundary,
                },
                method,
                core_executor,
            )
            .map_err(CoreBatchLsmError::into_py_batch_lsm_error)
    })?;
    native_batch_lsm_result(py, result)
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
fn price_lsm_call_grouped_batch_escrowed_cash_dividends_f64<'py>(
    py: Python<'py>,
    group_offsets: PyReadonlyArray1<'py, u32>,
    initial_spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    constant_continuously_compounded_rates: PyReadonlyArray1<'py, f64>,
    constant_volatilities: PyReadonlyArray1<'py, f64>,
    dividend_offsets: PyReadonlyArray1<'py, i64>,
    dividend_times: PyReadonlyArray1<'py, f64>,
    dividend_amounts: PyReadonlyArray1<'py, f64>,
    volatility_convention: &str,
    maturity_boundary: &str,
    exercise_mesh: &str,
    base_intervals: Option<u32>,
    origin_exercise: Option<&str>,
    explicit_exercise_offsets: Option<PyReadonlyArray1<'py, i64>>,
    explicit_exercise_times: Option<PyReadonlyArray1<'py, f64>>,
    executor: PyRef<'py, NativeCpuExecutor>,
    training_paths: usize,
    evaluation_paths: usize,
    training_seed: u64,
    training_domain_start: u32,
    evaluation_seed: u64,
    evaluation_domain_start: u32,
    polynomial_degree: usize,
    rank_tolerance: f64,
    condition_limit: f64,
    tie_rule: &str,
    insufficient_exercise_support: &str,
) -> PyResult<NativeBatchLsmResult<'py>> {
    let columns = NativeGroupedBatchColumns {
        group_offsets: group_offsets.as_slice()?,
        initial_spots: initial_spots.as_slice()?,
        strikes: strikes.as_slice()?,
        maturities: maturities.as_slice()?,
        constant_continuously_compounded_rates: constant_continuously_compounded_rates
            .as_slice()?,
        constant_volatilities: constant_volatilities.as_slice()?,
    };
    let dividend_offsets = dividend_offsets.as_slice()?;
    let dividend_times = dividend_times.as_slice()?;
    let dividend_amounts = dividend_amounts.as_slice()?;
    let volatility_convention = parse_escrowed_volatility_convention(volatility_convention)?;
    let maturity_boundary = parse_maturity_boundary(maturity_boundary)?;
    let explicit_exercise_offsets = explicit_exercise_offsets
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let explicit_exercise_times = explicit_exercise_times
        .as_ref()
        .map(PyReadonlyArray1::as_slice)
        .transpose()?;
    let method = parse_batch_lsm_method(
        exercise_mesh,
        base_intervals,
        origin_exercise,
        explicit_exercise_offsets,
        explicit_exercise_times,
        training_paths,
        evaluation_paths,
        training_seed,
        training_domain_start,
        evaluation_seed,
        evaluation_domain_start,
        polynomial_degree,
        rank_tolerance,
        condition_limit,
        tie_rule,
        insufficient_exercise_support,
    )?;
    let core_executor = &executor.inner;
    let result = py.detach(move || -> PyResult<CoreBatchLsmCallResult> {
        let schedules = RaggedCashDividends::try_new_with_executor(
            dividend_offsets,
            dividend_times,
            dividend_amounts,
            core_executor,
        )
        .map_err(CoreBatchError::into_py_batch_lsm_dividend_error)?;
        columns
            .price(
                CoreBatchDividendModel::EscrowedCashDividends {
                    schedules,
                    volatility_convention,
                    maturity_boundary,
                },
                method,
                core_executor,
            )
            .map_err(CoreBatchLsmError::into_py_batch_lsm_error)
    })?;
    native_batch_lsm_result(py, result)
}

/// Private native module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCpuExecutor>()?;
    module.add_class::<NativeEventAlignedQuadrature>()?;
    module.add("InputError", module.py().get_type::<InputError>())?;
    module.add("BatchError", module.py().get_type::<BatchError>())?;
    module.add("ExecutionError", module.py().get_type::<ExecutionError>())?;
    module.add("SimulationError", module.py().get_type::<SimulationError>())?;
    module.add("LsmError", module.py().get_type::<LsmError>())?;
    module.add("QuadratureError", module.py().get_type::<QuadratureError>())?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(bin_dividends_f32, module)?)?;
    module.add_function(wrap_pyfunction!(pack_padded_dividends_f32, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_gbm_f64, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_gbm_seeded_f64, module)?)?;
    module.add_function(wrap_pyfunction!(price_lsm_call_seeded_f64, module)?)?;
    module.add_function(wrap_pyfunction!(price_event_aligned_call_f64, module)?)?;
    module.add_function(wrap_pyfunction!(
        price_lsm_call_batch_continuous_yield_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        price_lsm_call_batch_cash_dividends_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        price_lsm_call_batch_escrowed_cash_dividends_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        price_lsm_call_grouped_batch_continuous_yield_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        price_lsm_call_grouped_batch_cash_dividends_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        price_lsm_call_grouped_batch_escrowed_cash_dividends_f64,
        module
    )?)?;
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
    module.add_function(wrap_pyfunction!(
        validate_ragged_exercise_times_f64,
        module
    )?)?;
    Ok(())
}
