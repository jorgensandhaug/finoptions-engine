//! Private Python extension. The stable public surface lives in `python/finoptions`.

use finoptions_core::{
    CashDividend, ConstantVolatility, DividendPolicy, EventSchedule, InputError as CoreInputError,
    LogDiscountCurve, TimeGrid,
};
use finoptions_sim::{
    CpuExecutor as CoreCpuExecutor, GbmPathPlan, RandomStream as CoreRandomStream, SimulatedPaths,
    SimulationError as CoreSimulationError,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

create_exception!(_native, InputError, PyValueError);
create_exception!(_native, SimulationError, PyRuntimeError);

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
            .map_err(CoreSimulationError::into_py_simulation_error)
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
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

#[allow(clippy::too_many_arguments)]
fn parse_gbm_plan(
    schedule_times: Vec<f64>,
    curve_times: Vec<f64>,
    log_discounts: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
    annualized_volatility: f64,
    dividend_policy: &str,
) -> PyResult<GbmPathPlan> {
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
    GbmPathPlan::try_new(&schedule, &curve, volatility, policy)
        .map_err(CoreSimulationError::into_py_simulation_error)
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
        .detach(move || core_executor.simulate_gbm(&plan, initial_spot, paths, normal_values))
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
        .detach(move || core_executor.simulate_gbm_seeded(&plan, initial_spot, paths, stream))
        .map_err(CoreSimulationError::into_py_simulation_error)?;
    simulated_into_array(py, simulated)
}

/// Private native module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCpuExecutor>()?;
    module.add("InputError", module.py().get_type::<InputError>())?;
    module.add("SimulationError", module.py().get_type::<SimulationError>())?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_gbm_f64, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_gbm_seeded_f64, module)?)?;
    module.add_function(wrap_pyfunction!(validate_cash_dividend_amount, module)?)?;
    module.add_function(wrap_pyfunction!(validate_constant_volatility, module)?)?;
    module.add_function(wrap_pyfunction!(validate_event_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_log_discount_curve, module)?)?;
    Ok(())
}
