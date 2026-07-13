//! Private Python extension. The stable public surface lives in `python/finoptions`.

use finoptions_core::{
    CashDividend, EventSchedule, InputError as CoreInputError, LogDiscountCurve, TimeGrid,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(_native, InputError, PyValueError);

trait IntoPyInputError {
    fn into_py_input_error(self) -> PyErr;
}

impl IntoPyInputError for CoreInputError {
    fn into_py_input_error(self) -> PyErr {
        InputError::new_err(self.to_string())
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

/// Validates an exact event schedule without returning a mutable native handle.
#[pyfunction]
fn validate_event_schedule(
    times: Vec<f64>,
    dividend_nodes: Vec<u32>,
    dividend_amounts: Vec<f64>,
    exercise_nodes: Vec<u32>,
    expiry_node: u32,
) -> PyResult<()> {
    if dividend_nodes.len() != dividend_amounts.len() {
        return Err(CoreInputError::DividendColumnLengthMismatch {
            nodes: dividend_nodes.len(),
            amounts: dividend_amounts.len(),
        }
        .into_py_input_error());
    }
    let grid = TimeGrid::try_new(times).map_err(CoreInputError::into_py_input_error)?;
    let dividends = dividend_nodes
        .into_iter()
        .zip(dividend_amounts)
        .map(|(node, amount)| {
            let node = grid.node_id(node as usize)?;
            CashDividend::try_new(node, amount)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(CoreInputError::into_py_input_error)?;
    let exercise_nodes = exercise_nodes
        .into_iter()
        .map(|node| grid.node_id(node as usize))
        .collect::<Result<Vec<_>, _>>()
        .map_err(CoreInputError::into_py_input_error)?;
    let expiry = grid
        .node_id(expiry_node as usize)
        .map_err(CoreInputError::into_py_input_error)?;
    EventSchedule::try_new(grid, dividends, exercise_nodes, expiry)
        .map(|_| ())
        .map_err(CoreInputError::into_py_input_error)
}

/// Validates node-aligned log discount factors.
#[pyfunction]
fn validate_log_discount_curve(times: Vec<f64>, log_discounts: Vec<f64>) -> PyResult<()> {
    let grid = TimeGrid::try_new(times).map_err(CoreInputError::into_py_input_error)?;
    LogDiscountCurve::try_new(&grid, log_discounts)
        .map(|_| ())
        .map_err(CoreInputError::into_py_input_error)
}

/// Private native module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("InputError", module.py().get_type::<InputError>())?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(validate_cash_dividend_amount, module)?)?;
    module.add_function(wrap_pyfunction!(validate_event_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_log_discount_curve, module)?)?;
    Ok(())
}
