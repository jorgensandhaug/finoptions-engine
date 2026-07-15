//! Explicit reusable execution resources shared by independent engine kernels.

use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

/// Explicit double-precision numerical implementation for CPU kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuMathBackend {
    /// Rust standard-library scalar `f64` operations.
    RustStdF64,
    /// Apple vForce plus architecture-specific vector kernels. Available only on macOS.
    AppleAccelerate,
}

impl CpuMathBackend {
    /// Returns the stable cross-language backend identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RustStdF64 => "rust_std_f64",
            Self::AppleAccelerate => "apple_accelerate",
        }
    }

    /// Computes `exp(input[i])` into the distinct output slice.
    ///
    /// # Errors
    ///
    /// Returns a typed error on a length mismatch or when an unavailable backend is requested.
    pub fn exp_f64(self, input: &[f64], output: &mut [f64]) -> Result<(), CpuMathError> {
        if input.len() != output.len() {
            return Err(CpuMathError::LengthMismatch {
                input: input.len(),
                output: output.len(),
            });
        }
        match self {
            Self::RustStdF64 => {
                for (result, exponent) in output.iter_mut().zip(input) {
                    *result = exponent.exp();
                }
                Ok(())
            }
            Self::AppleAccelerate => apple_exp_f64(input, output),
        }
    }

    /// Computes the three reductions used by one empirical orthogonal-polynomial basis vector.
    ///
    /// The result is `(sum(basis^2), sum(responses * basis),
    /// sum(predictors * basis^2))`. `squared_basis` is caller-owned workspace whose contents after
    /// a successful call are backend-specific and must not be observed by the caller.
    ///
    /// # Errors
    ///
    /// Returns before writing workspace unless every operand has the same length, or when the
    /// explicitly requested implementation is unavailable. No other backend is attempted.
    #[inline]
    pub fn polynomial_moments_f64(
        self,
        predictors: &[f64],
        responses: &[f64],
        basis: &[f64],
        squared_basis: &mut [f64],
    ) -> Result<(f64, f64, f64), CpuMathError> {
        const OPERATION: &str = "polynomial_moments_f64";
        let expected = predictors.len();
        for (operand, observed) in [
            ("responses", responses.len()),
            ("basis", basis.len()),
            ("squared_basis", squared_basis.len()),
        ] {
            if observed != expected {
                return Err(CpuMathError::OperandLengthMismatch {
                    operation: OPERATION,
                    operand,
                    expected,
                    observed,
                });
            }
        }
        match self {
            Self::RustStdF64 => {
                let mut norm = 0.0;
                let mut projection = 0.0;
                let mut weighted_predictor = 0.0;
                for ((&predictor, &response), &basis_value) in
                    predictors.iter().zip(responses).zip(basis)
                {
                    let squared_basis = basis_value * basis_value;
                    norm += squared_basis;
                    projection = response.mul_add(basis_value, projection);
                    weighted_predictor = predictor.mul_add(squared_basis, weighted_predictor);
                }
                Ok((norm, projection, weighted_predictor))
            }
            Self::AppleAccelerate => {
                apple_polynomial_moments_f64(predictors, responses, basis, squared_basis)
            }
        }
    }

    /// Constructs the next empirical orthogonal basis vector and computes its moments in one pass.
    ///
    /// The recurrence is `next = (predictor - alpha) * current - beta * previous`. Every operand
    /// length is validated before output is written, and the selected backend is never changed.
    ///
    /// # Errors
    ///
    /// Returns a typed length or unavailable-backend error without retrying another implementation.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn polynomial_recurrence_moments_f64(
        self,
        predictors: &[f64],
        responses: &[f64],
        previous_basis: &[f64],
        current_basis: &[f64],
        next_basis_output: &mut [f64],
        alpha: f64,
        beta: f64,
    ) -> Result<(f64, f64, f64), CpuMathError> {
        const OPERATION: &str = "polynomial_recurrence_moments_f64";
        let expected = predictors.len();
        for (operand, observed) in [
            ("responses", responses.len()),
            ("previous_basis", previous_basis.len()),
            ("current_basis", current_basis.len()),
            ("next_basis_output", next_basis_output.len()),
        ] {
            if observed != expected {
                return Err(CpuMathError::OperandLengthMismatch {
                    operation: OPERATION,
                    operand,
                    expected,
                    observed,
                });
            }
        }
        match self {
            Self::RustStdF64 => {
                let mut norm = 0.0;
                let mut projection = 0.0;
                let mut weighted_predictor = 0.0;
                for ((((predictor, response), previous), current), next) in predictors
                    .iter()
                    .zip(responses)
                    .zip(previous_basis)
                    .zip(current_basis)
                    .zip(next_basis_output)
                {
                    *next = (*predictor - alpha).mul_add(*current, -beta * *previous);
                    let squared = *next * *next;
                    norm += squared;
                    projection = response.mul_add(*next, projection);
                    weighted_predictor = predictor.mul_add(squared, weighted_predictor);
                }
                Ok((norm, projection, weighted_predictor))
            }
            Self::AppleAccelerate => apple_polynomial_recurrence_moments_f64(
                predictors,
                responses,
                previous_basis,
                current_basis,
                next_basis_output,
                alpha,
                beta,
            ),
        }
    }
}

/// A requested CPU math operation cannot honor its exact backend/buffer contract.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CpuMathError {
    /// Input and output vectors have different element counts.
    #[error("CPU math input/output length mismatch: input {input}, output {output}")]
    LengthMismatch {
        /// Input elements.
        input: usize,
        /// Output elements.
        output: usize,
    },

    /// Operands of one named multi-vector kernel have different element counts.
    #[error(
        "CPU math kernel {operation} operand {operand} has length {observed}; expected {expected}"
    )]
    OperandLengthMismatch {
        /// Stable kernel identifier.
        operation: &'static str,
        /// Stable operand identifier.
        operand: &'static str,
        /// Required element count.
        expected: usize,
        /// Observed element count.
        observed: usize,
    },

    /// The requested named implementation is not compiled for this platform.
    #[error("CPU math backend {backend} is unavailable on this platform")]
    BackendUnavailable {
        /// Stable backend identifier.
        backend: &'static str,
    },
}

#[cfg(target_os = "macos")]
fn apple_exp_f64(input: &[f64], output: &mut [f64]) -> Result<(), CpuMathError> {
    finoptions_accelerate::exp_f64(input, output).map_err(|error| match error {
        finoptions_accelerate::AccelerateError::LengthMismatch { input, output } => {
            CpuMathError::LengthMismatch { input, output }
        }
    })
}

#[cfg(not(target_os = "macos"))]
fn apple_exp_f64(_input: &[f64], _output: &mut [f64]) -> Result<(), CpuMathError> {
    Err(CpuMathError::BackendUnavailable {
        backend: CpuMathBackend::AppleAccelerate.as_str(),
    })
}

#[cfg(target_os = "macos")]
fn apple_polynomial_moments_f64(
    predictors: &[f64],
    responses: &[f64],
    basis: &[f64],
    squared_basis: &mut [f64],
) -> Result<(f64, f64, f64), CpuMathError> {
    finoptions_accelerate::polynomial_moments_f64(predictors, responses, basis, squared_basis)
        .map_err(|error| match error {
            finoptions_accelerate::VectorKernelError::OperandLengthMismatch {
                operation,
                operand,
                expected,
                observed,
            } => CpuMathError::OperandLengthMismatch {
                operation,
                operand,
                expected,
                observed,
            },
        })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn apple_polynomial_recurrence_moments_f64(
    predictors: &[f64],
    responses: &[f64],
    previous_basis: &[f64],
    current_basis: &[f64],
    next_basis_output: &mut [f64],
    alpha: f64,
    beta: f64,
) -> Result<(f64, f64, f64), CpuMathError> {
    finoptions_accelerate::polynomial_recurrence_moments_f64(
        predictors,
        responses,
        previous_basis,
        current_basis,
        next_basis_output,
        alpha,
        beta,
    )
    .map_err(|error| match error {
        finoptions_accelerate::VectorKernelError::OperandLengthMismatch {
            operation,
            operand,
            expected,
            observed,
        } => CpuMathError::OperandLengthMismatch {
            operation,
            operand,
            expected,
            observed,
        },
    })
}

#[cfg(not(target_os = "macos"))]
fn apple_polynomial_moments_f64(
    _predictors: &[f64],
    _responses: &[f64],
    _basis: &[f64],
    _squared_basis: &mut [f64],
) -> Result<(f64, f64, f64), CpuMathError> {
    Err(CpuMathError::BackendUnavailable {
        backend: CpuMathBackend::AppleAccelerate.as_str(),
    })
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn apple_polynomial_recurrence_moments_f64(
    _predictors: &[f64],
    _responses: &[f64],
    _previous_basis: &[f64],
    _current_basis: &[f64],
    _next_basis_output: &mut [f64],
    _alpha: f64,
    _beta: f64,
) -> Result<(f64, f64, f64), CpuMathError> {
    Err(CpuMathError::BackendUnavailable {
        backend: CpuMathBackend::AppleAccelerate.as_str(),
    })
}

/// A failure to construct an explicitly requested execution resource.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionError {
    /// A CPU executor requested no worker threads.
    #[error("CPU executor thread count must be positive")]
    EmptyThreadPool,

    /// A dedicated CPU worker pool could not be created.
    #[error("failed to build CPU executor with {threads} threads: {reason}")]
    ThreadPoolBuild {
        /// Explicit requested worker count.
        threads: usize,
        /// Stable upstream failure message.
        reason: String,
    },

    /// A named CPU math backend is not built for the current platform.
    #[error("CPU math backend {backend} is unavailable on this platform")]
    MathBackendUnavailable {
        /// Stable requested backend identifier.
        backend: &'static str,
    },
}

/// Dedicated, explicitly sized CPU executor shared across engine kernels.
///
/// A one-thread executor dispatches the serial kernel without constructing a worker pool. Larger
/// executors own an isolated Rayon pool and never consult or mutate Rayon's global configuration.
#[derive(Debug)]
pub struct CpuExecutor {
    threads: usize,
    math_backend: CpuMathBackend,
    pool: Option<ThreadPool>,
}

impl CpuExecutor {
    /// Creates a reusable executor with an exact worker count.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::EmptyThreadPool`] for zero or a typed pool-build failure when
    /// operating-system resources cannot be created. The requested worker count is never changed.
    pub fn try_new(threads: usize) -> Result<Self, ExecutionError> {
        Self::try_new_with_math_backend(threads, CpuMathBackend::RustStdF64)
    }

    /// Creates a reusable executor with exact worker and CPU math implementations.
    ///
    /// # Errors
    ///
    /// Returns before pool construction when the named math backend is unavailable. Otherwise,
    /// applies the same strict thread-pool validation as [`Self::try_new`].
    pub fn try_new_with_math_backend(
        threads: usize,
        math_backend: CpuMathBackend,
    ) -> Result<Self, ExecutionError> {
        if threads == 0 {
            return Err(ExecutionError::EmptyThreadPool);
        }
        if math_backend == CpuMathBackend::AppleAccelerate && !cfg!(target_os = "macos") {
            return Err(ExecutionError::MathBackendUnavailable {
                backend: math_backend.as_str(),
            });
        }
        let pool = if threads == 1 {
            None
        } else {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("finoptions-cpu-{index}"))
                    .build()
                    .map_err(|error| ExecutionError::ThreadPoolBuild {
                        threads,
                        reason: error.to_string(),
                    })?,
            )
        };
        Ok(Self {
            threads,
            math_backend,
            pool,
        })
    }

    /// Returns the exact configured worker count.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
    }

    /// Returns the exact configured CPU math implementation.
    #[must_use]
    pub const fn math_backend(&self) -> CpuMathBackend {
        self.math_backend
    }

    /// Dispatches exactly one caller-supplied serial or parallel kernel.
    ///
    /// The parallel closure runs only inside this executor's dedicated pool. A one-thread executor
    /// runs the distinct serial closure, so callers can avoid Rayon scheduling overhead while
    /// preserving an explicit backend contract.
    pub fn dispatch<R, S, P>(&self, serial: S, parallel: P) -> R
    where
        R: Send,
        S: FnOnce() -> R,
        P: FnOnce() -> R + Send,
    {
        match &self.pool {
            None => serial(),
            Some(pool) => pool.install(parallel),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn rejects_zero_and_preserves_explicit_thread_count() {
        assert_eq!(
            CpuExecutor::try_new(0).unwrap_err(),
            ExecutionError::EmptyThreadPool
        );
        assert_eq!(CpuExecutor::try_new(1).unwrap().threads(), 1);
        assert_eq!(CpuExecutor::try_new(2).unwrap().threads(), 2);
        assert_eq!(
            CpuExecutor::try_new(2).unwrap().math_backend(),
            CpuMathBackend::RustStdF64
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn preserves_explicit_apple_math_backend() {
        let executor =
            CpuExecutor::try_new_with_math_backend(2, CpuMathBackend::AppleAccelerate).unwrap();
        assert_eq!(executor.math_backend(), CpuMathBackend::AppleAccelerate);
        let mut output = [0.0; 3];
        executor
            .math_backend()
            .exp_f64(&[-1.0, 0.0, 1.0], &mut output)
            .unwrap();
        for (&observed, exponent) in output.iter().zip([-1.0_f64, 0.0, 1.0]) {
            assert!(observed.to_bits().abs_diff(exponent.exp().to_bits()) <= 1);
        }
    }

    #[test]
    fn polynomial_moments_validate_operands_before_dispatch() {
        let mut scratch = [7.0; 2];
        assert_eq!(
            CpuMathBackend::RustStdF64.polynomial_moments_f64(
                &[1.0, 2.0],
                &[3.0],
                &[4.0, 5.0],
                &mut scratch,
            ),
            Err(CpuMathError::OperandLengthMismatch {
                operation: "polynomial_moments_f64",
                operand: "responses",
                expected: 2,
                observed: 1,
            })
        );
        assert_eq!(scratch.map(f64::to_bits), [7.0, 7.0].map(f64::to_bits));

        assert_eq!(
            CpuMathBackend::RustStdF64
                .polynomial_moments_f64(&[0.5, 1.5], &[2.0, 3.0], &[-1.0, 2.0], &mut scratch,)
                .unwrap(),
            (5.0, 4.0, 6.5),
        );

        assert_eq!(
            CpuMathBackend::RustStdF64.polynomial_recurrence_moments_f64(
                &[0.5, 1.5],
                &[2.0, 3.0],
                &[0.0],
                &[1.0, 1.0],
                &mut scratch,
                0.0,
                0.0,
            ),
            Err(CpuMathError::OperandLengthMismatch {
                operation: "polynomial_recurrence_moments_f64",
                operand: "previous_basis",
                expected: 2,
                observed: 1,
            })
        );
        assert_eq!(scratch.map(f64::to_bits), [7.0, 7.0].map(f64::to_bits));

        let observed = CpuMathBackend::RustStdF64
            .polynomial_recurrence_moments_f64(
                &[0.5, 1.5],
                &[2.0, 3.0],
                &[0.0, 1.0],
                &[1.0, 2.0],
                &mut scratch,
                0.5,
                0.25,
            )
            .unwrap();
        assert_eq!(scratch.map(f64::to_bits), [0.0, 1.75].map(f64::to_bits));
        assert_eq!(observed, (3.0625, 5.25, 4.59375));
    }

    #[test]
    fn dispatches_only_the_kernel_selected_by_the_executor() {
        let serial_called = AtomicBool::new(false);
        let parallel_called = AtomicBool::new(false);
        let result = CpuExecutor::try_new(1).unwrap().dispatch(
            || {
                serial_called.store(true, Ordering::Relaxed);
                7
            },
            || {
                parallel_called.store(true, Ordering::Relaxed);
                11
            },
        );
        assert_eq!(result, 7);
        assert!(serial_called.load(Ordering::Relaxed));
        assert!(!parallel_called.load(Ordering::Relaxed));

        let serial_called = AtomicBool::new(false);
        let parallel_called = AtomicBool::new(false);
        let result = CpuExecutor::try_new(2).unwrap().dispatch(
            || {
                serial_called.store(true, Ordering::Relaxed);
                7
            },
            || {
                parallel_called.store(true, Ordering::Relaxed);
                11
            },
        );
        assert_eq!(result, 11);
        assert!(!serial_called.load(Ordering::Relaxed));
        assert!(parallel_called.load(Ordering::Relaxed));
    }
}
