//! Explicit reusable execution resources shared by independent engine kernels.

use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

/// Explicit double-precision transcendental implementation for CPU kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuMathBackend {
    /// Rust standard-library scalar `f64` operations.
    RustStdF64,
    /// Apple Accelerate vForce vector kernels. Available only on macOS.
    AppleAccelerateVforce,
}

impl CpuMathBackend {
    /// Returns the stable cross-language backend identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RustStdF64 => "rust_std_f64",
            Self::AppleAccelerateVforce => "apple_accelerate_vforce",
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
            Self::AppleAccelerateVforce => apple_exp_f64(input, output),
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
        backend: CpuMathBackend::AppleAccelerateVforce.as_str(),
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
        if math_backend == CpuMathBackend::AppleAccelerateVforce && !cfg!(target_os = "macos") {
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
            CpuExecutor::try_new_with_math_backend(2, CpuMathBackend::AppleAccelerateVforce)
                .unwrap();
        assert_eq!(
            executor.math_backend(),
            CpuMathBackend::AppleAccelerateVforce
        );
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
