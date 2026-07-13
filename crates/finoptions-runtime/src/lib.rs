//! Explicit reusable execution resources shared by independent engine kernels.

use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

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
}

/// Dedicated, explicitly sized CPU executor shared across engine kernels.
///
/// A one-thread executor dispatches the serial kernel without constructing a worker pool. Larger
/// executors own an isolated Rayon pool and never consult or mutate Rayon's global configuration.
#[derive(Debug)]
pub struct CpuExecutor {
    threads: usize,
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
        if threads == 0 {
            return Err(ExecutionError::EmptyThreadPool);
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
        Ok(Self { threads, pool })
    }

    /// Returns the exact configured worker count.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
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
