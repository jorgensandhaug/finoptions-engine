//! Explicit reusable CPU execution resources.

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::{GbmPathPlan, RandomStream, SimulatedPaths, SimulationError};

/// Dedicated, explicitly sized CPU executor for reproducible path kernels.
///
/// A one-thread executor uses the serial kernel without constructing a worker pool. Larger
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
    /// Returns [`SimulationError::EmptyThreadPool`] for zero or a typed pool-build failure when
    /// operating-system resources cannot be created.
    pub fn try_new(threads: usize) -> Result<Self, SimulationError> {
        if threads == 0 {
            return Err(SimulationError::EmptyThreadPool);
        }
        let pool = if threads == 1 {
            None
        } else {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("finoptions-cpu-{index}"))
                    .build()
                    .map_err(|error| SimulationError::ThreadPoolBuild {
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

    /// Simulates GBM paths using this executor's explicit serial or dedicated-pool kernel.
    ///
    /// # Errors
    ///
    /// Propagates strict simulation input or execution failures. There is no pool/global/serial
    /// fallback when more than one thread was requested.
    pub fn simulate_gbm(
        &self,
        plan: &GbmPathPlan,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        match &self.pool {
            None => plan.simulate(initial_spot, paths, normals),
            Some(pool) => pool.install(|| plan.simulate_parallel(initial_spot, paths, normals)),
        }
    }

    /// Generates and simulates GBM paths in one fused kernel using an explicit random namespace.
    ///
    /// # Errors
    ///
    /// Propagates strict simulation input, random-generation, or execution failures. The requested
    /// worker count and random stream are never substituted.
    pub fn simulate_gbm_seeded(
        &self,
        plan: &GbmPathPlan,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        match &self.pool {
            None => plan.simulate_seeded(initial_spot, paths, stream),
            Some(pool) => {
                pool.install(|| plan.simulate_seeded_parallel(initial_spot, paths, stream))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_and_preserves_explicit_thread_count() {
        assert!(matches!(
            CpuExecutor::try_new(0),
            Err(SimulationError::EmptyThreadPool)
        ));
        assert_eq!(CpuExecutor::try_new(1).unwrap().threads(), 1);
        assert_eq!(CpuExecutor::try_new(2).unwrap().threads(), 2);
    }
}
