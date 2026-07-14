//! Execution-policy dispatch for GBM path plans.

use finoptions_runtime::CpuExecutor;

use crate::{EscrowedGbmPathPlan, GbmPathPlan, RandomStream, SimulatedPaths, SimulationError};

impl GbmPathPlan {
    /// Simulates GBM paths using an explicit serial or dedicated-pool executor.
    ///
    /// # Errors
    ///
    /// Propagates strict simulation input or execution failures. There is no pool/global/serial
    /// fallback when more than one thread was requested.
    pub fn simulate_with(
        &self,
        executor: &CpuExecutor,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        executor.dispatch(
            || self.simulate(initial_spot, paths, normals),
            || self.simulate_parallel(initial_spot, paths, normals),
        )
    }

    /// Generates and simulates paths using an explicit executor and random namespace.
    ///
    /// # Errors
    ///
    /// Propagates strict simulation input, random-generation, or execution failures. The requested
    /// worker count and random stream are never substituted.
    pub fn simulate_seeded_with(
        &self,
        executor: &CpuExecutor,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        executor.dispatch(
            || self.simulate_seeded(initial_spot, paths, stream),
            || self.simulate_seeded_parallel(initial_spot, paths, stream),
        )
    }
}

impl EscrowedGbmPathPlan {
    /// Simulates escrowed-dividend paths using an explicit serial or dedicated-pool executor.
    ///
    /// # Errors
    ///
    /// Propagates strict construction, input, simulation, or reconstruction failures. The
    /// requested execution policy is never substituted.
    pub fn simulate_with(
        &self,
        executor: &CpuExecutor,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        executor.dispatch(
            || self.simulate(initial_spot, paths, normals),
            || self.simulate_parallel(initial_spot, paths, normals),
        )
    }

    /// Generates escrowed-dividend paths using an explicit executor and random namespace.
    ///
    /// # Errors
    ///
    /// Propagates strict construction, input, random-generation, simulation, or reconstruction
    /// failures. The requested worker count and random stream are never substituted.
    pub fn simulate_seeded_with(
        &self,
        executor: &CpuExecutor,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        executor.dispatch(
            || self.simulate_seeded(initial_spot, paths, stream),
            || self.simulate_seeded_parallel(initial_spot, paths, stream),
        )
    }
}
