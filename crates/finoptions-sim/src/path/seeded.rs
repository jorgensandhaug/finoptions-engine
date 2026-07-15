//! Fused path-local random generation and event-aligned GBM evolution.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::{GbmPathPlan, SimulatedPaths};
use crate::random::{NormalError, PhiloxPathStream, sample_standard_normal};
use crate::{CpuMathBackend, RandomStream, SimulationError};

// Keeps contract-parallel worker scratch cache-sized and independent of batch rows. Blocking only
// changes transpose storage: random addresses and arithmetic within every path remain identical.
const SERIAL_PATH_BLOCK: usize = 32;
// Representative 150-step LSM batches reached the vector-throughput plateau at 64 paths while
// keeping one path-major observation tile substantially smaller than the 128/256-path candidates.
const VECTOR_PATH_BLOCK: usize = 64;

#[derive(Default)]
struct SeededVectorWorkspace {
    generators: Vec<PhiloxPathStream>,
    spots: Vec<f64>,
    normals: Vec<f64>,
    exponents: Vec<f64>,
    growth: Vec<f64>,
}

impl GbmPathPlan {
    /// Generates standard-normal variates and evolves paths without a dense shock matrix.
    ///
    /// Each path is a pure function of the plan, initial spot, [`RandomStream`], and path index.
    /// The result is therefore invariant to CPU worker count and scheduling on the same numerical
    /// build. Returned states preserve the usual time-major `[observations, paths]` layout.
    ///
    /// # Errors
    ///
    /// Returns a typed input, random-generation, dividend, or non-finite transition failure. No
    /// random algorithm, stream, precision, or execution backend is substituted on failure.
    pub fn simulate_seeded(
        &self,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        let (normal_budget, output_values) = self.validate_seeded(initial_spot, paths)?;
        let observations = self.observation_nodes.len();
        let block_capacity = paths.min(SERIAL_PATH_BLOCK);
        let mut block = vec![0.0; observations * block_capacity];
        let mut values = vec![0.0; output_values];
        for block_start in (0..paths).step_by(SERIAL_PATH_BLOCK) {
            let block_paths = (paths - block_start).min(SERIAL_PATH_BLOCK);
            let block_values = &mut block[..observations * block_paths];
            for (local_path, path_observations) in block_values.chunks_mut(observations).enumerate()
            {
                self.simulate_one_seeded(
                    initial_spot,
                    block_start + local_path,
                    normal_budget,
                    stream,
                    path_observations,
                )?;
            }
            for observation in 0..observations {
                let output_start = observation * paths + block_start;
                let output_row = &mut values[output_start..output_start + block_paths];
                for (local_path, output) in output_row.iter_mut().enumerate() {
                    *output = block_values[local_path * observations + observation];
                }
            }
        }
        Ok(SimulatedPaths {
            plan: self.clone(),
            random_stream: Some(stream),
            initial_spot,
            paths,
            values,
        })
    }

    /// Generates serial seeded paths with one explicit GBM exponential implementation.
    ///
    /// The scalar backend preserves the path-local reference kernel bit for bit. Vector backends
    /// preserve the Philox stream assigned to every path but evaluate transition exponentials in
    /// fixed path blocks, so their floating-point results are backend-specific by design.
    ///
    /// # Errors
    ///
    /// Returns a typed input, random-generation, backend, dividend, or transition failure. The
    /// named backend is never replaced.
    pub fn simulate_seeded_with_math_backend(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        match math_backend {
            CpuMathBackend::RustStdF64 => self.simulate_seeded(initial_spot, paths, stream),
            CpuMathBackend::AppleAccelerate => {
                self.simulate_seeded_vector_exp(math_backend, initial_spot, paths, stream)
            }
        }
    }

    fn simulate_seeded_vector_exp(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        let (normal_budget, output_values) = self.validate_seeded(initial_spot, paths)?;
        let observations = self.observation_nodes.len();
        let block_capacity = paths.min(VECTOR_PATH_BLOCK);
        let mut block = vec![0.0; observations * block_capacity];
        let mut values = vec![0.0; output_values];
        let mut workspace = SeededVectorWorkspace::default();
        for block_start in (0..paths).step_by(VECTOR_PATH_BLOCK) {
            let block_paths = (paths - block_start).min(VECTOR_PATH_BLOCK);
            let block_values = &mut block[..observations * block_paths];
            self.simulate_seeded_vector_block(
                math_backend,
                initial_spot,
                block_start,
                block_paths,
                normal_budget,
                stream,
                block_values,
                &mut workspace,
            )?;
            for observation in 0..observations {
                let output_start = observation * paths + block_start;
                let output_row = &mut values[output_start..output_start + block_paths];
                for (local_path, output) in output_row.iter_mut().enumerate() {
                    *output = block_values[local_path * observations + observation];
                }
            }
        }
        Ok(SimulatedPaths {
            plan: self.clone(),
            random_stream: Some(stream),
            initial_spot,
            paths,
            values,
        })
    }

    pub(crate) fn simulate_seeded_parallel(
        &self,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        let (normal_budget, output_values) = self.validate_seeded(initial_spot, paths)?;
        let mut path_major = vec![0.0; output_values];
        let first_failure = AtomicUsize::new(usize::MAX);
        path_major
            .par_chunks_mut(self.observation_nodes.len())
            .enumerate()
            .for_each(|(path, observations)| {
                if path > first_failure.load(Ordering::Relaxed) {
                    return;
                }
                if self
                    .simulate_one_seeded(initial_spot, path, normal_budget, stream, observations)
                    .is_err()
                {
                    first_failure.fetch_min(path, Ordering::Relaxed);
                }
            });

        let failed_path = first_failure.load(Ordering::Relaxed);
        if failed_path != usize::MAX {
            // Replay only the deterministic lowest failing path to recover its complete typed
            // diagnostic; this does not produce a value or select another algorithm/backend.
            let mut diagnostic = vec![0.0; self.observation_nodes.len()];
            return match self.simulate_one_seeded(
                initial_spot,
                failed_path,
                normal_budget,
                stream,
                &mut diagnostic,
            ) {
                Err(error) => Err(error),
                Ok(()) => Err(SimulationError::ParallelFailureReplayMismatch { path: failed_path }),
            };
        }

        let values = self.transpose_seeded_parallel(&path_major, paths);
        Ok(SimulatedPaths {
            plan: self.clone(),
            random_stream: Some(stream),
            initial_spot,
            paths,
            values,
        })
    }

    pub(crate) fn simulate_seeded_parallel_with_math_backend(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        match math_backend {
            CpuMathBackend::RustStdF64 => {
                self.simulate_seeded_parallel(initial_spot, paths, stream)
            }
            CpuMathBackend::AppleAccelerate => {
                self.simulate_seeded_vector_exp_parallel(math_backend, initial_spot, paths, stream)
            }
        }
    }

    fn simulate_seeded_vector_exp_parallel(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        let (normal_budget, output_values) = self.validate_seeded(initial_spot, paths)?;
        let observations = self.observation_nodes.len();
        let block_values = observations * VECTOR_PATH_BLOCK;
        let mut path_major = vec![0.0; output_values];
        let first_failed_block = AtomicUsize::new(usize::MAX);
        path_major
            .par_chunks_mut(block_values)
            .enumerate()
            .for_each_init(
                SeededVectorWorkspace::default,
                |workspace, (block_index, output)| {
                    let block_start = block_index * VECTOR_PATH_BLOCK;
                    if block_start > first_failed_block.load(Ordering::Relaxed) {
                        return;
                    }
                    let block_paths = output.len() / observations;
                    if self
                        .simulate_seeded_vector_block(
                            math_backend,
                            initial_spot,
                            block_start,
                            block_paths,
                            normal_budget,
                            stream,
                            output,
                            workspace,
                        )
                        .is_err()
                    {
                        first_failed_block.fetch_min(block_start, Ordering::Relaxed);
                    }
                },
            );

        let failed_block = first_failed_block.load(Ordering::Relaxed);
        if failed_block != usize::MAX {
            let block_paths = (paths - failed_block).min(VECTOR_PATH_BLOCK);
            let mut diagnostic = vec![0.0; observations * block_paths];
            let mut workspace = SeededVectorWorkspace::default();
            return match self.simulate_seeded_vector_block(
                math_backend,
                initial_spot,
                failed_block,
                block_paths,
                normal_budget,
                stream,
                &mut diagnostic,
                &mut workspace,
            ) {
                Err(error) => Err(error),
                Ok(()) => {
                    Err(SimulationError::ParallelFailureReplayMismatch { path: failed_block })
                }
            };
        }

        let values = self.transpose_seeded_parallel(&path_major, paths);
        Ok(SimulatedPaths {
            plan: self.clone(),
            random_stream: Some(stream),
            initial_spot,
            paths,
            values,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn simulate_seeded_vector_block(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        block_start: usize,
        block_paths: usize,
        normal_budget: u32,
        stream: RandomStream,
        output: &mut [f64],
        workspace: &mut SeededVectorWorkspace,
    ) -> Result<(), SimulationError> {
        let observations = self.observation_nodes.len();
        debug_assert_eq!(output.len(), observations * block_paths);
        workspace.generators.clear();
        for path in block_start..block_start + block_paths {
            let path_id =
                u32::try_from(path).map_err(|_| SimulationError::RandomPathNamespaceOverflow {
                    paths: path.saturating_add(1),
                    maximum: u32::MAX,
                })?;
            workspace.generators.push(PhiloxPathStream::new(
                stream.seed(),
                stream.domain(),
                path_id,
                normal_budget,
            ));
        }
        workspace.spots.resize(block_paths, initial_spot);
        workspace.spots.fill(initial_spot);
        workspace.normals.resize(block_paths, 0.0);
        workspace.exponents.resize(block_paths, 0.0);
        workspace.growth.resize(block_paths, 0.0);
        let mut observation = 0;
        if self.observation_nodes[0].get() == 0 {
            for local_path in 0..block_paths {
                output[local_path * observations] = initial_spot;
            }
            observation = 1;
        }

        for (interval, &step) in self.steps.iter().enumerate() {
            for local_path in 0..block_paths {
                let path = block_start + local_path;
                let normal = sample_standard_normal(&mut workspace.generators[local_path])
                    .map_err(|error| match error {
                        NormalError::BudgetExhausted => {
                            SimulationError::RandomNormalBudgetExhausted { interval, path }
                        }
                        NormalError::RejectionLimit => {
                            SimulationError::RandomNormalRejectionLimit { interval, path }
                        }
                    })?;
                workspace.normals[local_path] = normal;
                workspace.exponents[local_path] = if workspace.spots[local_path] == 0.0 {
                    0.0
                } else {
                    step.diffusion().mul_add(normal, step.log_drift())
                };
            }
            math_backend.exp_f64(&workspace.exponents, &mut workspace.growth)?;
            for local_path in 0..block_paths {
                let before = workspace.spots[local_path];
                let next = before * workspace.growth[local_path];
                if !next.is_finite() {
                    return Err(SimulationError::NonFinitePathTransition {
                        from: step.from().get(),
                        to: step.to().get(),
                        path: block_start + local_path,
                        spot: before,
                        normal: workspace.normals[local_path],
                    });
                }
                workspace.spots[local_path] = next;
            }

            let node = step.to();
            let dividend = self.dividends_by_node[node.get() as usize];
            let terminal_dividend = dividend.filter(|_| self.is_terminal_observation(node));
            if let Some(cash) = terminal_dividend {
                Self::apply_dividend(cash, self.dividend_policy, &mut workspace.spots)?;
            }
            if self.observation_nodes.get(observation).copied() == Some(node) {
                for (local_path, &spot) in workspace.spots.iter().enumerate() {
                    output[local_path * observations + observation] = spot;
                }
                observation += 1;
            }
            if let Some(cash) = dividend.filter(|_| terminal_dividend.is_none()) {
                Self::apply_dividend(cash, self.dividend_policy, &mut workspace.spots)?;
            }
        }
        debug_assert_eq!(observation, observations);
        Ok(())
    }

    fn validate_seeded(
        &self,
        initial_spot: f64,
        paths: usize,
    ) -> Result<(u32, usize), SimulationError> {
        Self::validate_initial_spot(initial_spot)?;
        let output_values = self.required_output_values(paths)?;
        u32::try_from(paths).map_err(|_| SimulationError::RandomPathNamespaceOverflow {
            paths,
            maximum: u32::MAX,
        })?;
        let final_node = self.grid().node_id(self.grid().len() - 1)?;
        Ok((final_node.get(), output_values))
    }

    fn simulate_one_seeded(
        &self,
        initial_spot: f64,
        path: usize,
        normal_budget: u32,
        stream: RandomStream,
        observations: &mut [f64],
    ) -> Result<(), SimulationError> {
        let path_id =
            u32::try_from(path).map_err(|_| SimulationError::RandomPathNamespaceOverflow {
                paths: path.saturating_add(1),
                maximum: u32::MAX,
            })?;
        let mut generator =
            PhiloxPathStream::new(stream.seed(), stream.domain(), path_id, normal_budget);
        let mut spot = initial_spot;
        let mut observation = 0;
        if self.observation_nodes[0].get() == 0 {
            observations[0] = spot;
            observation = 1;
        }

        for (interval, &step) in self.steps.iter().enumerate() {
            let normal = sample_standard_normal(&mut generator).map_err(|error| match error {
                NormalError::BudgetExhausted => {
                    SimulationError::RandomNormalBudgetExhausted { interval, path }
                }
                NormalError::RejectionLimit => {
                    SimulationError::RandomNormalRejectionLimit { interval, path }
                }
            })?;
            let before = spot;
            let next = step.advance_validated(before, normal);
            if !next.is_finite() {
                return Err(SimulationError::NonFinitePathTransition {
                    from: step.from().get(),
                    to: step.to().get(),
                    path,
                    spot: before,
                    normal,
                });
            }
            spot = next;

            let node = step.to();
            let dividend = self.dividends_by_node[node.get() as usize];
            let terminal_dividend = dividend.filter(|_| self.is_terminal_observation(node));
            if let Some(cash) = terminal_dividend {
                Self::apply_dividend(cash, self.dividend_policy, std::slice::from_mut(&mut spot))?;
            }
            if self.observation_nodes.get(observation).copied() == Some(node) {
                observations[observation] = spot;
                observation += 1;
            }
            if let Some(cash) = dividend.filter(|_| terminal_dividend.is_none()) {
                Self::apply_dividend(cash, self.dividend_policy, std::slice::from_mut(&mut spot))?;
            }
        }
        debug_assert_eq!(observation, observations.len());
        Ok(())
    }

    fn transpose_seeded_parallel(&self, path_major: &[f64], paths: usize) -> Vec<f64> {
        let observations = self.observation_nodes.len();
        let mut time_major = vec![0.0; path_major.len()];
        let transpose_row = |observation: usize, row: &mut [f64]| {
            row.iter_mut().enumerate().for_each(|(path, value)| {
                *value = path_major[path * observations + observation];
            });
        };
        time_major
            .par_chunks_mut(paths)
            .enumerate()
            .for_each(|(observation, row)| transpose_row(observation, row));
        time_major
    }
}

#[cfg(test)]
mod tests {
    use finoptions_core::{
        CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy,
        EventSchedule, LogDiscountCurve, TimeGrid,
    };

    use super::*;
    use crate::CpuExecutor;

    fn plan() -> GbmPathPlan {
        let grid = TimeGrid::try_new([0.0, 0.25, 0.5, 1.0]).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, [0.0, -0.01, -0.02, -0.05]).unwrap();
        let dividend_yield_curve = CumulativeDividendYieldCurve::try_new(&grid, [0.0; 4]).unwrap();
        let dividend = CashDividend::try_new(grid.node_id(2).unwrap(), 3.0).unwrap();
        let exercises = [grid.node_id(1).unwrap(), grid.node_id(2).unwrap()];
        let expiry = grid.node_id(3).unwrap();
        let schedule = EventSchedule::try_new(grid, [dividend], exercises, expiry).unwrap();
        GbmPathPlan::try_new(
            &schedule,
            &curve,
            &dividend_yield_curve,
            ConstantVolatility::try_new(0.20).unwrap(),
            DividendPolicy::Liquidator,
        )
        .unwrap()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn fused_paths_exactly_match_the_existing_kernel_for_the_same_generated_normals() {
        let plan = plan();
        let paths = 257;
        let stream = RandomStream::new(0x01d6_328f_7a4c_915b, 0x1729_0042);
        let mut normals = vec![0.0; plan.intervals() * paths];
        for path in 0..paths {
            let mut generator = PhiloxPathStream::new(
                stream.seed(),
                stream.domain(),
                u32::try_from(path).unwrap(),
                u32::try_from(plan.intervals()).unwrap(),
            );
            for interval in 0..plan.intervals() {
                normals[interval * paths + path] = sample_standard_normal(&mut generator).unwrap();
            }
        }
        let expected = plan.simulate(100.0, paths, &normals).unwrap();
        let observed = plan.simulate_seeded(100.0, paths, stream).unwrap();
        assert_eq!(bits(observed.as_slice()), bits(expected.as_slice()));
    }

    #[test]
    fn worker_count_and_path_batch_size_do_not_change_path_identity() {
        let plan = plan();
        let stream = RandomStream::new(7, 11);
        let serial = plan.simulate_seeded(100.0, 257, stream).unwrap();
        let executor = CpuExecutor::try_new(4).unwrap();
        let parallel = plan
            .simulate_seeded_with(&executor, 100.0, 257, stream)
            .unwrap();
        assert_eq!(bits(serial.as_slice()), bits(parallel.as_slice()));

        let larger = plan.simulate_seeded(100.0, 263, stream).unwrap();
        for row in 0..plan.observation_nodes.len() {
            let serial_row = &serial.as_slice()[row * 257..(row + 1) * 257];
            let larger_row = &larger.as_slice()[row * 263..row * 263 + 257];
            assert_eq!(bits(serial_row), bits(larger_row));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apple_vector_backend_is_thread_invariant_and_close_to_the_scalar_reference() {
        let plan = plan();
        let stream = RandomStream::new(7, 11);
        let scalar = plan.simulate_seeded(100.0, 257, stream).unwrap();
        let serial_executor =
            CpuExecutor::try_new_with_math_backend(1, CpuMathBackend::AppleAccelerate).unwrap();
        let parallel_executor =
            CpuExecutor::try_new_with_math_backend(4, CpuMathBackend::AppleAccelerate).unwrap();
        let apple_serial = plan
            .simulate_seeded_with(&serial_executor, 100.0, 257, stream)
            .unwrap();
        let apple_parallel = plan
            .simulate_seeded_with(&parallel_executor, 100.0, 257, stream)
            .unwrap();
        assert_eq!(
            bits(apple_serial.as_slice()),
            bits(apple_parallel.as_slice())
        );
        for (&reference, &observed) in scalar.as_slice().iter().zip(apple_serial.as_slice()) {
            let scale = reference.abs().max(observed.abs()).max(1.0);
            assert!((reference - observed).abs() <= 16.0 * f64::EPSILON * scale);
        }

        let larger = plan
            .simulate_seeded_with(&parallel_executor, 100.0, 263, stream)
            .unwrap();
        for row in 0..plan.observation_nodes.len() {
            let serial_row = &apple_serial.as_slice()[row * 257..(row + 1) * 257];
            let larger_row = &larger.as_slice()[row * 263..row * 263 + 257];
            assert_eq!(bits(serial_row), bits(larger_row));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apple_vector_backend_reports_transition_failure_without_scalar_substitution() {
        let executor =
            CpuExecutor::try_new_with_math_backend(4, CpuMathBackend::AppleAccelerate).unwrap();
        let error = plan()
            .simulate_seeded_with(&executor, f64::MAX, 257, RandomStream::new(7, 11))
            .unwrap_err();
        assert!(matches!(
            error,
            SimulationError::NonFinitePathTransition { path: 0, .. }
        ));
    }

    #[test]
    fn domains_are_explicitly_distinct_and_zero_is_absorbing() {
        let plan = plan();
        let first = plan
            .simulate_seeded(100.0, 257, RandomStream::new(7, 11))
            .unwrap();
        let second = plan
            .simulate_seeded(100.0, 257, RandomStream::new(7, 12))
            .unwrap();
        assert_ne!(bits(first.as_slice()), bits(second.as_slice()));

        let zero = plan
            .simulate_seeded(0.0, 257, RandomStream::new(7, 11))
            .unwrap();
        assert!(zero.as_slice().iter().all(|value| value.to_bits() == 0));
    }

    #[test]
    fn parallel_overflow_reports_the_deterministic_lowest_path() {
        let plan = plan();
        let executor = CpuExecutor::try_new(4).unwrap();
        let error = plan
            .simulate_seeded_with(&executor, f64::MAX, 257, RandomStream::new(7, 11))
            .unwrap_err();
        assert!(matches!(
            error,
            SimulationError::NonFinitePathTransition { path: 0, .. }
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_path_counts_outside_the_explicit_counter_lane_before_allocation() {
        let plan = plan();
        let paths = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
        assert_eq!(
            plan.validate_seeded(100.0, paths).unwrap_err(),
            SimulationError::RandomPathNamespaceOverflow {
                paths,
                maximum: u32::MAX,
            }
        );
    }
}
