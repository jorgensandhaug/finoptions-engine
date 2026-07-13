//! Fused path-local random generation and event-aligned GBM evolution.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::{GbmPathPlan, SimulatedPaths};
use crate::random::{NormalError, PhiloxPathStream, sample_standard_normal};
use crate::{RandomStream, SimulationError};

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
        let mut path_major = vec![0.0; output_values];
        for (path, observations) in path_major
            .chunks_mut(self.observation_nodes.len())
            .enumerate()
        {
            self.simulate_one_seeded(initial_spot, path, normal_budget, stream, observations)?;
        }
        let values = self.transpose_seeded(&path_major, paths, false);
        Ok(SimulatedPaths {
            observation_nodes: self.observation_nodes.clone(),
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

        let values = self.transpose_seeded(&path_major, paths, true);
        Ok(SimulatedPaths {
            observation_nodes: self.observation_nodes.clone(),
            paths,
            values,
        })
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
        let final_node = self.grid.node_id(self.grid.len() - 1)?;
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
            if self.observation_nodes[observation] == node {
                observations[observation] = spot;
                observation += 1;
            }
            if let Some(dividend) = self.dividends_by_node[node.get() as usize] {
                spot = dividend.apply(self.dividend_policy, spot)?;
            }
        }
        debug_assert_eq!(observation, observations.len());
        Ok(())
    }

    fn transpose_seeded(&self, path_major: &[f64], paths: usize, parallel: bool) -> Vec<f64> {
        let observations = self.observation_nodes.len();
        let mut time_major = vec![0.0; path_major.len()];
        let transpose_row = |observation: usize, row: &mut [f64]| {
            row.iter_mut().enumerate().for_each(|(path, value)| {
                *value = path_major[path * observations + observation];
            });
        };
        if parallel {
            time_major
                .par_chunks_mut(paths)
                .enumerate()
                .for_each(|(observation, row)| transpose_row(observation, row));
        } else {
            time_major
                .chunks_mut(paths)
                .enumerate()
                .for_each(|(observation, row)| transpose_row(observation, row));
        }
        time_major
    }
}

#[cfg(test)]
mod tests {
    use finoptions_core::{
        CashDividend, ConstantVolatility, DividendPolicy, EventSchedule, LogDiscountCurve, TimeGrid,
    };

    use super::*;
    use crate::CpuExecutor;

    fn plan() -> GbmPathPlan {
        let grid = TimeGrid::try_new([0.0, 0.25, 0.5, 1.0]).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, [0.0, -0.01, -0.02, -0.05]).unwrap();
        let dividend = CashDividend::try_new(grid.node_id(2).unwrap(), 3.0).unwrap();
        let exercises = [grid.node_id(1).unwrap(), grid.node_id(2).unwrap()];
        let expiry = grid.node_id(3).unwrap();
        let schedule = EventSchedule::try_new(grid, [dividend], exercises, expiry).unwrap();
        GbmPathPlan::try_new(
            &schedule,
            &curve,
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
        let parallel = executor
            .simulate_gbm_seeded(&plan, 100.0, 257, stream)
            .unwrap();
        assert_eq!(bits(serial.as_slice()), bits(parallel.as_slice()));

        let larger = plan.simulate_seeded(100.0, 263, stream).unwrap();
        for row in 0..plan.observation_nodes.len() {
            let serial_row = &serial.as_slice()[row * 257..(row + 1) * 257];
            let larger_row = &larger.as_slice()[row * 263..row * 263 + 257];
            assert_eq!(bits(serial_row), bits(larger_row));
        }
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
        let error = executor
            .simulate_gbm_seeded(&plan, f64::MAX, 257, RandomStream::new(7, 11))
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
