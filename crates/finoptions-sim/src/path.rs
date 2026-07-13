//! Time-major path simulation with explicit event-phase ordering.

use std::sync::atomic::{AtomicUsize, Ordering};

use finoptions_core::{
    CashDividend, ConstantVolatility, DividendPolicy, EventSchedule, LogDiscountCurve, NodeId,
    TimeGrid,
};
use rayon::prelude::*;

use crate::{GbmStep, SimulationError};

/// Immutable plan for GBM paths observed immediately before exercise/expiry decisions.
#[derive(Clone, Debug, PartialEq)]
pub struct GbmPathPlan {
    grid: TimeGrid,
    steps: Box<[GbmStep]>,
    dividends_by_node: Box<[Option<CashDividend>]>,
    observation_nodes: Box<[NodeId]>,
    dividend_policy: DividendPolicy,
}

impl GbmPathPlan {
    /// Precomputes transitions and event lookup for one schedule and market model.
    ///
    /// Observation nodes are every configured exercise node followed by terminal expiry. If an
    /// exercise and dividend share a node, the observed state is the pre-dividend state.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TimeGridMismatch`] unless the curve and schedule own identical
    /// grids, or propagates a transition-construction failure.
    pub fn try_new(
        schedule: &EventSchedule,
        curve: &LogDiscountCurve,
        volatility: ConstantVolatility,
        dividend_policy: DividendPolicy,
    ) -> Result<Self, SimulationError> {
        if schedule.grid() != curve.grid() {
            return Err(SimulationError::TimeGridMismatch);
        }

        let grid = schedule.grid().clone();
        let mut steps = Vec::with_capacity(grid.len() - 1);
        for to_index in 1..grid.len() {
            steps.push(GbmStep::try_new(
                curve,
                grid.node_id(to_index - 1)?,
                grid.node_id(to_index)?,
                volatility,
            )?);
        }

        let mut dividends_by_node = vec![None; grid.len()];
        for &dividend in schedule.dividends() {
            dividends_by_node[dividend.node().get() as usize] = Some(dividend);
        }

        let mut observation_nodes = Vec::with_capacity(schedule.exercise_nodes().len() + 1);
        observation_nodes.extend_from_slice(schedule.exercise_nodes());
        observation_nodes.push(schedule.expiry());

        Ok(Self {
            grid,
            steps: steps.into_boxed_slice(),
            dividends_by_node: dividends_by_node.into_boxed_slice(),
            observation_nodes: observation_nodes.into_boxed_slice(),
            dividend_policy,
        })
    }

    /// Borrows the exact event-aligned model grid.
    #[must_use]
    pub const fn grid(&self) -> &TimeGrid {
        &self.grid
    }

    /// Borrows exercise nodes followed by terminal expiry.
    #[must_use]
    pub const fn observation_nodes(&self) -> &[NodeId] {
        &self.observation_nodes
    }

    /// Returns the number of independent adjacent normal shocks required per path.
    #[must_use]
    pub const fn intervals(&self) -> usize {
        self.steps.len()
    }

    /// Simulates and owns time-major states at every observation node.
    ///
    /// `normals` must have exact shape `[intervals, paths]` in contiguous time-major order.
    ///
    /// # Errors
    ///
    /// Returns a typed shape, input, dividend, or non-finite transition error. The complete normal
    /// buffer is validated before any path computation begins.
    pub fn simulate(
        &self,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        let normal_values = self.required_normal_values(paths)?;
        if normals.len() != normal_values {
            return Err(SimulationError::BufferLengthMismatch {
                buffer: "normal",
                expected: normal_values,
                observed: normals.len(),
            });
        }
        let output_values = self.required_output_values(paths)?;
        let mut current = vec![0.0; paths];
        let mut values = vec![0.0; output_values];
        self.simulate_into(initial_spot, paths, normals, &mut current, &mut values)?;
        Ok(SimulatedPaths {
            observation_nodes: self.observation_nodes.clone(),
            paths,
            values,
        })
    }

    pub(crate) fn simulate_parallel(
        &self,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        let normal_values = self.required_normal_values(paths)?;
        if normals.len() != normal_values {
            return Err(SimulationError::BufferLengthMismatch {
                buffer: "normal",
                expected: normal_values,
                observed: normals.len(),
            });
        }
        let output_values = self.required_output_values(paths)?;
        let mut current = vec![0.0; paths];
        let mut values = vec![0.0; output_values];
        self.validate_execution(initial_spot, paths, normals, &current, &values)?;
        self.simulate_validated(
            initial_spot,
            paths,
            normals,
            &mut current,
            &mut values,
            true,
        )?;
        Ok(SimulatedPaths {
            observation_nodes: self.observation_nodes.clone(),
            paths,
            values,
        })
    }

    /// Simulates into caller-owned reusable workspace and output buffers.
    ///
    /// # Errors
    ///
    /// Returns before mutating either output when an input value or exact buffer shape is invalid.
    /// Transition overflow can occur only during computation and returns the offending node/path.
    pub fn simulate_into(
        &self,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
        current: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), SimulationError> {
        self.validate_execution(initial_spot, paths, normals, current, output)?;

        self.simulate_validated(initial_spot, paths, normals, current, output, false)
    }

    fn simulate_validated(
        &self,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
        current: &mut [f64],
        output: &mut [f64],
        parallel: bool,
    ) -> Result<(), SimulationError> {
        current.fill(initial_spot);
        let mut observation_index = 0;
        if self.observation_nodes[0].get() == 0 {
            output[..paths].copy_from_slice(current);
            observation_index += 1;
        }

        for (interval, &step) in self.steps.iter().enumerate() {
            let normal_start = interval * paths;
            let shocks = &normals[normal_start..normal_start + paths];
            if parallel {
                Self::advance_parallel(step, current, shocks)?;
            } else {
                Self::advance_serial(step, current, shocks)?;
            }

            let node = step.to();
            if self.observation_nodes[observation_index] == node {
                let output_start = observation_index * paths;
                output[output_start..output_start + paths].copy_from_slice(current);
                observation_index += 1;
            }

            if let Some(dividend) = self.dividends_by_node[node.get() as usize] {
                for spot in &mut *current {
                    *spot = dividend.apply(self.dividend_policy, *spot)?;
                }
            }
        }
        debug_assert_eq!(observation_index, self.observation_nodes.len());
        Ok(())
    }

    fn advance_serial(
        step: GbmStep,
        current: &mut [f64],
        shocks: &[f64],
    ) -> Result<(), SimulationError> {
        for (path, (spot, &normal)) in current.iter_mut().zip(shocks).enumerate() {
            let before = *spot;
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
            *spot = next;
        }
        Ok(())
    }

    fn advance_parallel(
        step: GbmStep,
        current: &mut [f64],
        shocks: &[f64],
    ) -> Result<(), SimulationError> {
        let first_failure = AtomicUsize::new(usize::MAX);
        current
            .par_iter_mut()
            .zip(shocks.par_iter())
            .enumerate()
            .for_each(|(path, (spot, &normal))| {
                let next = step.advance_validated(*spot, normal);
                if next.is_finite() {
                    *spot = next;
                } else {
                    first_failure.fetch_min(path, Ordering::Relaxed);
                }
            });

        let path = first_failure.load(Ordering::Relaxed);
        if path != usize::MAX {
            return Err(SimulationError::NonFinitePathTransition {
                from: step.from().get(),
                to: step.to().get(),
                path,
                spot: current[path],
                normal: shocks[path],
            });
        }
        Ok(())
    }

    fn required_normal_values(&self, paths: usize) -> Result<usize, SimulationError> {
        if paths == 0 {
            return Err(SimulationError::EmptyPathSet);
        }
        self.intervals()
            .checked_mul(paths)
            .ok_or(SimulationError::ShapeOverflow {
                rows: self.intervals(),
                paths,
            })
    }

    fn required_output_values(&self, paths: usize) -> Result<usize, SimulationError> {
        if paths == 0 {
            return Err(SimulationError::EmptyPathSet);
        }
        self.observation_nodes
            .len()
            .checked_mul(paths)
            .ok_or(SimulationError::ShapeOverflow {
                rows: self.observation_nodes.len(),
                paths,
            })
    }

    fn validate_execution(
        &self,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
        current: &[f64],
        output: &[f64],
    ) -> Result<(), SimulationError> {
        if !initial_spot.is_finite()
            || initial_spot < 0.0
            || (initial_spot == 0.0 && initial_spot.is_sign_negative())
        {
            return Err(SimulationError::InvalidInitialSpot {
                observed: initial_spot,
            });
        }

        let normal_values = self.required_normal_values(paths)?;
        if normals.len() != normal_values {
            return Err(SimulationError::BufferLengthMismatch {
                buffer: "normal",
                expected: normal_values,
                observed: normals.len(),
            });
        }
        if current.len() != paths {
            return Err(SimulationError::BufferLengthMismatch {
                buffer: "current-state workspace",
                expected: paths,
                observed: current.len(),
            });
        }
        let output_values = self.required_output_values(paths)?;
        if output.len() != output_values {
            return Err(SimulationError::BufferLengthMismatch {
                buffer: "observation output",
                expected: output_values,
                observed: output.len(),
            });
        }
        if let Some((index, &observed)) = normals
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(SimulationError::NonFiniteNormalBuffer {
                interval: index / paths,
                path: index % paths,
                observed,
            });
        }
        Ok(())
    }
}

/// Owned time-major pre-event states at exercise nodes and expiry.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulatedPaths {
    observation_nodes: Box<[NodeId]>,
    paths: usize,
    values: Vec<f64>,
}

impl SimulatedPaths {
    /// Borrows exercise nodes followed by terminal expiry.
    #[must_use]
    pub const fn observation_nodes(&self) -> &[NodeId] {
        &self.observation_nodes
    }

    /// Returns the number of paths in every time-major row.
    #[must_use]
    pub const fn path_count(&self) -> usize {
        self.paths
    }

    /// Borrows the contiguous time-major matrix `[observations, paths]`.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Consumes the result and returns its contiguous time-major values.
    #[must_use]
    pub fn into_values(self) -> Vec<f64> {
        self.values
    }

    /// Borrows all path states at one observation node, if the node is observed.
    #[must_use]
    pub fn values_at(&self, node: NodeId) -> Option<&[f64]> {
        let row = self.observation_nodes.binary_search(&node).ok()?;
        let start = row * self.paths;
        Some(&self.values[start..start + self.paths])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(
        times: &[f64],
        log_discounts: &[f64],
        dividend: Option<(usize, f64)>,
        exercise_nodes: &[usize],
        policy: DividendPolicy,
    ) -> GbmPathPlan {
        let grid = TimeGrid::try_new(times).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
        let dividends = dividend
            .map(|(node, amount)| {
                vec![CashDividend::try_new(grid.node_id(node).unwrap(), amount).unwrap()]
            })
            .unwrap_or_default();
        let exercises = exercise_nodes
            .iter()
            .map(|&node| grid.node_id(node).unwrap())
            .collect::<Vec<_>>();
        let expiry = grid.node_id(grid.len() - 1).unwrap();
        let schedule = EventSchedule::try_new(grid, dividends, exercises, expiry).unwrap();
        GbmPathPlan::try_new(
            &schedule,
            &curve,
            ConstantVolatility::try_new(0.20).unwrap(),
            policy,
        )
        .unwrap()
    }

    fn assert_exact_slice(actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn stores_only_time_major_exercise_and_expiry_states() {
        let plan = plan(
            &[0.0, 0.25, 0.5, 1.0],
            &[0.0, -0.01, -0.02, -0.05],
            None,
            &[1, 2],
            DividendPolicy::Liquidator,
        );
        let normals = [0.0, 1.0, 0.0, -1.0, 0.5, -0.5];
        let simulated = plan.simulate(100.0, 2, &normals).unwrap();
        assert_eq!(
            simulated
                .observation_nodes()
                .iter()
                .map(|node| node.get())
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(simulated.path_count(), 2);
        assert_eq!(simulated.as_slice().len(), 6);

        let curve = LogDiscountCurve::try_new(plan.grid(), [0.0, -0.01, -0.02, -0.05]).unwrap();
        let first_step = GbmStep::try_new(
            &curve,
            plan.grid().node_id(0).unwrap(),
            plan.grid().node_id(1).unwrap(),
            ConstantVolatility::try_new(0.20).unwrap(),
        )
        .unwrap();
        assert_exact_slice(
            simulated
                .values_at(plan.grid().node_id(1).unwrap())
                .unwrap(),
            &[
                first_step.advance(100.0, 0.0).unwrap(),
                first_step.advance(100.0, 1.0).unwrap(),
            ],
        );
    }

    #[test]
    fn observes_exercise_before_applying_same_node_dividend() {
        let plan = plan(
            &[0.0, 0.5, 1.0],
            &[0.0, 0.0, 0.0],
            Some((1, 200.0)),
            &[1],
            DividendPolicy::Liquidator,
        );
        let simulated = plan.simulate(100.0, 1, &[0.0, 0.0]).unwrap();
        let exercise = simulated
            .values_at(plan.grid().node_id(1).unwrap())
            .unwrap()[0];
        let terminal = simulated
            .values_at(plan.grid().node_id(2).unwrap())
            .unwrap()[0];
        assert!(exercise > 0.0);
        assert_eq!(terminal.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn survivor_and_liquidator_are_distinct_below_dividend_boundary() {
        let liquidator = plan(
            &[0.0, 0.5, 1.0],
            &[0.0, 0.0, 0.0],
            Some((1, 200.0)),
            &[],
            DividendPolicy::Liquidator,
        );
        let survivor = plan(
            &[0.0, 0.5, 1.0],
            &[0.0, 0.0, 0.0],
            Some((1, 200.0)),
            &[],
            DividendPolicy::Survivor,
        );
        let liquidator_terminal = liquidator.simulate(100.0, 1, &[0.0, 0.0]).unwrap();
        let survivor_terminal = survivor.simulate(100.0, 1, &[0.0, 0.0]).unwrap();
        assert_eq!(
            liquidator_terminal.as_slice()[0].to_bits(),
            0.0_f64.to_bits()
        );
        assert!(survivor_terminal.as_slice()[0] > 0.0);
    }

    #[test]
    fn adjacent_steps_telescope_to_the_exact_terminal_gbm_solution() {
        let times = [0.0, 0.1, 0.4, 1.0];
        let log_discounts = [0.0, -0.003, -0.014, -0.041];
        let volatility = 0.20;
        let plan = plan(
            &times,
            &log_discounts,
            None,
            &[],
            DividendPolicy::Liquidator,
        );
        let normals = [0.2, -1.1, 0.7, -0.4, 1.3, 0.1];
        let simulated = plan.simulate(83.0, 2, &normals).unwrap();

        for path in 0..2 {
            let diffusion_sum = times
                .windows(2)
                .enumerate()
                .map(|(interval, pair)| {
                    volatility * (pair[1] - pair[0]).sqrt() * normals[interval * 2 + path]
                })
                .sum::<f64>();
            let expected = 83.0
                * (-log_discounts[3] - 0.5 * volatility.powi(2) * times[3] + diffusion_sum).exp();
            let observed = simulated.as_slice()[path];
            assert!((observed - expected).abs() <= 32.0 * f64::EPSILON * expected);
        }
    }

    #[test]
    fn rejects_grid_mismatch() {
        let schedule_grid = TimeGrid::try_new([0.0, 0.5, 1.0]).unwrap();
        let curve_grid = TimeGrid::try_new([0.0, 0.25, 1.0]).unwrap();
        let schedule = EventSchedule::try_new(
            schedule_grid.clone(),
            [],
            [],
            schedule_grid.node_id(2).unwrap(),
        )
        .unwrap();
        let curve = LogDiscountCurve::try_new(&curve_grid, [0.0, -0.01, -0.05]).unwrap();
        assert_eq!(
            GbmPathPlan::try_new(
                &schedule,
                &curve,
                ConstantVolatility::try_new(0.20).unwrap(),
                DividendPolicy::Liquidator,
            )
            .unwrap_err(),
            SimulationError::TimeGridMismatch
        );
    }

    #[test]
    fn validates_all_inputs_before_mutating_workspace_or_output() {
        let plan = plan(
            &[0.0, 0.5, 1.0],
            &[0.0, -0.01, -0.03],
            None,
            &[1],
            DividendPolicy::Liquidator,
        );
        let mut current = [91.0, 92.0];
        let mut output = [81.0, 82.0, 83.0, 84.0];
        let error = plan
            .simulate_into(
                100.0,
                2,
                &[0.0, 0.0, 0.0, f64::NAN],
                &mut current,
                &mut output,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            SimulationError::NonFiniteNormalBuffer {
                interval: 1,
                path: 1,
                ..
            }
        ));
        assert_exact_slice(&current, &[91.0, 92.0]);
        assert_exact_slice(&output, &[81.0, 82.0, 83.0, 84.0]);
    }
}
