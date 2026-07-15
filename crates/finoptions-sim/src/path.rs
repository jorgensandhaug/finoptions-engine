//! Time-major path simulation with explicit event-phase ordering.

mod seeded;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve, NodeId, TimeGrid,
};
use rayon::prelude::*;

use crate::{CpuMathBackend, GbmStep, RandomStream, SimulationError};

/// Immutable plan for GBM paths observed before exercise and after any terminal cash jump.
#[derive(Clone, Debug, PartialEq)]
pub struct GbmPathPlan {
    curve: LogDiscountCurve,
    dividend_yield_curve: CumulativeDividendYieldCurve,
    steps: Box<[GbmStep]>,
    dividends_by_node: Box<[Option<CashDividend>]>,
    observation_nodes: Box<[NodeId]>,
    dividend_policy: DividendPolicy,
    model_identity: GbmModelIdentity,
}

/// Exact model identity carried by path populations and fitted policies.
#[derive(Clone, Debug, PartialEq)]
enum GbmModelIdentity {
    /// The GBM state is the observed stock, including explicit post-observation cash jumps.
    Spot,
    /// Observations were reconstructed from a risky prepaid forward and deterministic cash.
    Escrowed { prepaid_cash_by_node: Arc<[f64]> },
}

/// Escrowed approximation over an unchanged hot-path [`GbmPathPlan`] for the risky component.
#[derive(Clone, Debug, PartialEq)]
pub struct EscrowedGbmPathPlan {
    risky_prepaid_forward: GbmPathPlan,
    prepaid_cash_by_node: Arc<[f64]>,
}

impl GbmPathPlan {
    /// Precomputes transitions and event lookup for one schedule and market model.
    ///
    /// Observation nodes are every configured exercise node followed by terminal expiry. If an
    /// exercise and dividend share a nonterminal node, the observed state is pre-dividend. A cash
    /// jump at expiry is applied before the terminal state is observed.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TimeGridMismatch`] unless the curve and schedule own identical
    /// grids, or propagates a transition-construction failure.
    pub fn try_new(
        schedule: &EventSchedule,
        curve: &LogDiscountCurve,
        dividend_yield_curve: &CumulativeDividendYieldCurve,
        volatility: ConstantVolatility,
        dividend_policy: DividendPolicy,
    ) -> Result<Self, SimulationError> {
        if schedule.grid() != curve.grid() {
            return Err(SimulationError::TimeGridMismatch);
        }
        if schedule.grid() != dividend_yield_curve.grid() {
            return Err(SimulationError::DividendYieldTimeGridMismatch);
        }

        let grid = schedule.grid();
        let mut steps = Vec::with_capacity(grid.len() - 1);
        for to_index in 1..grid.len() {
            steps.push(GbmStep::try_new(
                curve,
                dividend_yield_curve,
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
            curve: curve.clone(),
            dividend_yield_curve: dividend_yield_curve.clone(),
            steps: steps.into_boxed_slice(),
            dividends_by_node: dividends_by_node.into_boxed_slice(),
            observation_nodes: observation_nodes.into_boxed_slice(),
            dividend_policy,
            model_identity: GbmModelIdentity::Spot,
        })
    }

    /// Borrows the exact event-aligned model grid.
    #[must_use]
    pub const fn grid(&self) -> &TimeGrid {
        self.curve.grid()
    }

    /// Borrows the exact discount curve used to construct every risk-neutral transition.
    #[must_use]
    pub const fn curve(&self) -> &LogDiscountCurve {
        &self.curve
    }

    /// Borrows the exact cumulative proportional-dividend yield used by every transition.
    #[must_use]
    pub const fn dividend_yield_curve(&self) -> &CumulativeDividendYieldCurve {
        &self.dividend_yield_curve
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
            plan: self.clone(),
            random_stream: None,
            initial_spot,
            paths,
            values,
        })
    }

    /// Simulates external shocks with one explicitly named GBM exponential implementation.
    ///
    /// This serial entry point exists for outer-parallel callers such as contract batches. The
    /// backend is never inferred or replaced. [`CpuMathBackend::RustStdF64`] preserves the scalar
    /// kernel, while vector backends use an interval-major vector kernel.
    ///
    /// # Errors
    ///
    /// Returns a typed shape, input, backend, dividend, or non-finite transition failure.
    pub fn simulate_with_math_backend(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        match math_backend {
            CpuMathBackend::RustStdF64 => self.simulate(initial_spot, paths, normals),
            CpuMathBackend::AppleAccelerate => {
                self.simulate_vector_exp(math_backend, initial_spot, paths, normals, false)
            }
        }
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
            plan: self.clone(),
            random_stream: None,
            initial_spot,
            paths,
            values,
        })
    }

    pub(crate) fn simulate_parallel_with_math_backend(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        match math_backend {
            CpuMathBackend::RustStdF64 => self.simulate_parallel(initial_spot, paths, normals),
            CpuMathBackend::AppleAccelerate => {
                self.simulate_vector_exp(math_backend, initial_spot, paths, normals, true)
            }
        }
    }

    fn simulate_vector_exp(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
        parallel: bool,
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
        current.fill(initial_spot);
        let mut exponents = vec![0.0; paths];
        let mut growth = vec![0.0; paths];
        let mut observation_index = 0;
        if self.observation_nodes[0].get() == 0 {
            values[..paths].copy_from_slice(&current);
            observation_index += 1;
        }

        for (interval, &step) in self.steps.iter().enumerate() {
            let normal_start = interval * paths;
            let shocks = &normals[normal_start..normal_start + paths];
            if parallel {
                exponents
                    .par_iter_mut()
                    .zip(current.par_iter())
                    .zip(shocks.par_iter())
                    .for_each(|((exponent, &spot), &normal)| {
                        *exponent = if spot == 0.0 {
                            0.0
                        } else {
                            step.diffusion().mul_add(normal, step.log_drift())
                        };
                    });
            } else {
                for ((exponent, &spot), &normal) in exponents.iter_mut().zip(&current).zip(shocks) {
                    *exponent = if spot == 0.0 {
                        0.0
                    } else {
                        step.diffusion().mul_add(normal, step.log_drift())
                    };
                }
            }
            math_backend.exp_f64(&exponents, &mut growth)?;
            if parallel {
                Self::multiply_growth_parallel(step, &mut current, &growth, shocks)?;
            } else {
                for (path, ((spot, &factor), &normal)) in
                    current.iter_mut().zip(&growth).zip(shocks).enumerate()
                {
                    let before = *spot;
                    let next = before * factor;
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
            }

            self.process_time_major_node(
                step.to(),
                &mut current,
                &mut values,
                &mut observation_index,
                paths,
            )?;
        }
        debug_assert_eq!(observation_index, self.observation_nodes.len());
        Ok(SimulatedPaths {
            plan: self.clone(),
            random_stream: None,
            initial_spot,
            paths,
            values,
        })
    }

    fn multiply_growth_parallel(
        step: GbmStep,
        current: &mut [f64],
        growth: &[f64],
        shocks: &[f64],
    ) -> Result<(), SimulationError> {
        let first_failure = AtomicUsize::new(usize::MAX);
        current
            .par_iter_mut()
            .zip(growth.par_iter())
            .enumerate()
            .for_each(|(path, (spot, &factor))| {
                let next = *spot * factor;
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

            self.process_time_major_node(
                step.to(),
                current,
                output,
                &mut observation_index,
                paths,
            )?;
        }
        debug_assert_eq!(observation_index, self.observation_nodes.len());
        Ok(())
    }

    #[inline]
    fn process_time_major_node(
        &self,
        node: NodeId,
        current: &mut [f64],
        output: &mut [f64],
        observation: &mut usize,
        paths: usize,
    ) -> Result<(), SimulationError> {
        match self.dividends_by_node[node.get() as usize] {
            Some(dividend) if self.is_terminal_observation(node) => {
                Self::apply_dividend(dividend, self.dividend_policy, current)?;
                self.record_time_major_observation(node, current, output, observation, paths);
            }
            Some(dividend) => {
                self.record_time_major_observation(node, current, output, observation, paths);
                Self::apply_dividend(dividend, self.dividend_policy, current)?;
            }
            None => {
                self.record_time_major_observation(node, current, output, observation, paths);
            }
        }
        Ok(())
    }

    #[inline]
    fn record_time_major_observation(
        &self,
        node: NodeId,
        current: &[f64],
        output: &mut [f64],
        observation: &mut usize,
        paths: usize,
    ) {
        if self.observation_nodes.get(*observation).copied() == Some(node) {
            let output_start = *observation * paths;
            output[output_start..output_start + paths].copy_from_slice(current);
            *observation += 1;
        }
    }

    #[inline]
    fn apply_dividend(
        dividend: CashDividend,
        policy: DividendPolicy,
        current: &mut [f64],
    ) -> Result<(), SimulationError> {
        for spot in current {
            *spot = dividend.apply(policy, *spot)?;
        }
        Ok(())
    }

    #[inline]
    fn is_terminal_observation(&self, node: NodeId) -> bool {
        self.observation_nodes.last().copied() == Some(node)
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
        Self::validate_initial_spot(initial_spot)?;

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

    fn validate_initial_spot(initial_spot: f64) -> Result<(), SimulationError> {
        if !initial_spot.is_finite()
            || initial_spot < 0.0
            || (initial_spot == 0.0 && initial_spot.is_sign_negative())
        {
            return Err(SimulationError::InvalidInitialSpot {
                observed: initial_spot,
            });
        }
        Ok(())
    }
}

impl EscrowedGbmPathPlan {
    /// Constructs the escrowed cash-dividend approximation on an exact event-aligned grid.
    ///
    /// The supplied volatility belongs to the risky prepaid-forward component. Converting an
    /// observed spot volatility, when desired, is a separate explicit financial-model decision.
    ///
    /// # Errors
    ///
    /// Returns a typed grid, curve, transition, or deterministic-cash valuation failure. The
    /// original schedule is never shifted, merged, or otherwise repaired.
    pub fn try_new(
        schedule: &EventSchedule,
        curve: &LogDiscountCurve,
        volatility: ConstantVolatility,
    ) -> Result<Self, SimulationError> {
        if schedule.grid() != curve.grid() {
            return Err(SimulationError::TimeGridMismatch);
        }

        let prepaid_cash_by_node: Arc<[f64]> = Arc::from(escrowed_prepaid_cash(schedule, curve)?);
        let zero_yield_curve = CumulativeDividendYieldCurve::try_new(
            schedule.grid(),
            vec![0.0; schedule.grid().len()],
        )?;
        let risky_schedule = EventSchedule::try_new(
            schedule.grid().clone(),
            Vec::<CashDividend>::new(),
            schedule.exercise_nodes().to_vec(),
            schedule.expiry(),
        )?;
        let mut risky_prepaid_forward = GbmPathPlan::try_new(
            &risky_schedule,
            curve,
            &zero_yield_curve,
            volatility,
            DividendPolicy::Liquidator,
        )?;
        risky_prepaid_forward.model_identity = GbmModelIdentity::Escrowed {
            prepaid_cash_by_node: Arc::clone(&prepaid_cash_by_node),
        };
        Ok(Self {
            risky_prepaid_forward,
            prepaid_cash_by_node,
        })
    }

    /// Borrows the exact event-aligned model grid.
    #[must_use]
    pub const fn grid(&self) -> &TimeGrid {
        self.risky_prepaid_forward.grid()
    }

    /// Borrows the exact discount curve used by the risky component and deterministic cash.
    #[must_use]
    pub const fn curve(&self) -> &LogDiscountCurve {
        self.risky_prepaid_forward.curve()
    }

    /// Borrows exercise nodes followed by terminal expiry.
    #[must_use]
    pub const fn observation_nodes(&self) -> &[NodeId] {
        self.risky_prepaid_forward.observation_nodes()
    }

    /// Simulates serial paths and reconstructs pre-event exercise states and post-event expiry.
    ///
    /// # Errors
    ///
    /// Returns a typed initial-state, random-input, transition, or reconstruction failure. A
    /// nonpositive origin prepaid forward is rejected before path allocation or simulation.
    pub fn simulate(
        &self,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
        let simulated =
            self.risky_prepaid_forward
                .simulate(initial_prepaid_forward, paths, normals)?;
        self.reconstruct_observed_stock(initial_spot, simulated)
    }

    /// Simulates serial escrowed paths with an explicit GBM exponential implementation.
    ///
    /// # Errors
    ///
    /// Returns a typed initial-state, backend, transition, or reconstruction failure. The named
    /// implementation is never replaced.
    pub fn simulate_with_math_backend(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
        let simulated = self.risky_prepaid_forward.simulate_with_math_backend(
            math_backend,
            initial_prepaid_forward,
            paths,
            normals,
        )?;
        self.reconstruct_observed_stock(initial_spot, simulated)
    }

    /// Generates serial seeded paths and reconstructs exercise and terminal stock observations.
    ///
    /// # Errors
    ///
    /// Returns a typed initial-state, random-generation, transition, or reconstruction failure.
    pub fn simulate_seeded(
        &self,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
        let simulated =
            self.risky_prepaid_forward
                .simulate_seeded(initial_prepaid_forward, paths, stream)?;
        self.reconstruct_observed_stock(initial_spot, simulated)
    }

    /// Generates serial escrowed paths with an explicit GBM exponential implementation.
    ///
    /// # Errors
    ///
    /// Returns a typed initial-state, random-generation, backend, transition, or reconstruction
    /// failure. The named implementation is never replaced.
    pub fn simulate_seeded_with_math_backend(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
        let simulated = self
            .risky_prepaid_forward
            .simulate_seeded_with_math_backend(
                math_backend,
                initial_prepaid_forward,
                paths,
                stream,
            )?;
        self.reconstruct_observed_stock(initial_spot, simulated)
    }

    pub(crate) fn simulate_parallel(
        &self,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
        let simulated = self.risky_prepaid_forward.simulate_parallel(
            initial_prepaid_forward,
            paths,
            normals,
        )?;
        self.reconstruct_observed_stock(initial_spot, simulated)
    }

    pub(crate) fn simulate_parallel_with_math_backend(
        &self,
        math_backend: CpuMathBackend,
        initial_spot: f64,
        paths: usize,
        normals: &[f64],
    ) -> Result<SimulatedPaths, SimulationError> {
        match math_backend {
            CpuMathBackend::RustStdF64 => self.simulate_parallel(initial_spot, paths, normals),
            CpuMathBackend::AppleAccelerate => {
                let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
                let simulated = self
                    .risky_prepaid_forward
                    .simulate_parallel_with_math_backend(
                        math_backend,
                        initial_prepaid_forward,
                        paths,
                        normals,
                    )?;
                self.reconstruct_observed_stock(initial_spot, simulated)
            }
        }
    }

    pub(crate) fn simulate_seeded_parallel(
        &self,
        initial_spot: f64,
        paths: usize,
        stream: RandomStream,
    ) -> Result<SimulatedPaths, SimulationError> {
        let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
        let simulated = self.risky_prepaid_forward.simulate_seeded_parallel(
            initial_prepaid_forward,
            paths,
            stream,
        )?;
        self.reconstruct_observed_stock(initial_spot, simulated)
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
                let initial_prepaid_forward = self.initial_prepaid_forward(initial_spot)?;
                let simulated = self
                    .risky_prepaid_forward
                    .simulate_seeded_parallel_with_math_backend(
                        math_backend,
                        initial_prepaid_forward,
                        paths,
                        stream,
                    )?;
                self.reconstruct_observed_stock(initial_spot, simulated)
            }
        }
    }

    fn initial_prepaid_forward(&self, initial_spot: f64) -> Result<f64, SimulationError> {
        GbmPathPlan::validate_initial_spot(initial_spot)?;
        let prepaid_cash = self.prepaid_cash_by_node[0];
        let initial_prepaid_forward = initial_spot - prepaid_cash;
        if !initial_prepaid_forward.is_finite() || initial_prepaid_forward <= 0.0 {
            return Err(SimulationError::NonPositiveEscrowedPrepaidForward {
                initial_spot,
                prepaid_cash,
            });
        }
        Ok(initial_prepaid_forward)
    }

    fn reconstruct_observed_stock(
        &self,
        initial_spot: f64,
        mut simulated: SimulatedPaths,
    ) -> Result<SimulatedPaths, SimulationError> {
        for (observation, node) in self.observation_nodes().iter().copied().enumerate() {
            let start = observation * simulated.paths;
            let row = &mut simulated.values[start..start + simulated.paths];
            if node.get() == 0 {
                row.fill(initial_spot);
                continue;
            }

            let prepaid_cash = if node.get() as usize == self.grid().len() - 1 {
                0.0
            } else {
                self.prepaid_cash_by_node[node.get() as usize]
            };
            for (path, observed) in row.iter_mut().enumerate() {
                let prepaid_forward = *observed;
                let stock = prepaid_forward + prepaid_cash;
                if !stock.is_finite() {
                    return Err(SimulationError::NonFiniteEscrowedObservation {
                        node: node.get(),
                        path,
                        prepaid_forward,
                        prepaid_cash,
                    });
                }
                *observed = stock;
            }
        }
        simulated.initial_spot = initial_spot;
        Ok(simulated)
    }
}

fn escrowed_prepaid_cash(
    schedule: &EventSchedule,
    curve: &LogDiscountCurve,
) -> Result<Box<[f64]>, SimulationError> {
    let grid = schedule.grid();
    let mut prepaid = vec![0.0; grid.len()];
    let mut event = schedule.dividends().len();
    let expiry = grid.len() - 1;
    if event > 0 && schedule.dividends()[event - 1].node().get() as usize == expiry {
        event -= 1;
        prepaid[expiry] = schedule.dividends()[event].amount();
    }
    for node in (0..grid.len() - 1).rev() {
        let interval_log_discount = curve.as_slice()[node + 1] - curve.as_slice()[node];
        let discounted_future = prepaid[node + 1] * interval_log_discount.exp();
        let same_node_cash =
            if event > 0 && schedule.dividends()[event - 1].node().get() as usize == node {
                event -= 1;
                schedule.dividends()[event].amount()
            } else {
                0.0
            };
        let value = discounted_future + same_node_cash;
        if !value.is_finite() {
            return Err(SimulationError::NonFiniteEscrowedPrepaidCash {
                node: u32::try_from(node).unwrap_or(u32::MAX),
                discounted_future,
                same_node_cash,
            });
        }
        prepaid[node] = value;
    }
    debug_assert_eq!(event, 0);
    let mut origin = 0.0;
    for dividend in schedule.dividends() {
        let node = dividend.node().get() as usize;
        origin += dividend.amount() * curve.as_slice()[node].exp();
        if !origin.is_finite() {
            return Err(SimulationError::NonFiniteEscrowedPrepaidCash {
                node: 0,
                discounted_future: origin,
                same_node_cash: 0.0,
            });
        }
    }
    prepaid[0] = origin;
    Ok(prepaid.into_boxed_slice())
}

/// Owned time-major pre-event exercise states followed by post-event terminal expiry.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulatedPaths {
    plan: GbmPathPlan,
    random_stream: Option<RandomStream>,
    initial_spot: f64,
    paths: usize,
    values: Vec<f64>,
}

impl SimulatedPaths {
    /// Borrows the complete immutable GBM plan that generated this population.
    #[must_use]
    pub const fn path_plan(&self) -> &GbmPathPlan {
        &self.plan
    }

    /// Borrows the exact model grid that defines every observation node.
    #[must_use]
    pub const fn grid(&self) -> &TimeGrid {
        self.plan.grid()
    }

    /// Borrows the exact discount curve that generated the risk-neutral path dynamics.
    #[must_use]
    pub const fn curve(&self) -> &LogDiscountCurve {
        self.plan.curve()
    }

    /// Borrows the exact cumulative proportional-dividend yield used by the path dynamics.
    #[must_use]
    pub const fn dividend_yield_curve(&self) -> &CumulativeDividendYieldCurve {
        self.plan.dividend_yield_curve()
    }

    /// Returns the exact seeded namespace, or `None` for externally supplied normal variates.
    #[must_use]
    pub const fn random_stream(&self) -> Option<RandomStream> {
        self.random_stream
    }

    /// Returns the exact common pre-event state at the valuation origin.
    #[must_use]
    pub const fn initial_spot(&self) -> f64 {
        self.initial_spot
    }

    /// Borrows exercise nodes followed by terminal expiry.
    #[must_use]
    pub const fn observation_nodes(&self) -> &[NodeId] {
        self.plan.observation_nodes()
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
        let row = self.plan.observation_nodes.binary_search(&node).ok()?;
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
        let dividend_yield_curve =
            CumulativeDividendYieldCurve::try_new(&grid, vec![0.0; grid.len()]).unwrap();
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
            &dividend_yield_curve,
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

    fn escrowed_plan(
        times: &[f64],
        rate: f64,
        dividends: &[(usize, f64)],
        exercise_nodes: &[usize],
        volatility: f64,
    ) -> EscrowedGbmPathPlan {
        let grid = TimeGrid::try_new(times).unwrap();
        let log_discounts = times
            .iter()
            .enumerate()
            .map(|(node, &time)| if node == 0 { 0.0 } else { -rate * time })
            .collect::<Vec<_>>();
        let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
        let dividends = dividends
            .iter()
            .map(|&(node, amount)| {
                CashDividend::try_new(grid.node_id(node).unwrap(), amount).unwrap()
            })
            .collect::<Vec<_>>();
        let exercises = exercise_nodes
            .iter()
            .map(|&node| grid.node_id(node).unwrap())
            .collect::<Vec<_>>();
        let expiry = grid.node_id(grid.len() - 1).unwrap();
        let schedule = EventSchedule::try_new(grid, dividends, exercises, expiry).unwrap();
        EscrowedGbmPathPlan::try_new(
            &schedule,
            &curve,
            ConstantVolatility::try_new(volatility).unwrap(),
        )
        .unwrap()
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
            plan.dividend_yield_curve(),
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
    fn applies_an_expiry_cash_jump_before_terminal_observation() {
        let terminal_dividend_plan = plan(
            &[0.0, 1.0],
            &[0.0, 0.0],
            Some((1, 10.0)),
            &[],
            DividendPolicy::Liquidator,
        );
        let simulated = terminal_dividend_plan.simulate(100.0, 1, &[0.0]).unwrap();
        let terminal = 100.0 * (-0.5_f64 * 0.20_f64 * 0.20_f64).exp() - 10.0;
        assert_exact_slice(simulated.as_slice(), &[terminal]);

        let seeded = terminal_dividend_plan
            .simulate_seeded(100.0, 1, RandomStream::new(7, 11))
            .unwrap();
        let no_dividend = plan(
            &[0.0, 1.0],
            &[0.0, 0.0],
            None,
            &[],
            DividendPolicy::Liquidator,
        )
        .simulate_seeded(100.0, 1, RandomStream::new(7, 11))
        .unwrap();
        assert_exact_slice(
            seeded.as_slice(),
            &[(no_dividend.as_slice()[0] - 10.0).max(0.0)],
        );
    }

    #[test]
    fn escrowed_paths_observe_pre_event_stock_and_keep_the_prepaid_forward_continuous() {
        let rate = 0.05;
        let volatility = 0.20;
        let dividend = 10.0;
        let plan = escrowed_plan(
            &[0.0, 0.5, 1.0],
            rate,
            &[(1, dividend)],
            &[0, 1],
            volatility,
        );
        let initial_spot = 100.0;
        let prepaid_cash = dividend * (-rate * 0.5_f64).exp();
        let initial_prepaid_forward = initial_spot - prepaid_cash;
        let half_step_growth = ((rate - 0.5 * volatility * volatility) * 0.5_f64).exp();
        let event_prepaid_forward = initial_prepaid_forward * half_step_growth;
        let terminal_prepaid_forward = event_prepaid_forward * half_step_growth;
        let paths = plan.simulate(initial_spot, 1, &[0.0, 0.0]).unwrap();
        assert_exact_slice(
            paths.as_slice(),
            &[
                initial_spot,
                event_prepaid_forward + dividend,
                terminal_prepaid_forward,
            ],
        );
    }

    #[test]
    fn escrowed_expiry_observation_excludes_paid_terminal_cash() {
        let rate = 0.0;
        let volatility = 0.20;
        let dividend = 10.0;
        let plan = escrowed_plan(&[0.0, 1.0], rate, &[(1, dividend)], &[], volatility);
        let initial_prepaid_forward = 100.0 - dividend;
        let growth = ((rate - 0.5 * volatility * volatility) * 1.0_f64).exp();
        let paths = plan.simulate(100.0, 1, &[0.0]).unwrap();
        assert_exact_slice(paths.as_slice(), &[initial_prepaid_forward * growth]);
    }

    #[test]
    fn escrowed_plan_rejects_a_nonpositive_prepaid_forward_before_path_work() {
        let plan = escrowed_plan(&[0.0, 0.5, 1.0], 0.0, &[(1, 100.0)], &[], 0.20);
        assert_eq!(
            plan.simulate(100.0, 1, &[0.0, 0.0]).unwrap_err(),
            SimulationError::NonPositiveEscrowedPrepaidForward {
                initial_spot: 100.0,
                prepaid_cash: 100.0,
            }
        );
    }

    #[test]
    fn escrowed_terminal_payoffs_converge_to_their_own_european_closed_form() {
        let rate = 0.03;
        let volatility = 0.25;
        let maturity = 1.0;
        let initial_spot = 100.0;
        let strike = 100.0;
        let dividends = [(1, 2.0), (2, 2.0)];
        let times = [0.0, 0.4, 0.8, maturity];
        let plan = escrowed_plan(&times, rate, &dividends, &[], volatility);
        let path_count = 262_144_u32;
        let path_count_usize = usize::try_from(path_count).unwrap();
        let simulated = plan
            .simulate_seeded(
                initial_spot,
                path_count_usize,
                RandomStream::new(0x0f31_7a11_5eec_2026, 0x51c0_0001),
            )
            .unwrap();
        let discount = (-rate * maturity).exp();
        let discounted_payoffs = simulated
            .as_slice()
            .iter()
            .map(|&terminal| (terminal - strike).max(0.0) * discount)
            .collect::<Vec<_>>();
        let mean = discounted_payoffs.iter().sum::<f64>() / f64::from(path_count);
        let sum_squared = discounted_payoffs
            .iter()
            .map(|payoff| (payoff - mean).powi(2))
            .sum::<f64>();
        let standard_error =
            (sum_squared / f64::from(path_count - 1) / f64::from(path_count)).sqrt();
        let prepaid_cash = 2.0 * (-rate * 0.4_f64).exp() + 2.0 * (-rate * 0.8_f64).exp();
        let prepaid_forward = initial_spot - prepaid_cash;
        let expected = black_scholes_call(prepaid_forward, strike, rate, volatility, maturity);
        assert!(
            (mean - expected).abs() <= 6.0 * standard_error + 1.0e-4,
            "mean={mean}, expected={expected}, standard_error={standard_error}"
        );
    }

    fn black_scholes_call(spot: f64, strike: f64, rate: f64, volatility: f64, time: f64) -> f64 {
        let standard_deviation = volatility * time.sqrt();
        let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * time)
            / standard_deviation;
        let d2 = d1 - standard_deviation;
        spot * standard_normal_cdf(d1) - strike * (-rate * time).exp() * standard_normal_cdf(d2)
    }

    fn standard_normal_cdf(value: f64) -> f64 {
        let absolute = value.abs();
        let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
        let polynomial =
            (((((1.330_274_429 * t - 1.821_255_978) * t) + 1.781_477_937) * t - 0.356_563_782) * t
                + 0.319_381_530)
                * t;
        let positive = 1.0
            - (-0.5 * absolute * absolute).exp() / (2.0 * std::f64::consts::PI).sqrt() * polynomial;
        if value >= 0.0 {
            positive
        } else {
            1.0 - positive
        }
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
    fn cumulative_dividend_yield_telescopes_separately_from_risk_free_discounting() {
        let times = [0.0, 0.2, 0.7, 1.0];
        let log_discounts = [0.0, -0.009, -0.031, -0.047];
        let cumulative_yields = [0.0, 0.004, 0.018, 0.025];
        let grid = TimeGrid::try_new(times).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
        let dividend_yield_curve =
            CumulativeDividendYieldCurve::try_new(&grid, cumulative_yields).unwrap();
        let expiry = grid.node_id(grid.len() - 1).unwrap();
        let schedule = EventSchedule::try_new(grid, [], [], expiry).unwrap();
        let volatility = ConstantVolatility::try_new(0.20).unwrap();
        let plan = GbmPathPlan::try_new(
            &schedule,
            &curve,
            &dividend_yield_curve,
            volatility,
            DividendPolicy::Liquidator,
        )
        .unwrap();
        let normals = [0.2, -1.1, 0.7];
        let observed = plan.simulate(83.0, 1, &normals).unwrap().as_slice()[0];
        let diffusion_sum = times
            .windows(2)
            .enumerate()
            .map(|(interval, pair)| {
                volatility.get() * (pair[1] - pair[0]).sqrt() * normals[interval]
            })
            .sum::<f64>();
        let expected = 83.0
            * (-log_discounts[3]
                - cumulative_yields[3]
                - 0.5 * volatility.get().powi(2) * times[3]
                + diffusion_sum)
                .exp();
        assert!((observed - expected).abs() <= 32.0 * f64::EPSILON * expected);
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
        let schedule_yield_curve =
            CumulativeDividendYieldCurve::try_new(&schedule_grid, [0.0; 3]).unwrap();
        assert_eq!(
            GbmPathPlan::try_new(
                &schedule,
                &curve,
                &schedule_yield_curve,
                ConstantVolatility::try_new(0.20).unwrap(),
                DividendPolicy::Liquidator,
            )
            .unwrap_err(),
            SimulationError::TimeGridMismatch
        );

        let schedule_curve =
            LogDiscountCurve::try_new(&schedule_grid, [0.0, -0.02, -0.05]).unwrap();
        let mismatched_yield_curve =
            CumulativeDividendYieldCurve::try_new(&curve_grid, [0.0; 3]).unwrap();
        assert_eq!(
            GbmPathPlan::try_new(
                &schedule,
                &schedule_curve,
                &mismatched_yield_curve,
                ConstantVolatility::try_new(0.20).unwrap(),
                DividendPolicy::Liquidator,
            )
            .unwrap_err(),
            SimulationError::DividendYieldTimeGridMismatch
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
