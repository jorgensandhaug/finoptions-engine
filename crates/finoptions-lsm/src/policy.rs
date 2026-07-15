//! Fixed-policy construction and independent path evaluation for call options.

use finoptions_core::{LogDiscountCurve, NodeId};
use finoptions_sim::{
    CpuExecutor, CpuMathBackend, EscrowedGbmPathPlan, GbmPathPlan, RandomStream, SimulatedPaths,
};

use crate::regression::{PolynomialModel, RegressionWorkspace};
use crate::{LsmError, PolynomialRegressionSpec, RegressionDiagnostics};

/// Explicit rule when immediate exercise and fitted continuation compare equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExerciseTieRule {
    /// Exercise only when intrinsic value is strictly larger than fitted continuation.
    StrictlyGreater,
    /// Exercise when intrinsic value is greater than or equal to fitted continuation.
    GreaterOrEqual,
}

impl ExerciseTieRule {
    fn should_exercise(self, intrinsic: f64, continuation: f64) -> bool {
        match self {
            Self::StrictlyGreater => intrinsic > continuation,
            Self::GreaterOrEqual => intrinsic >= continuation,
        }
    }
}

/// One exercise node's immutable policy-fitting diagnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExerciseFitDiagnostics {
    /// A common valuation-origin decision was fitted from discounted training cashflows.
    OriginDecision {
        /// Whether every holder following the policy exercises immediately.
        exercise: bool,
        /// Training estimate of continuation value at valuation.
        estimated_continuation: f64,
    },
    /// Training contained no in-the-money state at this node.
    NoInTheMoneyPaths,
    /// A continuation model was fitted on the declared in-the-money population.
    Regression {
        /// Training rows used by the regression.
        in_the_money_paths: u32,
        /// Numerical regression diagnostics.
        diagnostics: RegressionDiagnostics,
    },
}

/// Node identity paired with its immutable fit diagnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExerciseNodeDiagnostics {
    node: NodeId,
    fit: ExerciseFitDiagnostics,
}

impl ExerciseNodeDiagnostics {
    /// Returns the exact exercise node.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.node
    }

    /// Returns the node's fit classification and numerical evidence.
    #[must_use]
    pub const fn fit(self) -> ExerciseFitDiagnostics {
        self.fit
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ExerciseRule {
    Origin { exercise: bool },
    NoTrainingSupport,
    Regression(PolynomialModel),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TrainingPopulationIdentity {
    Seeded(RandomStream),
    ExternalNormals([u8; 32]),
}

/// Call-option LSM estimator with a complete explicit regression and tie identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsmCallEstimator {
    strike: f64,
    regression: PolynomialRegressionSpec,
    tie_rule: ExerciseTieRule,
}

impl LsmCallEstimator {
    /// Validates an independently evaluated call-option LSM specification.
    ///
    /// # Errors
    ///
    /// Returns [`LsmError::InvalidStrike`] unless strike is finite and strictly positive. The
    /// regression specification must already be validated and is never altered.
    pub fn try_new(
        strike: f64,
        regression: PolynomialRegressionSpec,
        tie_rule: ExerciseTieRule,
    ) -> Result<Self, LsmError> {
        if !strike.is_finite() || strike <= 0.0 {
            return Err(LsmError::InvalidStrike { observed: strike });
        }
        Ok(Self {
            strike,
            regression,
            tie_rule,
        })
    }

    /// Returns the fixed strike.
    #[must_use]
    pub const fn strike(self) -> f64 {
        self.strike
    }

    /// Returns the exact regression estimator identity.
    #[must_use]
    pub const fn regression(self) -> PolynomialRegressionSpec {
        self.regression
    }

    /// Returns the exact exercise tie convention.
    #[must_use]
    pub const fn tie_rule(self) -> ExerciseTieRule {
        self.tie_rule
    }

    /// Fits a stopping policy on one training population without valuing it in sample.
    ///
    /// A valuation-origin exercise node is a common decision: the intrinsic value is compared with
    /// the mean discounted training continuation. Every positive-time exercise node fits only on
    /// in-the-money paths. Zero such paths is recorded as absent support; a positive unsupported
    /// population is a typed regression failure.
    ///
    /// # Errors
    ///
    /// Returns a typed grid, state, discount, regression, or prediction failure. No model, degree,
    /// population, or exercise rule is substituted.
    pub fn fit(
        self,
        training: &SimulatedPaths,
        curve: &LogDiscountCurve,
        initial_spot: f64,
    ) -> Result<FittedLsmCallPolicy, LsmError> {
        self.fit_with_math_backend(training, curve, initial_spot, CpuMathBackend::RustStdF64)
    }

    pub(crate) fn fit_with_math_backend(
        self,
        training: &SimulatedPaths,
        curve: &LogDiscountCurve,
        initial_spot: f64,
        math_backend: CpuMathBackend,
    ) -> Result<FittedLsmCallPolicy, LsmError> {
        validate_initial_spot(initial_spot)?;
        validate_curve(training, curve)?;
        let training_count = path_count("training", training)?;
        validate_initial_state(training, initial_spot)?;

        let nodes = training.observation_nodes();
        let exercise_rows = nodes.len() - 1;
        let terminal = row(training, exercise_rows);
        let mut values = terminal
            .iter()
            .map(|&spot| call_intrinsic(spot, self.strike))
            .collect::<Vec<_>>();
        let mut regression_workspace = RegressionWorkspace::default();
        let mut candidate_states = Vec::new();
        let mut candidate_values = Vec::new();
        let mut reversed_rules = Vec::with_capacity(exercise_rows);
        let mut reversed_diagnostics = Vec::with_capacity(exercise_rows);
        let mut later_node = nodes[exercise_rows];

        for exercise_row in (0..exercise_rows).rev() {
            let node = nodes[exercise_row];
            if node.get() == 0 {
                continue;
            }
            discount_values(&mut values, curve, node, later_node)?;
            let spots = row(training, exercise_row);
            let (rule, diagnostic) = self.fit_positive_time_rule(
                node,
                spots,
                &mut values,
                training_count,
                &mut candidate_states,
                &mut candidate_values,
                &mut regression_workspace,
                math_backend,
            )?;
            reversed_rules.push(rule);
            reversed_diagnostics.push(ExerciseNodeDiagnostics {
                node,
                fit: diagnostic,
            });
            later_node = node;
        }

        discount_values_to_origin(&mut values, curve, later_node)?;
        if nodes.first().is_some_and(|node| node.get() == 0) {
            let estimated_continuation = online_mean(&values, training_count)?;
            let intrinsic = call_intrinsic(initial_spot, self.strike);
            let exercise = self
                .tie_rule
                .should_exercise(intrinsic, estimated_continuation);
            reversed_rules.push(ExerciseRule::Origin { exercise });
            reversed_diagnostics.push(ExerciseNodeDiagnostics {
                node: nodes[0],
                fit: ExerciseFitDiagnostics::OriginDecision {
                    exercise,
                    estimated_continuation,
                },
            });
        }
        reversed_rules.reverse();
        reversed_diagnostics.reverse();

        let discount_to_origin = observation_discounts_to_origin(curve, nodes)?;
        Ok(FittedLsmCallPolicy {
            path_plan: training.path_plan().clone(),
            observation_nodes: nodes.into(),
            discount_to_origin,
            training_population: training_population_identity(training),
            initial_spot,
            estimator: self,
            rules: reversed_rules.into_boxed_slice(),
            diagnostics: reversed_diagnostics.into_boxed_slice(),
        })
    }

    /// Simulates disjoint training/evaluation populations, fits once, and values only out of sample.
    ///
    /// # Errors
    ///
    /// Returns before simulation if the random domains collide, then propagates role-specific
    /// simulation or fixed-policy failures. Training storage is released before evaluation paths
    /// are generated.
    #[allow(clippy::too_many_arguments)]
    pub fn price_seeded(
        self,
        executor: &CpuExecutor,
        plan: &GbmPathPlan,
        curve: &LogDiscountCurve,
        initial_spot: f64,
        training_paths: usize,
        evaluation_paths: usize,
        training_stream: RandomStream,
        evaluation_stream: RandomStream,
    ) -> Result<LsmRun, LsmError> {
        if training_stream.domain() == evaluation_stream.domain() {
            return Err(LsmError::RandomDomainCollision {
                domain: training_stream.domain(),
            });
        }
        if plan.curve() != curve {
            return Err(LsmError::SimulationCurveMismatch);
        }
        let training = plan
            .simulate_seeded_with(executor, initial_spot, training_paths, training_stream)
            .map_err(|source| LsmError::TrainingSimulation { source })?;
        let policy =
            self.fit_with_math_backend(&training, curve, initial_spot, executor.math_backend())?;
        drop(training);
        let evaluation_paths = plan
            .simulate_seeded_with(executor, initial_spot, evaluation_paths, evaluation_stream)
            .map_err(|source| LsmError::EvaluationSimulation { source })?;
        let evaluation = policy.evaluate(&evaluation_paths)?;
        Ok(LsmRun { policy, evaluation })
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn price_seeded_serial(
        self,
        plan: &GbmPathPlan,
        curve: &LogDiscountCurve,
        initial_spot: f64,
        training_paths: usize,
        evaluation_paths: usize,
        training_stream: RandomStream,
        evaluation_stream: RandomStream,
    ) -> Result<LsmRun, LsmError> {
        self.price_seeded_serial_with_math_backend(
            CpuMathBackend::RustStdF64,
            plan,
            curve,
            initial_spot,
            training_paths,
            evaluation_paths,
            training_stream,
            evaluation_stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn price_seeded_serial_with_math_backend(
        self,
        math_backend: CpuMathBackend,
        plan: &GbmPathPlan,
        curve: &LogDiscountCurve,
        initial_spot: f64,
        training_paths: usize,
        evaluation_paths: usize,
        training_stream: RandomStream,
        evaluation_stream: RandomStream,
    ) -> Result<LsmRun, LsmError> {
        if training_stream.domain() == evaluation_stream.domain() {
            return Err(LsmError::RandomDomainCollision {
                domain: training_stream.domain(),
            });
        }
        if plan.curve() != curve {
            return Err(LsmError::SimulationCurveMismatch);
        }
        let training = plan
            .simulate_seeded_with_math_backend(
                math_backend,
                initial_spot,
                training_paths,
                training_stream,
            )
            .map_err(|source| LsmError::TrainingSimulation { source })?;
        let policy = self.fit_with_math_backend(&training, curve, initial_spot, math_backend)?;
        drop(training);
        let evaluation_paths = plan
            .simulate_seeded_with_math_backend(
                math_backend,
                initial_spot,
                evaluation_paths,
                evaluation_stream,
            )
            .map_err(|source| LsmError::EvaluationSimulation { source })?;
        let evaluation = policy.evaluate(&evaluation_paths)?;
        Ok(LsmRun { policy, evaluation })
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn price_escrowed_seeded_serial(
        self,
        plan: &EscrowedGbmPathPlan,
        curve: &LogDiscountCurve,
        initial_spot: f64,
        training_paths: usize,
        evaluation_paths: usize,
        training_stream: RandomStream,
        evaluation_stream: RandomStream,
    ) -> Result<LsmRun, LsmError> {
        self.price_escrowed_seeded_serial_with_math_backend(
            CpuMathBackend::RustStdF64,
            plan,
            curve,
            initial_spot,
            training_paths,
            evaluation_paths,
            training_stream,
            evaluation_stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn price_escrowed_seeded_serial_with_math_backend(
        self,
        math_backend: CpuMathBackend,
        plan: &EscrowedGbmPathPlan,
        curve: &LogDiscountCurve,
        initial_spot: f64,
        training_paths: usize,
        evaluation_paths: usize,
        training_stream: RandomStream,
        evaluation_stream: RandomStream,
    ) -> Result<LsmRun, LsmError> {
        if training_stream.domain() == evaluation_stream.domain() {
            return Err(LsmError::RandomDomainCollision {
                domain: training_stream.domain(),
            });
        }
        if plan.curve() != curve {
            return Err(LsmError::SimulationCurveMismatch);
        }
        let training = plan
            .simulate_seeded_with_math_backend(
                math_backend,
                initial_spot,
                training_paths,
                training_stream,
            )
            .map_err(|source| LsmError::TrainingSimulation { source })?;
        let policy = self.fit_with_math_backend(&training, curve, initial_spot, math_backend)?;
        drop(training);
        let evaluation_paths = plan
            .simulate_seeded_with_math_backend(
                math_backend,
                initial_spot,
                evaluation_paths,
                evaluation_stream,
            )
            .map_err(|source| LsmError::EvaluationSimulation { source })?;
        let evaluation = policy.evaluate(&evaluation_paths)?;
        Ok(LsmRun { policy, evaluation })
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_positive_time_rule(
        self,
        node: NodeId,
        spots: &[f64],
        values: &mut [f64],
        training_count: u32,
        candidate_states: &mut Vec<f64>,
        candidate_values: &mut Vec<f64>,
        workspace: &mut RegressionWorkspace,
        math_backend: CpuMathBackend,
    ) -> Result<(ExerciseRule, ExerciseFitDiagnostics), LsmError> {
        candidate_states.clear();
        candidate_values.clear();
        for (&spot, &value) in spots.iter().zip(&*values) {
            if spot > self.strike {
                candidate_states.push(spot / self.strike);
                candidate_values.push(value);
            }
        }
        if candidate_states.is_empty() {
            return Ok((
                ExerciseRule::NoTrainingSupport,
                ExerciseFitDiagnostics::NoInTheMoneyPaths,
            ));
        }
        let model = workspace
            .fit_with_math_backend(
                &candidate_states[..],
                &candidate_values[..],
                self.regression,
                math_backend,
            )
            .map_err(|source| LsmError::RegressionAtNode {
                node: node.get(),
                source,
            })?;
        let diagnostics = model.diagnostics();
        self.apply_training_rule(node, spots, values, &model)?;
        let in_the_money_paths = u32::try_from(candidate_states.len()).map_err(|_| {
            LsmError::PathCountOutsideDomain {
                population: "training in-the-money",
                observed: candidate_states.len(),
                maximum: u32::MAX,
            }
        })?;
        debug_assert!(in_the_money_paths <= training_count);
        Ok((
            ExerciseRule::Regression(model),
            ExerciseFitDiagnostics::Regression {
                in_the_money_paths,
                diagnostics,
            },
        ))
    }

    fn apply_training_rule(
        self,
        node: NodeId,
        spots: &[f64],
        values: &mut [f64],
        model: &PolynomialModel,
    ) -> Result<(), LsmError> {
        match model.degree() {
            0 => self.apply_training_rule_for::<0>(node, spots, values, model),
            1 => self.apply_training_rule_for::<1>(node, spots, values, model),
            2 => self.apply_training_rule_for::<2>(node, spots, values, model),
            3 => self.apply_training_rule_for::<3>(node, spots, values, model),
            4 => self.apply_training_rule_for::<4>(node, spots, values, model),
            _ => unreachable!("validated polynomial model has degree zero through four"),
        }
    }

    fn apply_training_rule_for<const DEGREE: usize>(
        self,
        node: NodeId,
        spots: &[f64],
        values: &mut [f64],
        model: &PolynomialModel,
    ) -> Result<(), LsmError> {
        for (path, (&spot, value)) in spots.iter().zip(values).enumerate() {
            if spot <= self.strike {
                continue;
            }
            let intrinsic = spot - self.strike;
            let continuation =
                model
                    .predict_for::<DEGREE>(spot / self.strike)
                    .map_err(|source| LsmError::PredictionAtPath {
                        node: node.get(),
                        path,
                        source,
                    })?;
            if self.tie_rule.should_exercise(intrinsic, continuation) {
                *value = intrinsic;
            }
        }
        Ok(())
    }
}

/// Immutable stopping policy fitted on one training population.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedLsmCallPolicy {
    path_plan: GbmPathPlan,
    observation_nodes: Box<[NodeId]>,
    discount_to_origin: Box<[f64]>,
    training_population: TrainingPopulationIdentity,
    initial_spot: f64,
    estimator: LsmCallEstimator,
    rules: Box<[ExerciseRule]>,
    diagnostics: Box<[ExerciseNodeDiagnostics]>,
}

impl FittedLsmCallPolicy {
    /// Returns the estimator identity used to fit this policy.
    #[must_use]
    pub const fn estimator(&self) -> LsmCallEstimator {
        self.estimator
    }

    /// Returns exact exercise nodes followed by terminal expiry.
    #[must_use]
    pub const fn observation_nodes(&self) -> &[NodeId] {
        &self.observation_nodes
    }

    /// Returns one immutable diagnostic per exercise node.
    #[must_use]
    pub const fn diagnostics(&self) -> &[ExerciseNodeDiagnostics] {
        &self.diagnostics
    }

    /// Evaluates this fixed policy on a distinct path population without refitting.
    ///
    /// # Errors
    ///
    /// Returns a typed identity, support, prediction, or arithmetic failure. A missing training
    /// model is accepted only while no active evaluation path enters that in-the-money region.
    pub fn evaluate(&self, evaluation: &SimulatedPaths) -> Result<LsmEvaluation, LsmError> {
        let evaluation_count = path_count("evaluation", evaluation)?;
        if evaluation.curve() != self.path_plan.curve() {
            return Err(LsmError::SimulationCurveMismatch);
        }
        if evaluation.path_plan() != &self.path_plan {
            return Err(LsmError::EvaluationPathPlanMismatch);
        }
        if evaluation.observation_nodes() != &*self.observation_nodes {
            return Err(LsmError::ObservationNodeMismatch);
        }
        self.validate_evaluation_population(evaluation)?;
        validate_initial_state(evaluation, self.initial_spot)?;

        let mut exercise_counts = vec![0_u32; self.observation_nodes.len()];
        if self
            .rules
            .first()
            .is_some_and(|rule| matches!(rule, ExerciseRule::Origin { exercise: true }))
        {
            exercise_counts[0] = evaluation_count;
            let intrinsic = call_intrinsic(self.initial_spot, self.estimator.strike);
            return Ok(LsmEvaluation::deterministic(
                evaluation_count,
                intrinsic,
                exercise_counts,
            ));
        }

        let paths = evaluation.path_count();
        let mut active = vec![1_u8; paths];
        let mut discounted_payoffs = vec![0.0; paths];
        for (exercise_row, rule) in self.rules.iter().enumerate() {
            match rule {
                ExerciseRule::Origin { exercise: false } => {}
                ExerciseRule::Origin { exercise: true } => {
                    return Err(LsmError::InvalidPolicyOriginRule {
                        row: exercise_row,
                        node: self.observation_nodes[exercise_row].get(),
                    });
                }
                ExerciseRule::NoTrainingSupport => {
                    let active_in_the_money_paths = row(evaluation, exercise_row)
                        .iter()
                        .zip(&active)
                        .filter(|(spot, is_active)| {
                            **is_active != 0 && **spot > self.estimator.strike
                        })
                        .count();
                    if active_in_the_money_paths > 0 {
                        return Err(LsmError::EvaluationOutsideTrainingSupport {
                            node: self.observation_nodes[exercise_row].get(),
                            active_in_the_money_paths: u32::try_from(active_in_the_money_paths)
                                .map_err(|_| LsmError::PathCountOutsideDomain {
                                    population: "active evaluation in-the-money",
                                    observed: active_in_the_money_paths,
                                    maximum: u32::MAX,
                                })?,
                        });
                    }
                }
                ExerciseRule::Regression(model) => self.apply_evaluation_rule(
                    evaluation,
                    exercise_row,
                    model,
                    &mut active,
                    &mut discounted_payoffs,
                    &mut exercise_counts,
                )?,
            }
        }

        let terminal_row = self.observation_nodes.len() - 1;
        let terminal_discount = self.discount_to_origin[terminal_row];
        let mut terminal_count = 0_u32;
        for (path, (&spot, is_active)) in row(evaluation, terminal_row)
            .iter()
            .zip(&mut active)
            .enumerate()
        {
            if *is_active == 0 {
                continue;
            }
            let payoff = call_intrinsic(spot, self.estimator.strike) * terminal_discount;
            if !payoff.is_finite() {
                return Err(LsmError::NonFiniteDiscountedPayoff { path });
            }
            discounted_payoffs[path] = payoff;
            *is_active = 0;
            terminal_count += 1;
        }
        exercise_counts[terminal_row] = terminal_count;
        LsmEvaluation::from_path_payoffs(evaluation_count, &discounted_payoffs, exercise_counts)
    }

    fn apply_evaluation_rule(
        &self,
        evaluation: &SimulatedPaths,
        exercise_row: usize,
        model: &PolynomialModel,
        active: &mut [u8],
        discounted_payoffs: &mut [f64],
        exercise_counts: &mut [u32],
    ) -> Result<(), LsmError> {
        match model.degree() {
            0 => self.apply_evaluation_rule_for::<0>(
                evaluation,
                exercise_row,
                model,
                active,
                discounted_payoffs,
                exercise_counts,
            ),
            1 => self.apply_evaluation_rule_for::<1>(
                evaluation,
                exercise_row,
                model,
                active,
                discounted_payoffs,
                exercise_counts,
            ),
            2 => self.apply_evaluation_rule_for::<2>(
                evaluation,
                exercise_row,
                model,
                active,
                discounted_payoffs,
                exercise_counts,
            ),
            3 => self.apply_evaluation_rule_for::<3>(
                evaluation,
                exercise_row,
                model,
                active,
                discounted_payoffs,
                exercise_counts,
            ),
            4 => self.apply_evaluation_rule_for::<4>(
                evaluation,
                exercise_row,
                model,
                active,
                discounted_payoffs,
                exercise_counts,
            ),
            _ => unreachable!("validated polynomial model has degree zero through four"),
        }
    }

    fn apply_evaluation_rule_for<const DEGREE: usize>(
        &self,
        evaluation: &SimulatedPaths,
        exercise_row: usize,
        model: &PolynomialModel,
        active: &mut [u8],
        discounted_payoffs: &mut [f64],
        exercise_counts: &mut [u32],
    ) -> Result<(), LsmError> {
        let node = self.observation_nodes[exercise_row];
        let discount = self.discount_to_origin[exercise_row];
        let mut count = 0_u32;
        for (path, (&spot, is_active)) in
            row(evaluation, exercise_row).iter().zip(active).enumerate()
        {
            if *is_active == 0 || spot <= self.estimator.strike {
                continue;
            }
            let intrinsic = spot - self.estimator.strike;
            let continuation = model
                .predict_for::<DEGREE>(spot / self.estimator.strike)
                .map_err(|source| LsmError::PredictionAtPath {
                    node: node.get(),
                    path,
                    source,
                })?;
            if self
                .estimator
                .tie_rule
                .should_exercise(intrinsic, continuation)
            {
                let payoff = intrinsic * discount;
                if !payoff.is_finite() {
                    return Err(LsmError::NonFiniteDiscountedPayoff { path });
                }
                discounted_payoffs[path] = payoff;
                *is_active = 0;
                count += 1;
            }
        }
        exercise_counts[exercise_row] = count;
        Ok(())
    }

    fn validate_evaluation_population(&self, evaluation: &SimulatedPaths) -> Result<(), LsmError> {
        match (&self.training_population, evaluation.random_stream()) {
            (TrainingPopulationIdentity::Seeded(training), Some(evaluation)) => {
                if *training == evaluation {
                    return Err(LsmError::TrainingEvaluationPopulationCollision);
                }
                if training.domain() == evaluation.domain() {
                    return Err(LsmError::RandomDomainCollision {
                        domain: training.domain(),
                    });
                }
            }
            (TrainingPopulationIdentity::ExternalNormals(training), None) => {
                if *training == external_population_digest(evaluation) {
                    return Err(LsmError::TrainingEvaluationPopulationCollision);
                }
            }
            (TrainingPopulationIdentity::Seeded(_), None) => {
                return Err(LsmError::IncomparablePopulationIdentity {
                    training: "seeded",
                    evaluation: "external-normal",
                });
            }
            (TrainingPopulationIdentity::ExternalNormals(_), Some(_)) => {
                return Err(LsmError::IncomparablePopulationIdentity {
                    training: "external-normal",
                    evaluation: "seeded",
                });
            }
        }
        Ok(())
    }
}

/// Independent fixed-policy valuation and sampling diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct LsmEvaluation {
    paths: u32,
    price: f64,
    sample_standard_deviation: Option<f64>,
    standard_error: Option<f64>,
    exercise_counts: Box<[u32]>,
}

impl LsmEvaluation {
    fn deterministic(paths: u32, price: f64, exercise_counts: Vec<u32>) -> Self {
        Self {
            paths,
            price,
            sample_standard_deviation: Some(0.0),
            standard_error: Some(0.0),
            exercise_counts: exercise_counts.into_boxed_slice(),
        }
    }

    fn from_path_payoffs(
        paths: u32,
        payoffs: &[f64],
        exercise_counts: Vec<u32>,
    ) -> Result<Self, LsmError> {
        let mut count = 0_u32;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for (path, &payoff) in payoffs.iter().enumerate() {
            if !payoff.is_finite() {
                return Err(LsmError::NonFiniteDiscountedPayoff { path });
            }
            count += 1;
            let delta = payoff - mean;
            mean += delta / f64::from(count);
            m2 = delta.mul_add(payoff - mean, m2);
        }
        debug_assert_eq!(count, paths);
        let (sample_standard_deviation, standard_error) = if paths > 1 {
            let standard_deviation = (m2 / f64::from(paths - 1)).sqrt();
            let standard_error = standard_deviation / f64::from(paths).sqrt();
            if !standard_deviation.is_finite() || !standard_error.is_finite() {
                return Err(LsmError::NonFiniteDiscountedPayoff { path: 0 });
            }
            (Some(standard_deviation), Some(standard_error))
        } else {
            (None, None)
        };
        Ok(Self {
            paths,
            price: mean,
            sample_standard_deviation,
            standard_error,
            exercise_counts: exercise_counts.into_boxed_slice(),
        })
    }

    /// Returns the independent evaluation path count.
    #[must_use]
    pub const fn paths(&self) -> u32 {
        self.paths
    }

    /// Returns the mean discounted payoff under the fixed policy.
    #[must_use]
    pub const fn price(&self) -> f64 {
        self.price
    }

    /// Returns the sample standard deviation, or `None` for one evaluation path.
    #[must_use]
    pub const fn sample_standard_deviation(&self) -> Option<f64> {
        self.sample_standard_deviation
    }

    /// Returns the standard error of the independent mean, or `None` for one path.
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }

    /// Returns stopping counts aligned with exercise nodes followed by terminal expiry.
    #[must_use]
    pub const fn exercise_counts(&self) -> &[u32] {
        &self.exercise_counts
    }
}

/// A fitted policy retained alongside its independent evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct LsmRun {
    policy: FittedLsmCallPolicy,
    evaluation: LsmEvaluation,
}

impl LsmRun {
    /// Borrows the fitted stopping policy and all regression diagnostics.
    #[must_use]
    pub const fn policy(&self) -> &FittedLsmCallPolicy {
        &self.policy
    }

    /// Borrows the independent valuation.
    #[must_use]
    pub const fn evaluation(&self) -> &LsmEvaluation {
        &self.evaluation
    }

    /// Consumes the run into its policy and independent evaluation.
    #[must_use]
    pub fn into_parts(self) -> (FittedLsmCallPolicy, LsmEvaluation) {
        (self.policy, self.evaluation)
    }
}

fn validate_initial_spot(initial_spot: f64) -> Result<(), LsmError> {
    if !initial_spot.is_finite()
        || initial_spot < 0.0
        || (initial_spot == 0.0 && initial_spot.is_sign_negative())
    {
        return Err(LsmError::InvalidInitialSpot {
            observed: initial_spot,
        });
    }
    Ok(())
}

fn validate_curve(paths: &SimulatedPaths, curve: &LogDiscountCurve) -> Result<(), LsmError> {
    if paths.curve() != curve {
        return Err(LsmError::SimulationCurveMismatch);
    }
    Ok(())
}

fn path_count(population: &'static str, paths: &SimulatedPaths) -> Result<u32, LsmError> {
    u32::try_from(paths.path_count()).map_err(|_| LsmError::PathCountOutsideDomain {
        population,
        observed: paths.path_count(),
        maximum: u32::MAX,
    })
}

fn validate_initial_state(paths: &SimulatedPaths, initial_spot: f64) -> Result<(), LsmError> {
    if paths.initial_spot().to_bits() != initial_spot.to_bits() {
        return Err(LsmError::InitialStateMismatch {
            path: 0,
            expected: initial_spot,
            observed: paths.initial_spot(),
        });
    }
    if paths
        .observation_nodes()
        .first()
        .is_none_or(|node| node.get() != 0)
    {
        return Ok(());
    }
    for (path, &observed) in row(paths, 0).iter().enumerate() {
        if observed.to_bits() != initial_spot.to_bits() {
            return Err(LsmError::InitialStateMismatch {
                path,
                expected: initial_spot,
                observed,
            });
        }
    }
    Ok(())
}

fn row(paths: &SimulatedPaths, row: usize) -> &[f64] {
    let start = row * paths.path_count();
    &paths.as_slice()[start..start + paths.path_count()]
}

fn call_intrinsic(spot: f64, strike: f64) -> f64 {
    (spot - strike).max(0.0)
}

fn discount_values(
    values: &mut [f64],
    curve: &LogDiscountCurve,
    from: NodeId,
    to: NodeId,
) -> Result<(), LsmError> {
    let discount = discount_factor(curve, from, to)?;
    for (path, value) in values.iter_mut().enumerate() {
        *value *= discount;
        if !value.is_finite() {
            return Err(LsmError::NonFiniteDiscountedPayoff { path });
        }
    }
    Ok(())
}

fn discount_values_to_origin(
    values: &mut [f64],
    curve: &LogDiscountCurve,
    to: NodeId,
) -> Result<(), LsmError> {
    let log_discount = curve.as_slice()[to.get() as usize];
    let discount = checked_discount(0, to.get(), log_discount)?;
    for (path, value) in values.iter_mut().enumerate() {
        *value *= discount;
        if !value.is_finite() {
            return Err(LsmError::NonFiniteDiscountedPayoff { path });
        }
    }
    Ok(())
}

fn discount_factor(curve: &LogDiscountCurve, from: NodeId, to: NodeId) -> Result<f64, LsmError> {
    let values = curve.as_slice();
    let log_discount = values[to.get() as usize] - values[from.get() as usize];
    checked_discount(from.get(), to.get(), log_discount)
}

fn checked_discount(from: u32, to: u32, log_discount: f64) -> Result<f64, LsmError> {
    let discount = log_discount.exp();
    if !discount.is_finite() || discount <= 0.0 {
        return Err(LsmError::UnrepresentableObservationDiscount {
            from,
            to,
            log_discount,
        });
    }
    Ok(discount)
}

fn observation_discounts_to_origin(
    curve: &LogDiscountCurve,
    nodes: &[NodeId],
) -> Result<Box<[f64]>, LsmError> {
    nodes
        .iter()
        .map(|node| checked_discount(0, node.get(), curve.as_slice()[node.get() as usize]))
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn online_mean(values: &[f64], expected_count: u32) -> Result<f64, LsmError> {
    let mut count = 0_u32;
    let mut mean = 0.0;
    for (path, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(LsmError::NonFiniteDiscountedPayoff { path });
        }
        count += 1;
        mean += (value - mean) / f64::from(count);
    }
    debug_assert_eq!(count, expected_count);
    Ok(mean)
}

fn training_population_identity(paths: &SimulatedPaths) -> TrainingPopulationIdentity {
    paths.random_stream().map_or_else(
        || TrainingPopulationIdentity::ExternalNormals(external_population_digest(paths)),
        TrainingPopulationIdentity::Seeded,
    )
}

fn external_population_digest(paths: &SimulatedPaths) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"finoptions.external-path-population.v1\0");
    hasher.update(&paths.path_count().to_le_bytes());
    hasher.update(&paths.observation_nodes().len().to_le_bytes());
    hasher.update(&paths.initial_spot().to_bits().to_le_bytes());
    for node in paths.observation_nodes() {
        hasher.update(&node.get().to_le_bytes());
    }
    for value in paths.as_slice() {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use finoptions_core::{
        CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy,
        EventSchedule, TimeGrid,
    };

    use super::*;
    use crate::{PolynomialDegree, RegressionError};

    fn estimator() -> LsmCallEstimator {
        let regression = PolynomialRegressionSpec::try_new(
            PolynomialDegree::try_new(4).unwrap(),
            1.0e-12,
            1.0e8,
        )
        .unwrap();
        LsmCallEstimator::try_new(100.0, regression, ExerciseTieRule::StrictlyGreater).unwrap()
    }

    fn plan(
        times: &[f64],
        log_discounts: &[f64],
        dividend: Option<(usize, f64)>,
        exercise_nodes: &[usize],
    ) -> (GbmPathPlan, LogDiscountCurve) {
        let grid = TimeGrid::try_new(times).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
        let dividend_yield_curve =
            CumulativeDividendYieldCurve::try_new(&grid, vec![0.0; grid.len()]).unwrap();
        let dividends = dividend
            .map(|(node, amount)| {
                vec![CashDividend::try_new(grid.node_id(node).unwrap(), amount).unwrap()]
            })
            .unwrap_or_default();
        let exercise_nodes = exercise_nodes
            .iter()
            .map(|&node| grid.node_id(node).unwrap())
            .collect::<Vec<_>>();
        let expiry = grid.node_id(grid.len() - 1).unwrap();
        let schedule = EventSchedule::try_new(grid, dividends, exercise_nodes, expiry).unwrap();
        let plan = GbmPathPlan::try_new(
            &schedule,
            &curve,
            &dividend_yield_curve,
            ConstantVolatility::try_new(0.30).unwrap(),
            DividendPolicy::Liquidator,
        )
        .unwrap();
        (plan, curve)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }

    #[test]
    fn validates_strike_and_preserves_the_explicit_tie_rule() {
        let regression = PolynomialRegressionSpec::try_new(
            PolynomialDegree::try_new(4).unwrap(),
            1.0e-12,
            1.0e8,
        )
        .unwrap();
        assert!(matches!(
            LsmCallEstimator::try_new(0.0, regression, ExerciseTieRule::StrictlyGreater),
            Err(LsmError::InvalidStrike { .. })
        ));
        assert!(!ExerciseTieRule::StrictlyGreater.should_exercise(1.0, 1.0));
        assert!(ExerciseTieRule::GreaterOrEqual.should_exercise(1.0, 1.0));
    }

    #[test]
    fn european_slice_discounts_the_independent_terminal_population() {
        let (plan, curve) = plan(&[0.0, 1.0], &[0.0, -0.05], None, &[]);
        let training = plan
            .simulate(100.0, 6, &[-1.5, -0.8, -0.2, 0.3, 0.9, 1.4])
            .unwrap();
        let evaluation = plan
            .simulate(100.0, 6, &[-1.4, -0.7, -0.1, 0.4, 1.0, 1.5])
            .unwrap();
        let policy = estimator().fit(&training, &curve, 100.0).unwrap();
        let result = policy.evaluate(&evaluation).unwrap();
        let expected = evaluation
            .as_slice()
            .iter()
            .map(|&spot| call_intrinsic(spot, 100.0) * (-0.05_f64).exp())
            .sum::<f64>()
            / 6.0;
        assert_close(
            result.price(),
            expected,
            32.0 * f64::EPSILON * expected.abs(),
        );
        assert_eq!(result.exercise_counts(), &[6]);
        assert_eq!(result.paths(), 6);
    }

    #[test]
    fn rejects_a_discount_curve_different_from_the_simulated_risk_neutral_dynamics() {
        let (simulation_plan, simulation_curve) = plan(&[0.0, 1.0], &[0.0, -0.05], None, &[]);
        let (other_plan, other_curve) = plan(&[0.0, 1.0], &[0.0, -0.04], None, &[]);
        let training = simulation_plan
            .simulate(100.0, 6, &[-1.5, -0.8, -0.2, 0.3, 0.9, 1.4])
            .unwrap();
        let evaluation = other_plan
            .simulate(100.0, 6, &[-1.4, -0.7, -0.1, 0.4, 1.0, 1.5])
            .unwrap();

        assert_eq!(
            estimator().fit(&training, &other_curve, 100.0).unwrap_err(),
            LsmError::SimulationCurveMismatch
        );
        let policy = estimator()
            .fit(&training, &simulation_curve, 100.0)
            .unwrap();
        assert_eq!(
            policy.evaluate(&evaluation).unwrap_err(),
            LsmError::SimulationCurveMismatch
        );

        let grid = simulation_curve.grid().clone();
        let expiry = grid.node_id(1).unwrap();
        let schedule = EventSchedule::try_new(grid, [], [], expiry).unwrap();
        let dividend_yield_curve =
            CumulativeDividendYieldCurve::try_new(simulation_curve.grid(), [0.0, 0.0]).unwrap();
        let other_volatility_plan = GbmPathPlan::try_new(
            &schedule,
            &simulation_curve,
            &dividend_yield_curve,
            ConstantVolatility::try_new(0.20).unwrap(),
            DividendPolicy::Liquidator,
        )
        .unwrap();
        let other_dynamics = other_volatility_plan
            .simulate(100.0, 6, &[-1.3, -0.6, 0.0, 0.5, 1.1, 1.6])
            .unwrap();
        assert_eq!(
            policy.evaluate(&other_dynamics).unwrap_err(),
            LsmError::EvaluationPathPlanMismatch
        );

        let executor = CpuExecutor::try_new(2).unwrap();
        assert_eq!(
            estimator()
                .price_seeded(
                    &executor,
                    &simulation_plan,
                    &other_curve,
                    100.0,
                    6,
                    6,
                    RandomStream::new(7, 1),
                    RandomStream::new(7, 2),
                )
                .unwrap_err(),
            LsmError::SimulationCurveMismatch
        );
    }

    #[test]
    fn origin_exercise_is_one_common_decision_not_a_degenerate_regression() {
        let (plan, curve) = plan(&[0.0, 0.5, 1.0], &[0.0, 0.0, 0.0], Some((1, 200.0)), &[0]);
        let executor = CpuExecutor::try_new(2).unwrap();
        let run = estimator()
            .price_seeded(
                &executor,
                &plan,
                &curve,
                200.0,
                1_000,
                1_000,
                RandomStream::new(7, 1),
                RandomStream::new(7, 2),
            )
            .unwrap();
        assert_eq!(run.evaluation().price().to_bits(), 100.0_f64.to_bits());
        assert_eq!(run.evaluation().exercise_counts(), &[1_000, 0]);
        assert!(matches!(
            run.policy().diagnostics()[0].fit(),
            ExerciseFitDiagnostics::OriginDecision { exercise: true, .. }
        ));
    }

    #[test]
    fn exact_same_training_population_is_rejected_for_both_input_routes() {
        let (plan, curve) = plan(&[0.0, 1.0], &[0.0, -0.05], None, &[]);
        let external = plan
            .simulate(100.0, 5, &[-1.0, -0.5, 0.0, 0.5, 1.0])
            .unwrap();
        let external_policy = estimator().fit(&external, &curve, 100.0).unwrap();
        assert_eq!(
            external_policy.evaluate(&external).unwrap_err(),
            LsmError::TrainingEvaluationPopulationCollision
        );

        let seeded = plan
            .simulate_seeded(100.0, 5, RandomStream::new(9, 4))
            .unwrap();
        let seeded_policy = estimator().fit(&seeded, &curve, 100.0).unwrap();
        assert_eq!(
            seeded_policy.evaluate(&seeded).unwrap_err(),
            LsmError::TrainingEvaluationPopulationCollision
        );
    }

    #[test]
    fn missing_training_support_and_degenerate_positive_support_fail_at_the_node() {
        let (plan, curve) = plan(&[0.0, 0.5, 1.0], &[0.0, 0.0, 0.0], None, &[1]);
        let training_normals = [-5.0, -5.0, -5.0, -5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let evaluation_normals = [5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let training = plan.simulate(100.0, 5, &training_normals).unwrap();
        let evaluation = plan.simulate(100.0, 5, &evaluation_normals).unwrap();
        let policy = estimator().fit(&training, &curve, 100.0).unwrap();
        assert!(matches!(
            policy.evaluate(&evaluation),
            Err(LsmError::EvaluationOutsideTrainingSupport {
                node: 1,
                active_in_the_money_paths: 5
            })
        ));

        let identical_itm_normals = [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let degenerate = plan.simulate(100.0, 5, &identical_itm_normals).unwrap();
        assert!(matches!(
            estimator().fit(&degenerate, &curve, 100.0),
            Err(LsmError::RegressionAtNode {
                node: 1,
                source: RegressionError::DegeneratePredictor { .. }
            })
        ));
    }

    #[test]
    fn seeded_run_is_thread_invariant_and_requires_distinct_domains() {
        let (plan, curve) = plan(&[0.0, 0.5, 1.0], &[0.0, -0.03, -0.06], Some((1, 7.0)), &[1]);
        let training_stream = RandomStream::new(11, 1);
        let evaluation_stream = RandomStream::new(11, 2);
        let serial = estimator()
            .price_seeded(
                &CpuExecutor::try_new(1).unwrap(),
                &plan,
                &curve,
                100.0,
                10_000,
                10_000,
                training_stream,
                evaluation_stream,
            )
            .unwrap();
        let parallel = estimator()
            .price_seeded(
                &CpuExecutor::try_new(4).unwrap(),
                &plan,
                &curve,
                100.0,
                10_000,
                10_000,
                training_stream,
                evaluation_stream,
            )
            .unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(
            serial.evaluation().exercise_counts().iter().sum::<u32>(),
            10_000
        );
        assert!(matches!(
            estimator().price_seeded(
                &CpuExecutor::try_new(1).unwrap(),
                &plan,
                &curve,
                100.0,
                10,
                10,
                RandomStream::new(1, 7),
                RandomStream::new(2, 7),
            ),
            Err(LsmError::RandomDomainCollision { domain: 7 })
        ));
    }

    #[test]
    fn one_dividend_replications_cover_the_independent_reference() {
        const STEPS: u32 = 32;
        const PATHS: usize = 65_536;
        const REPLICATIONS: u32 = 12;
        const REFERENCE: f64 = 11.656_448_947_710_173;
        const STUDENT_T_99_DF_11: f64 = 3.105_806_515_539_280_4;

        let times = (0..=STEPS)
            .map(|node| f64::from(node) / f64::from(STEPS))
            .collect::<Vec<_>>();
        let log_discounts = times
            .iter()
            .map(|time| if *time == 0.0 { 0.0 } else { -0.06 * time })
            .collect::<Vec<_>>();
        let (plan, curve) = plan(
            &times,
            &log_discounts,
            Some((usize::try_from(STEPS / 2).unwrap(), 7.0)),
            &[usize::try_from(STEPS / 2).unwrap()],
        );
        let executor = CpuExecutor::try_new(4).unwrap();
        let mut count = 0_u32;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for replication in 0..REPLICATIONS {
            let value = estimator()
                .price_seeded(
                    &executor,
                    &plan,
                    &curve,
                    100.0,
                    PATHS,
                    PATHS,
                    RandomStream::new(10_000 + u64::from(replication), 1),
                    RandomStream::new(20_000 + u64::from(replication), 2),
                )
                .unwrap()
                .evaluation()
                .price();
            count += 1;
            let delta = value - mean;
            mean += delta / f64::from(count);
            m2 = delta.mul_add(value - mean, m2);
        }
        let sample_standard_deviation = (m2 / f64::from(REPLICATIONS - 1)).sqrt();
        let standard_error = sample_standard_deviation / f64::from(REPLICATIONS).sqrt();
        let half_width = STUDENT_T_99_DF_11 * standard_error;
        assert!(
            (mean - half_width..=mean + half_width).contains(&REFERENCE),
            "mean={mean:?}, standard_error={standard_error:?}, interval={:?}",
            [mean - half_width, mean + half_width]
        );
    }
}
