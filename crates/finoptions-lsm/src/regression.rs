//! Allocation-reusing low-degree polynomial least squares.

use finoptions_sim::CpuMathBackend;

use crate::RegressionError;

/// Largest polynomial degree admitted by the current production evidence gate.
pub const MAX_POLYNOMIAL_DEGREE: usize = 4;
const MAX_BASIS_COLUMNS: usize = MAX_POLYNOMIAL_DEGREE + 1;
const JACOBI_MAX_SWEEPS: usize = 32;
const JACOBI_RELATIVE_TOLERANCE: f64 = 16.0 * f64::EPSILON;

/// A polynomial degree validated before allocation or fitting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolynomialDegree(u8);

impl PolynomialDegree {
    /// Validates a polynomial degree against the current evidence gate.
    ///
    /// # Errors
    ///
    /// Returns [`RegressionError::UnsupportedDegree`] above degree four. Higher degrees are not
    /// silently truncated or regularized.
    pub fn try_new(degree: usize) -> Result<Self, RegressionError> {
        let degree = u8::try_from(degree).map_err(|_| RegressionError::UnsupportedDegree {
            observed: degree,
            maximum: MAX_POLYNOMIAL_DEGREE,
        })?;
        if usize::from(degree) > MAX_POLYNOMIAL_DEGREE {
            return Err(RegressionError::UnsupportedDegree {
                observed: usize::from(degree),
                maximum: MAX_POLYNOMIAL_DEGREE,
            });
        }
        Ok(Self(degree))
    }

    /// Returns the declared degree.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0 as usize
    }

    const fn columns(self) -> usize {
        self.get() + 1
    }
}

/// Complete numerical identity of the unregularized polynomial OLS solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolynomialRegressionSpec {
    degree: PolynomialDegree,
    rank_tolerance: f64,
    condition_limit: f64,
}

impl PolynomialRegressionSpec {
    /// Validates an explicit polynomial OLS specification.
    ///
    /// `rank_tolerance` is relative to the largest singular value of the standardized monomial
    /// design. `condition_limit` is a separate scientific/numerical domain restriction and may not
    /// exceed the reciprocal rank tolerance.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error. No tolerance or limit is substituted.
    pub fn try_new(
        degree: PolynomialDegree,
        rank_tolerance: f64,
        condition_limit: f64,
    ) -> Result<Self, RegressionError> {
        if !rank_tolerance.is_finite() || rank_tolerance <= 0.0 || rank_tolerance >= 1.0 {
            return Err(RegressionError::InvalidRankTolerance {
                observed: rank_tolerance,
            });
        }
        let maximum = rank_tolerance.recip();
        if !condition_limit.is_finite() || condition_limit < 1.0 || condition_limit > maximum {
            return Err(RegressionError::InvalidConditionLimit {
                observed: condition_limit,
                maximum,
            });
        }
        Ok(Self {
            degree,
            rank_tolerance,
            condition_limit,
        })
    }

    /// Returns the exact polynomial degree.
    #[must_use]
    pub const fn degree(self) -> PolynomialDegree {
        self.degree
    }

    /// Returns the relative numerical-rank threshold.
    #[must_use]
    pub const fn rank_tolerance(self) -> f64 {
        self.rank_tolerance
    }

    /// Returns the maximum admitted two-norm condition number.
    #[must_use]
    pub const fn condition_limit(self) -> f64 {
        self.condition_limit
    }
}

/// Diagnostics that define the accepted regression population and numerical domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionDiagnostics {
    observations: usize,
    columns: usize,
    rank: usize,
    condition_number: f64,
    predictor_mean: f64,
    predictor_scale: f64,
}

impl RegressionDiagnostics {
    /// Returns the fitted row count.
    #[must_use]
    pub const fn observations(self) -> usize {
        self.observations
    }

    /// Returns the number of polynomial basis columns.
    #[must_use]
    pub const fn columns(self) -> usize {
        self.columns
    }

    /// Returns the diagnosed numerical rank.
    #[must_use]
    pub const fn rank(self) -> usize {
        self.rank
    }

    /// Returns the two-norm condition number of the standardized monomial design.
    #[must_use]
    pub const fn condition_number(self) -> f64 {
        self.condition_number
    }

    /// Returns the fitted predictor location.
    #[must_use]
    pub const fn predictor_mean(self) -> f64 {
        self.predictor_mean
    }

    /// Returns the fitted population standard deviation.
    #[must_use]
    pub const fn predictor_scale(self) -> f64 {
        self.predictor_scale
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PolynomialModel {
    degree: PolynomialDegree,
    predictor_mean: f64,
    predictor_scale: f64,
    alphas: [f64; MAX_POLYNOMIAL_DEGREE],
    betas: [f64; MAX_POLYNOMIAL_DEGREE],
    coefficients: [f64; MAX_BASIS_COLUMNS],
    diagnostics: RegressionDiagnostics,
}

impl PolynomialModel {
    #[cfg(any(test, feature = "bench-internals"))]
    pub(crate) fn predict(&self, predictor: f64) -> Result<f64, RegressionError> {
        if !predictor.is_finite() {
            return Err(RegressionError::NonFinitePrediction { predictor });
        }
        let standardized = if self.degree.get() == 0 {
            0.0
        } else {
            (predictor - self.predictor_mean) / self.predictor_scale
        };
        let mut prediction = self.coefficients[0];
        let mut previous = 0.0;
        let mut current = 1.0;
        for basis_degree in 0..self.degree.get() {
            let next = (standardized - self.alphas[basis_degree])
                .mul_add(current, -self.betas[basis_degree] * previous);
            prediction = self.coefficients[basis_degree + 1].mul_add(next, prediction);
            previous = current;
            current = next;
        }
        if !prediction.is_finite() {
            return Err(RegressionError::NonFinitePrediction { predictor });
        }
        Ok(prediction)
    }

    pub(crate) fn predict_for<const DEGREE: usize>(
        &self,
        predictor: f64,
    ) -> Result<f64, RegressionError> {
        debug_assert_eq!(self.degree.get(), DEGREE);
        if !predictor.is_finite() {
            return Err(RegressionError::NonFinitePrediction { predictor });
        }
        let standardized = if DEGREE == 0 {
            0.0
        } else {
            (predictor - self.predictor_mean) / self.predictor_scale
        };
        let mut prediction = self.coefficients[0];
        let mut previous = 0.0;
        let mut current = 1.0;
        for basis_degree in 0..DEGREE {
            let next = (standardized - self.alphas[basis_degree])
                .mul_add(current, -self.betas[basis_degree] * previous);
            prediction = self.coefficients[basis_degree + 1].mul_add(next, prediction);
            previous = current;
            current = next;
        }
        if !prediction.is_finite() {
            return Err(RegressionError::NonFinitePrediction { predictor });
        }
        Ok(prediction)
    }

    pub(crate) const fn degree(&self) -> usize {
        self.degree.get()
    }

    pub(crate) const fn diagnostics(&self) -> RegressionDiagnostics {
        self.diagnostics
    }
}

#[derive(Debug, Default)]
pub(crate) struct RegressionWorkspace {
    standardized: Vec<f64>,
    previous: Vec<f64>,
    current: Vec<f64>,
    next: Vec<f64>,
}

impl RegressionWorkspace {
    #[cfg(test)]
    pub(crate) fn fit(
        &mut self,
        predictors: &[f64],
        responses: &[f64],
        spec: PolynomialRegressionSpec,
    ) -> Result<PolynomialModel, RegressionError> {
        self.fit_with_math_backend(predictors, responses, spec, CpuMathBackend::RustStdF64)
    }

    pub(crate) fn fit_with_math_backend(
        &mut self,
        predictors: &[f64],
        responses: &[f64],
        spec: PolynomialRegressionSpec,
        math_backend: CpuMathBackend,
    ) -> Result<PolynomialModel, RegressionError> {
        let observation_count = validate_columns(predictors, responses, spec)?;

        let observations = predictors.len();
        let degree = spec.degree.get();
        let (predictor_mean, predictor_scale) = predictor_location_scale(predictors, degree)?;
        self.prepare(observations);
        for (standardized, &predictor) in self.standardized.iter_mut().zip(predictors) {
            *standardized = if degree == 0 {
                0.0
            } else {
                (predictor - predictor_mean) / predictor_scale
            };
        }
        self.previous.fill(0.0);
        self.current.fill(1.0);

        let fit = self.fit_orthogonal(responses, observation_count, spec, math_backend)?;
        let (rank, condition_number) = diagnose_design(&fit, spec)?;

        Ok(PolynomialModel {
            degree: spec.degree,
            predictor_mean,
            predictor_scale,
            alphas: fit.alphas,
            betas: fit.betas,
            coefficients: fit.coefficients,
            diagnostics: RegressionDiagnostics {
                observations,
                columns: spec.degree.columns(),
                rank,
                condition_number,
                predictor_mean,
                predictor_scale,
            },
        })
    }

    fn fit_orthogonal(
        &mut self,
        responses: &[f64],
        observations: u32,
        spec: PolynomialRegressionSpec,
        math_backend: CpuMathBackend,
    ) -> Result<OrthogonalFit, RegressionError> {
        let degree = spec.degree.get();

        let mut alphas = [0.0; MAX_POLYNOMIAL_DEGREE];
        let mut betas = [0.0; MAX_POLYNOMIAL_DEGREE];
        let mut coefficients = [0.0; MAX_BASIS_COLUMNS];
        let mut norms = [0.0; MAX_BASIS_COLUMNS];
        let mut previous_norm = 1.0;

        for basis_degree in 0..=degree {
            let (norm, projection, weighted_predictor) = moments(
                math_backend,
                &self.standardized,
                responses,
                &self.current,
                &mut self.next,
            )?;
            if !norm.is_finite() || !projection.is_finite() || !weighted_predictor.is_finite() {
                return Err(RegressionError::NonFiniteFit { basis_degree });
            }
            if norm <= 0.0 {
                return Err(RegressionError::RankDeficient {
                    degree,
                    rank: basis_degree,
                    columns: spec.degree.columns(),
                    tolerance: spec.rank_tolerance,
                });
            }
            let relative_pivot = (norm / f64::from(observations)).sqrt();
            if relative_pivot <= spec.rank_tolerance {
                return Err(RegressionError::RankDeficient {
                    degree,
                    rank: basis_degree,
                    columns: spec.degree.columns(),
                    tolerance: spec.rank_tolerance,
                });
            }

            norms[basis_degree] = norm;
            coefficients[basis_degree] = projection / norm;
            if !coefficients[basis_degree].is_finite() {
                return Err(RegressionError::NonFiniteFit { basis_degree });
            }

            if basis_degree < degree {
                alphas[basis_degree] = weighted_predictor / norm;
                betas[basis_degree] = if basis_degree == 0 {
                    0.0
                } else {
                    norm / previous_norm
                };
                if !alphas[basis_degree].is_finite() || !betas[basis_degree].is_finite() {
                    return Err(RegressionError::NonFiniteFit { basis_degree });
                }
                for (((&standardized, &previous), &current), next) in self
                    .standardized
                    .iter()
                    .zip(&self.previous)
                    .zip(&self.current)
                    .zip(&mut self.next)
                {
                    *next = (standardized - alphas[basis_degree])
                        .mul_add(current, -betas[basis_degree] * previous);
                }
                if self.next.iter().any(|value| !value.is_finite()) {
                    return Err(RegressionError::NonFiniteFit {
                        basis_degree: basis_degree + 1,
                    });
                }
                previous_norm = norm;
                std::mem::swap(&mut self.previous, &mut self.current);
                std::mem::swap(&mut self.current, &mut self.next);
            }
        }

        Ok(OrthogonalFit {
            alphas,
            betas,
            coefficients,
            norms,
        })
    }

    fn prepare(&mut self, observations: usize) {
        self.standardized.resize(observations, 0.0);
        self.previous.resize(observations, 0.0);
        self.current.resize(observations, 0.0);
        self.next.resize(observations, 0.0);
    }
}

struct OrthogonalFit {
    alphas: [f64; MAX_POLYNOMIAL_DEGREE],
    betas: [f64; MAX_POLYNOMIAL_DEGREE],
    coefficients: [f64; MAX_BASIS_COLUMNS],
    norms: [f64; MAX_BASIS_COLUMNS],
}

fn diagnose_design(
    fit: &OrthogonalFit,
    spec: PolynomialRegressionSpec,
) -> Result<(usize, f64), RegressionError> {
    let degree = spec.degree.get();
    let columns = spec.degree.columns();
    let mut basis_transform =
        orthonormal_basis_transform(degree, &fit.alphas, &fit.betas, &fit.norms)?;
    let singular_values = one_sided_jacobi_singular_values(&mut basis_transform, columns)?;
    let largest = singular_values[0];
    let threshold = spec.rank_tolerance * largest;
    let rank = singular_values[..columns]
        .iter()
        .filter(|&&value| value > threshold)
        .count();
    if rank < columns {
        return Err(RegressionError::RankDeficient {
            degree,
            rank,
            columns,
            tolerance: spec.rank_tolerance,
        });
    }
    let condition_number = largest / singular_values[columns - 1];
    if !condition_number.is_finite() {
        return Err(RegressionError::NonFiniteFit {
            basis_degree: degree,
        });
    }
    if condition_number > spec.condition_limit {
        return Err(RegressionError::ConditionLimitExceeded {
            degree,
            observed: condition_number,
            limit: spec.condition_limit,
        });
    }
    Ok((rank, condition_number))
}

fn validate_columns(
    predictors: &[f64],
    responses: &[f64],
    spec: PolynomialRegressionSpec,
) -> Result<u32, RegressionError> {
    if predictors.len() != responses.len() {
        return Err(RegressionError::LengthMismatch {
            predictors: predictors.len(),
            responses: responses.len(),
        });
    }
    let required = spec.degree.columns();
    if predictors.len() < required {
        return Err(RegressionError::InsufficientObservations {
            observed: predictors.len(),
            required,
            degree: spec.degree.get(),
        });
    }
    let observations =
        u32::try_from(predictors.len()).map_err(|_| RegressionError::TooManyObservations {
            observed: predictors.len(),
            maximum: u32::MAX,
        })?;
    if let Some((index, &observed)) = predictors
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RegressionError::NonFinitePredictor { index, observed });
    }
    if let Some((index, &observed)) = responses
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RegressionError::NonFiniteResponse { index, observed });
    }
    Ok(observations)
}

fn predictor_location_scale(
    predictors: &[f64],
    degree: usize,
) -> Result<(f64, f64), RegressionError> {
    if degree == 0 {
        return Ok((0.0, 1.0));
    }
    let mut count = 0_u32;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for &predictor in predictors {
        count += 1;
        let delta = predictor - mean;
        mean += delta / f64::from(count);
        m2 = delta.mul_add(predictor - mean, m2);
    }
    let scale = (m2 / f64::from(count)).sqrt();
    if !mean.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err(RegressionError::DegeneratePredictor { degree, scale });
    }
    Ok((mean, scale))
}

fn moments(
    math_backend: CpuMathBackend,
    standardized: &[f64],
    responses: &[f64],
    basis: &[f64],
    squared_basis: &mut [f64],
) -> Result<(f64, f64, f64), RegressionError> {
    math_backend
        .polynomial_moments_f64(standardized, responses, basis, squared_basis)
        .map_err(|source| RegressionError::CpuMath {
            backend: math_backend.as_str(),
            source,
        })
}

fn orthonormal_basis_transform(
    degree: usize,
    alphas: &[f64; MAX_POLYNOMIAL_DEGREE],
    betas: &[f64; MAX_POLYNOMIAL_DEGREE],
    norms: &[f64; MAX_BASIS_COLUMNS],
) -> Result<[f64; MAX_BASIS_COLUMNS * MAX_BASIS_COLUMNS], RegressionError> {
    let mut transform = [0.0; MAX_BASIS_COLUMNS * MAX_BASIS_COLUMNS];
    let mut previous = [0.0; MAX_BASIS_COLUMNS];
    let mut current = [0.0; MAX_BASIS_COLUMNS];
    current[0] = 1.0;

    for basis_degree in 0..=degree {
        let inverse_norm = norms[basis_degree].sqrt().recip();
        if !inverse_norm.is_finite() {
            return Err(RegressionError::NonFiniteFit { basis_degree });
        }
        for monomial_degree in 0..=basis_degree {
            transform[basis_degree * MAX_BASIS_COLUMNS + monomial_degree] =
                current[monomial_degree] * inverse_norm;
        }
        if basis_degree == degree {
            break;
        }

        let mut next = [0.0; MAX_BASIS_COLUMNS];
        for monomial_degree in 0..=basis_degree {
            next[monomial_degree + 1] += current[monomial_degree];
            next[monomial_degree] -= alphas[basis_degree] * current[monomial_degree];
            next[monomial_degree] -= betas[basis_degree] * previous[monomial_degree];
        }
        if next.iter().any(|value| !value.is_finite()) {
            return Err(RegressionError::NonFiniteFit {
                basis_degree: basis_degree + 1,
            });
        }
        previous = current;
        current = next;
    }
    Ok(transform)
}

fn one_sided_jacobi_singular_values(
    matrix: &mut [f64; MAX_BASIS_COLUMNS * MAX_BASIS_COLUMNS],
    columns: usize,
) -> Result<[f64; MAX_BASIS_COLUMNS], RegressionError> {
    match columns {
        1 => one_sided_jacobi_singular_values_for::<1>(matrix),
        2 => one_sided_jacobi_singular_values_for::<2>(matrix),
        3 => one_sided_jacobi_singular_values_for::<3>(matrix),
        4 => one_sided_jacobi_singular_values_for::<4>(matrix),
        5 => one_sided_jacobi_singular_values_for::<5>(matrix),
        _ => unreachable!("validated polynomial degree has one to five columns"),
    }
}

fn one_sided_jacobi_singular_values_for<const COLUMNS: usize>(
    matrix: &mut [f64; MAX_BASIS_COLUMNS * MAX_BASIS_COLUMNS],
) -> Result<[f64; MAX_BASIS_COLUMNS], RegressionError> {
    let mut converged = false;
    for _ in 0..JACOBI_MAX_SWEEPS {
        converged = true;
        for left in 0..COLUMNS - 1 {
            for right in left + 1..COLUMNS {
                let (left_norm, right_norm, cross) =
                    column_products::<COLUMNS>(matrix, left, right);
                if !left_norm.is_finite()
                    || !right_norm.is_finite()
                    || !cross.is_finite()
                    || left_norm <= 0.0
                    || right_norm <= 0.0
                {
                    return Err(RegressionError::NonFiniteFit {
                        basis_degree: right,
                    });
                }
                if cross.abs() <= JACOBI_RELATIVE_TOLERANCE * (left_norm * right_norm).sqrt() {
                    continue;
                }
                converged = false;
                let zeta = (right_norm - left_norm) / (2.0 * cross);
                let tangent = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + zeta.hypot(1.0))
                };
                // Construction bounds |tangent| by one, so this direct norm cannot overflow or
                // underflow and need not pay for the general-magnitude `hypot` rescaling path.
                debug_assert!(tangent.abs() <= 1.0);
                let cosine = tangent.mul_add(tangent, 1.0).sqrt().recip();
                let sine = cosine * tangent;
                for row in 0..COLUMNS {
                    let left_index = left * MAX_BASIS_COLUMNS + row;
                    let right_index = right * MAX_BASIS_COLUMNS + row;
                    let left_value = matrix[left_index];
                    let right_value = matrix[right_index];
                    matrix[left_index] = cosine.mul_add(left_value, -sine * right_value);
                    matrix[right_index] = sine.mul_add(left_value, cosine * right_value);
                }
            }
        }
        if converged {
            break;
        }
    }
    if !converged {
        return Err(RegressionError::ConditionDiagnosticDidNotConverge {
            columns: COLUMNS,
            sweeps: JACOBI_MAX_SWEEPS,
        });
    }

    let mut singular_values = [0.0; MAX_BASIS_COLUMNS];
    for column in 0..COLUMNS {
        let mut squared_norm = 0.0;
        for row in 0..COLUMNS {
            squared_norm += matrix[column * MAX_BASIS_COLUMNS + row].powi(2);
        }
        singular_values[column] = squared_norm.sqrt();
    }
    singular_values[..COLUMNS].sort_by(|left, right| right.total_cmp(left));
    Ok(singular_values)
}

fn column_products<const ROWS: usize>(
    matrix: &[f64; MAX_BASIS_COLUMNS * MAX_BASIS_COLUMNS],
    left: usize,
    right: usize,
) -> (f64, f64, f64) {
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    let mut cross = 0.0;
    for row in 0..ROWS {
        let left_value = matrix[left * MAX_BASIS_COLUMNS + row];
        let right_value = matrix[right * MAX_BASIS_COLUMNS + row];
        left_norm = left_value.mul_add(left_value, left_norm);
        right_norm = right_value.mul_add(right_value, right_norm);
        cross = left_value.mul_add(right_value, cross);
    }
    (left_norm, right_norm, cross)
}

/// Allocation-reusing wrapper used only by the internal Criterion evidence gate.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[derive(Debug)]
pub struct RegressionBenchmark {
    workspace: RegressionWorkspace,
    spec: PolynomialRegressionSpec,
    math_backend: CpuMathBackend,
}

#[cfg(feature = "bench-internals")]
impl RegressionBenchmark {
    /// Creates a benchmark wrapper around an already validated estimator specification.
    #[must_use]
    pub fn new(spec: PolynomialRegressionSpec) -> Self {
        Self {
            workspace: RegressionWorkspace::default(),
            spec,
            math_backend: CpuMathBackend::RustStdF64,
        }
    }

    /// Creates a benchmark wrapper using one explicit CPU numerical implementation.
    #[must_use]
    pub fn new_with_math_backend(
        spec: PolynomialRegressionSpec,
        math_backend: CpuMathBackend,
    ) -> Self {
        Self {
            workspace: RegressionWorkspace::default(),
            spec,
            math_backend,
        }
    }

    /// Fits once and returns diagnostics plus one prediction to prevent dead-code elimination.
    ///
    /// # Errors
    ///
    /// Propagates the production fail-fast regression contract.
    pub fn fit(
        &mut self,
        predictors: &[f64],
        responses: &[f64],
    ) -> Result<(RegressionDiagnostics, f64), RegressionError> {
        let model = self.workspace.fit_with_math_backend(
            predictors,
            responses,
            self.spec,
            self.math_backend,
        )?;
        Ok((model.diagnostics(), model.predict(predictors[0])?))
    }
}

/// Fitted-model wrapper used only by the internal prediction Criterion evidence gate.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[derive(Debug)]
pub struct RegressionPredictionBenchmark {
    model: PolynomialModel,
}

#[cfg(feature = "bench-internals")]
impl RegressionPredictionBenchmark {
    /// Fits one model outside the timed prediction loop.
    ///
    /// # Errors
    ///
    /// Propagates the production fail-fast regression contract.
    pub fn fit(
        predictors: &[f64],
        responses: &[f64],
        spec: PolynomialRegressionSpec,
    ) -> Result<Self, RegressionError> {
        let model = RegressionWorkspace::default().fit_with_math_backend(
            predictors,
            responses,
            spec,
            CpuMathBackend::RustStdF64,
        )?;
        Ok(Self { model })
    }

    /// Evaluates the fitted production recurrence once.
    ///
    /// # Errors
    ///
    /// Propagates the production fail-fast prediction contract.
    pub fn predict(&self, predictor: f64) -> Result<f64, RegressionError> {
        self.model.predict(predictor)
    }

    /// Evaluates the fitted degree-four recurrence after caller-hoisted degree dispatch.
    ///
    /// # Errors
    ///
    /// Propagates the production fail-fast prediction contract.
    pub fn predict_degree_four(&self, predictor: f64) -> Result<f64, RegressionError> {
        self.model.predict_for::<4>(predictor)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use super::*;

    fn spec(degree: usize, condition_limit: f64) -> PolynomialRegressionSpec {
        PolynomialRegressionSpec::try_new(
            PolynomialDegree::try_new(degree).unwrap(),
            1.0e-12,
            condition_limit,
        )
        .unwrap()
    }

    fn representative_population(observations: usize) -> (Vec<f64>, Vec<f64>) {
        let observation_count = u32::try_from(observations).unwrap();
        let predictors = (0..observations)
            .map(|index| {
                let index = u32::try_from(index).unwrap();
                let permuted = (u64::from(index) * 104_729) % u64::from(observation_count);
                let unit =
                    f64::from(u32::try_from(permuted).unwrap()) / f64::from(observation_count);
                0.82 + 0.56 * unit + 0.015 * (17.0 * unit).sin()
            })
            .collect::<Vec<_>>();
        let responses = predictors
            .iter()
            .map(|&x| {
                2.0 - 0.7 * x + 0.35 * x.powi(2) - 0.08 * x.powi(3)
                    + 0.015 * x.powi(4)
                    + 0.002 * (31.0 * x).cos()
            })
            .collect();
        (predictors, responses)
    }

    fn svd_oracle(
        predictors: &[f64],
        responses: &[f64],
        model: &PolynomialModel,
    ) -> (Vec<f64>, f64) {
        let columns = model.degree.columns();
        let design = DMatrix::from_fn(predictors.len(), columns, |row, column| {
            let z = if columns == 1 {
                0.0
            } else {
                (predictors[row] - model.predictor_mean) / model.predictor_scale
            };
            z.powi(i32::try_from(column).unwrap())
        });
        let svd = design.clone().svd(true, true);
        let largest = svd.singular_values.max();
        let smallest = svd.singular_values.min();
        let coefficients = svd
            .solve(&DVector::from_column_slice(responses), 1.0e-12)
            .unwrap();
        let predictions = design * coefficients;
        (predictions.iter().copied().collect(), largest / smallest)
    }

    fn specialized_prediction(
        model: &PolynomialModel,
        degree: usize,
        predictor: f64,
    ) -> Result<f64, RegressionError> {
        match degree {
            0 => model.predict_for::<0>(predictor),
            1 => model.predict_for::<1>(predictor),
            2 => model.predict_for::<2>(predictor),
            3 => model.predict_for::<3>(predictor),
            4 => model.predict_for::<4>(predictor),
            _ => unreachable!(),
        }
    }

    fn independent_condition_and_rank(
        predictors: &[f64],
        degree: usize,
        rank_tolerance: f64,
    ) -> (f64, usize) {
        let count = u32::try_from(predictors.len()).unwrap();
        let mean = predictors.iter().sum::<f64>() / f64::from(count);
        let scale = (predictors
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / f64::from(count))
        .sqrt();
        let columns = degree + 1;
        let design = DMatrix::from_fn(predictors.len(), columns, |row, column| {
            ((predictors[row] - mean) / scale).powi(i32::try_from(column).unwrap())
        });
        let singular_values = design.svd(false, false).singular_values;
        let largest = singular_values.max();
        let smallest = singular_values.min();
        let rank = singular_values
            .iter()
            .filter(|&&value| value > rank_tolerance * largest)
            .count();
        (largest / smallest, rank)
    }

    fn assert_close(actual: f64, expected: f64, relative: f64, absolute: f64) {
        let tolerance = absolute.max(relative * expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }

    #[test]
    fn validates_configuration_without_substitution() {
        assert!(matches!(
            PolynomialDegree::try_new(5),
            Err(RegressionError::UnsupportedDegree { observed: 5, .. })
        ));
        let degree = PolynomialDegree::try_new(4).unwrap();
        assert!(matches!(
            PolynomialRegressionSpec::try_new(degree, 0.0, 1.0e8),
            Err(RegressionError::InvalidRankTolerance { .. })
        ));
        assert!(matches!(
            PolynomialRegressionSpec::try_new(degree, 1.0e-6, 1.0e7),
            Err(RegressionError::InvalidConditionLimit { .. })
        ));
    }

    #[test]
    fn agrees_with_independent_svd_on_predictions_and_conditioning() {
        for (degree, observations) in [
            (0, 500),
            (1, 500),
            (2, 500),
            (3, 500),
            (4, 500),
            (4, 65_536),
        ] {
            let (predictors, responses) = representative_population(observations);
            let mut workspace = RegressionWorkspace::default();
            let model = workspace
                .fit(&predictors, &responses, spec(degree, 1.0e8))
                .unwrap();
            let (oracle_predictions, oracle_condition) =
                svd_oracle(&predictors, &responses, &model);

            for (&predictor, &oracle) in predictors.iter().zip(&oracle_predictions) {
                let runtime = model.predict(predictor).unwrap();
                let specialized = specialized_prediction(&model, degree, predictor).unwrap();
                assert_eq!(runtime.to_bits(), specialized.to_bits());
                assert_close(runtime, oracle, 2.0e-11, 2.0e-12);
            }
            assert_close(
                model.diagnostics().condition_number(),
                oracle_condition,
                2.0e-11,
                2.0e-12,
            );
            assert_eq!(model.diagnostics().rank(), degree + 1);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apple_accelerate_agrees_with_independent_svd_on_predictions_and_conditioning() {
        for (degree, observations) in [
            (0, 500),
            (1, 500),
            (2, 500),
            (3, 500),
            (4, 500),
            (4, 65_536),
        ] {
            let (predictors, responses) = representative_population(observations);
            let mut workspace = RegressionWorkspace::default();
            let model = workspace
                .fit_with_math_backend(
                    &predictors,
                    &responses,
                    spec(degree, 1.0e8),
                    CpuMathBackend::AppleAccelerate,
                )
                .unwrap();
            let (oracle_predictions, oracle_condition) =
                svd_oracle(&predictors, &responses, &model);

            for (&predictor, &oracle) in predictors.iter().zip(&oracle_predictions) {
                assert_close(model.predict(predictor).unwrap(), oracle, 2.0e-11, 2.0e-12);
            }
            assert_close(
                model.diagnostics().condition_number(),
                oracle_condition,
                2.0e-11,
                2.0e-12,
            );
            assert_eq!(model.diagnostics().rank(), degree + 1);
        }
    }

    #[test]
    fn recovers_the_declared_polynomial_space_after_affine_rescaling() {
        let predictors = (0..101)
            .map(|index| -1.0 + 2.0 * f64::from(index) / 100.0)
            .collect::<Vec<_>>();
        let responses = predictors
            .iter()
            .map(|&x| 1.25 - 0.75 * x + 0.5 * x.powi(2) + 0.2 * x.powi(4))
            .collect::<Vec<_>>();
        let transformed = predictors
            .iter()
            .map(|&x| 1_000_000.0 + 0.001 * x)
            .collect::<Vec<_>>();
        let mut first_workspace = RegressionWorkspace::default();
        let first = first_workspace
            .fit(&predictors, &responses, spec(4, 1.0e8))
            .unwrap();
        let mut second_workspace = RegressionWorkspace::default();
        let second = second_workspace
            .fit(&transformed, &responses, spec(4, 1.0e8))
            .unwrap();

        for ((&x, &affine_x), &expected) in predictors.iter().zip(&transformed).zip(&responses) {
            assert_close(first.predict(x).unwrap(), expected, 3.0e-12, 3.0e-12);
            assert_close(second.predict(affine_x).unwrap(), expected, 2.0e-7, 2.0e-9);
        }
    }

    #[test]
    fn degree_zero_accepts_constant_predictors_and_returns_the_mean() {
        let predictors = [2.0, 2.0, 2.0, 2.0];
        let responses = [1.0, 2.0, 3.0, 4.0];
        let mut workspace = RegressionWorkspace::default();
        let model = workspace
            .fit(&predictors, &responses, spec(0, 1.0))
            .unwrap();
        assert_close(model.predict(100.0).unwrap(), 2.5, 0.0, f64::EPSILON);
        assert_eq!(
            model.diagnostics().condition_number().to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn rejects_insufficient_degenerate_and_rank_deficient_populations() {
        let mut workspace = RegressionWorkspace::default();
        assert!(matches!(
            workspace.fit(&[1.0; 4], &[2.0; 4], spec(4, 1.0e8)),
            Err(RegressionError::InsufficientObservations { .. })
        ));
        assert!(matches!(
            workspace.fit(&[1.0; 5], &[2.0; 5], spec(4, 1.0e8)),
            Err(RegressionError::DegeneratePredictor { .. })
        ));
        let rank_predictors = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let (_, oracle_rank) = independent_condition_and_rank(&rank_predictors, 4, 1.0e-12);
        let rank_error = workspace
            .fit(
                &rank_predictors,
                &[1.0, 4.0, 1.0, 4.0, 1.0, 4.0],
                spec(4, 1.0e8),
            )
            .unwrap_err();
        assert!(
            matches!(
                rank_error,
                RegressionError::RankDeficient { rank, columns: 5, .. } if rank == oracle_rank
            ),
            "{rank_error:?}"
        );
    }

    #[test]
    fn rejects_a_full_rank_design_outside_the_condition_domain() {
        let predictors = [
            1.0,
            1.0 + 3.0e-4,
            1.0 + 6.0e-4,
            2.0,
            2.0 + 3.0e-4,
            2.0 + 6.0e-4,
        ];
        let responses = [1.0, 1.1, 0.9, 2.0, 2.1, 1.9];
        let (oracle_condition, oracle_rank) =
            independent_condition_and_rank(&predictors, 4, 1.0e-12);
        assert_eq!(oracle_rank, 5);
        let mut workspace = RegressionWorkspace::default();
        let condition_error = workspace
            .fit(&predictors, &responses, spec(4, 1.0e6))
            .unwrap_err();
        let RegressionError::ConditionLimitExceeded {
            observed, limit, ..
        } = condition_error
        else {
            panic!("{condition_error:?}");
        };
        assert!(observed > limit);
        assert_close(observed, oracle_condition, 2.0e-5, 1.0e-3);
    }

    #[test]
    fn validates_all_values_before_reusing_workspace() {
        let mut workspace = RegressionWorkspace::default();
        let valid = workspace
            .fit(&[0.0, 1.0], &[2.0, 3.0], spec(1, 1.0e8))
            .unwrap();
        assert!(matches!(
            workspace.fit(&[0.0, f64::NAN], &[2.0, 3.0], spec(1, 1.0e8)),
            Err(RegressionError::NonFinitePredictor { index: 1, .. })
        ));
        assert!(matches!(
            workspace.fit(&[0.0, 1.0], &[2.0, f64::INFINITY], spec(1, 1.0e8)),
            Err(RegressionError::NonFiniteResponse { index: 1, .. })
        ));
        assert_close(valid.predict(0.5).unwrap(), 2.5, 0.0, 16.0 * f64::EPSILON);
    }
}
