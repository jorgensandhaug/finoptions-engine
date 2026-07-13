//! One exact geometric-Brownian-motion transition.

use finoptions_core::{ConstantVolatility, LogDiscountCurve, NodeId};

use crate::SimulationError;

/// Precomputed exact GBM coefficients for one adjacent event interval.
///
/// Under zero continuous yield/borrow, deterministic discount factors, and constant volatility,
/// the pre-event state at the next node is
/// `S_next = S_after * exp(log_drift + diffusion * Z)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GbmStep {
    from: NodeId,
    to: NodeId,
    log_drift: f64,
    diffusion: f64,
}

impl GbmStep {
    /// Precomputes one exact adjacent-node transition from a validated curve and volatility.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError`] if either node is outside the curve or the nodes are not adjacent
    /// and increasing.
    pub fn try_new(
        curve: &LogDiscountCurve,
        from: NodeId,
        to: NodeId,
        volatility: ConstantVolatility,
    ) -> Result<Self, SimulationError> {
        if from.get().checked_add(1) != Some(to.get()) {
            return Err(SimulationError::NonAdjacentNodes {
                from: from.get(),
                to: to.get(),
            });
        }
        let from_time = curve.grid().time(from)?;
        let to_time = curve.grid().time(to)?;
        let dt = to_time - from_time;
        let interval_log_discount = curve.interval_log_discount(from, to)?;
        let variance = volatility.get().mul_add(volatility.get(), 0.0) * dt;
        let log_drift = -interval_log_discount - 0.5 * variance;
        let diffusion = variance.sqrt();
        Ok(Self {
            from,
            to,
            log_drift,
            diffusion,
        })
    }

    /// Advances one post-event stock state to the next node's pre-event state.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError`] for an invalid state/variate or a non-finite result. Zero is an
    /// absorbing stock state under the configured GBM dynamics.
    pub fn advance(self, spot_after: f64, normal: f64) -> Result<f64, SimulationError> {
        if !spot_after.is_finite()
            || spot_after < 0.0
            || (spot_after == 0.0 && spot_after.is_sign_negative())
        {
            return Err(SimulationError::InvalidPostEventSpot {
                observed: spot_after,
            });
        }
        if !normal.is_finite() {
            return Err(SimulationError::NonFiniteNormalVariate { observed: normal });
        }
        let next = self.advance_validated(spot_after, normal);
        if !next.is_finite() {
            return Err(SimulationError::NonFiniteTransition {
                from: self.from.get(),
                to: self.to.get(),
                spot: spot_after,
                normal,
            });
        }
        Ok(next)
    }

    pub(crate) fn advance_validated(self, spot_after: f64, normal: f64) -> f64 {
        if spot_after == 0.0 {
            return 0.0;
        }
        let exponent = self.diffusion.mul_add(normal, self.log_drift);
        spot_after * exponent.exp()
    }

    /// Returns the start node.
    #[must_use]
    pub const fn from(self) -> NodeId {
        self.from
    }

    /// Returns the end node.
    #[must_use]
    pub const fn to(self) -> NodeId {
        self.to
    }

    /// Returns the exact interval log drift applied before the random term.
    #[must_use]
    pub const fn log_drift(self) -> f64 {
        self.log_drift
    }

    /// Returns `sigma * sqrt(dt)` for the interval.
    #[must_use]
    pub const fn diffusion(self) -> f64 {
        self.diffusion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use finoptions_core::TimeGrid;

    fn step(log_discount_at_one: f64, sigma: f64) -> GbmStep {
        let grid = TimeGrid::try_new([0.0, 1.0]).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, [0.0, log_discount_at_one]).unwrap();
        GbmStep::try_new(
            &curve,
            grid.node_id(0).unwrap(),
            grid.node_id(1).unwrap(),
            ConstantVolatility::try_new(sigma).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn implements_the_exact_risk_neutral_transition_sign() {
        let step = step(-0.05, 0.20);
        let expected_log_drift = 0.05 - 0.5 * 0.20_f64.powi(2);
        assert!((step.log_drift() - expected_log_drift).abs() <= 4.0 * f64::EPSILON);
        assert!((step.diffusion() - 0.20).abs() <= f64::EPSILON);
        let expected = 100.0 * expected_log_drift.exp();
        assert!(
            (step.advance(100.0, 0.0).unwrap() - expected).abs() <= 8.0 * f64::EPSILON * expected
        );
    }

    #[test]
    fn supports_negative_rates_without_changing_the_model() {
        let step = step(0.01, 0.20);
        let expected_log_drift = -0.01 - 0.5 * 0.20_f64.powi(2);
        assert!((step.log_drift() - expected_log_drift).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn preserves_absorbing_zero_and_rejects_nonfinite_inputs() {
        let step = step(-0.05, 0.20);
        assert_eq!(
            step.advance(0.0, 20.0).unwrap().to_bits(),
            0.0_f64.to_bits()
        );
        assert!(matches!(
            step.advance(-0.0, 0.0),
            Err(SimulationError::InvalidPostEventSpot { observed }) if observed.is_sign_negative()
        ));
        assert!(matches!(
            step.advance(100.0, f64::NAN),
            Err(SimulationError::NonFiniteNormalVariate { .. })
        ));
    }

    #[test]
    fn rejects_skipped_nodes() {
        let grid = TimeGrid::try_new([0.0, 0.5, 1.0]).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, [0.0, -0.02, -0.05]).unwrap();
        let result = GbmStep::try_new(
            &curve,
            grid.node_id(0).unwrap(),
            grid.node_id(2).unwrap(),
            ConstantVolatility::try_new(0.20).unwrap(),
        );
        assert_eq!(
            result.unwrap_err(),
            SimulationError::NonAdjacentNodes { from: 0, to: 2 }
        );
    }
}
