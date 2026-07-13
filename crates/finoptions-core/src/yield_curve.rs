//! Node-aligned deterministic continuous proportional-dividend yields.

use serde::Serialize;

use crate::{InputError, NodeId, TimeGrid};

/// Immutable cumulative yields `Q(t) = integral_0^t q(u) du` aligned with a [`TimeGrid`].
///
/// This type represents only a nonnegative continuous proportional cash-payout yield. Financing,
/// borrow, convenience yield, and deterministic cash dividends are separate financial concepts.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CumulativeDividendYieldCurve {
    grid: TimeGrid,
    cumulative_yields: Box<[f64]>,
}

impl CumulativeDividendYieldCurve {
    /// Validates and owns node-aligned cumulative continuous dividend yields.
    ///
    /// Values must begin at canonical positive zero and be finite and nondecreasing. Every
    /// adjacent proportional payout factor `exp(-(Q(to) - Q(from)))` must be representable as a
    /// finite, strictly positive `f64`. No interpolation or annualization convention is inferred.
    ///
    /// # Errors
    ///
    /// Returns [`InputError`] for a length mismatch, nonzero origin, non-finite value, decreasing
    /// cumulative yield, or unrepresentable adjacent payout factor.
    pub fn try_new(
        grid: &TimeGrid,
        cumulative_yields: impl Into<Box<[f64]>>,
    ) -> Result<Self, InputError> {
        let cumulative_yields = cumulative_yields.into();
        if cumulative_yields.len() != grid.len() {
            return Err(InputError::DividendYieldCurveLengthMismatch {
                expected: grid.len(),
                observed: cumulative_yields.len(),
            });
        }
        if cumulative_yields[0].to_bits() != 0.0_f64.to_bits() {
            return Err(InputError::InvalidCumulativeDividendYieldOrigin {
                observed: cumulative_yields[0],
            });
        }
        for (node, &observed) in cumulative_yields.iter().enumerate() {
            if !observed.is_finite()
                || observed < 0.0
                || (observed == 0.0 && observed.is_sign_negative())
            {
                return Err(InputError::InvalidCumulativeDividendYield { node, observed });
            }
            if node == 0 {
                continue;
            }
            let previous = cumulative_yields[node - 1];
            if observed < previous {
                return Err(InputError::DecreasingCumulativeDividendYield {
                    previous_node: node - 1,
                    previous,
                    node,
                    observed,
                });
            }
            let interval_yield = observed - previous;
            let payout_discount = (-interval_yield).exp();
            if !payout_discount.is_finite() || payout_discount <= 0.0 {
                return Err(InputError::UnrepresentableIntervalDividendYield {
                    from_node: node - 1,
                    to_node: node,
                    interval_yield,
                });
            }
        }
        Ok(Self {
            grid: grid.clone(),
            cumulative_yields,
        })
    }

    /// Borrows the exact time grid against which the curve was validated.
    #[must_use]
    pub const fn grid(&self) -> &TimeGrid {
        &self.grid
    }

    /// Borrows all `Q(t) = integral_0^t q(u) du` values.
    #[must_use]
    pub const fn as_slice(&self) -> &[f64] {
        &self.cumulative_yields
    }

    /// Returns `Q(t_to) - Q(t_from)`.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::NodeOutOfBounds`] if either node is outside the curve.
    pub fn interval_yield(&self, from: NodeId, to: NodeId) -> Result<f64, InputError> {
        let nodes = self.cumulative_yields.len();
        let from_value =
            self.cumulative_yields
                .get(from.get() as usize)
                .ok_or(InputError::NodeOutOfBounds {
                    node: from.get(),
                    nodes,
                })?;
        let to_value =
            self.cumulative_yields
                .get(to.get() as usize)
                .ok_or(InputError::NodeOutOfBounds {
                    node: to.get(),
                    nodes,
                })?;
        Ok(to_value - from_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_zero_and_positive_interval_yields() {
        let grid = TimeGrid::try_new([0.0, 0.5, 1.0]).unwrap();
        let curve = CumulativeDividendYieldCurve::try_new(&grid, [0.0, 0.01, 0.03]).unwrap();
        assert_eq!(curve.as_slice(), &[0.0, 0.01, 0.03]);
        let observed = curve
            .interval_yield(grid.node_id(1).unwrap(), grid.node_id(2).unwrap())
            .unwrap();
        assert!((observed - 0.02).abs() <= 4.0 * f64::EPSILON);

        let zero = CumulativeDividendYieldCurve::try_new(&grid, [0.0; 3]).unwrap();
        assert_eq!(zero.as_slice(), &[0.0; 3]);
    }

    #[test]
    fn rejects_ambiguous_or_unrepresentable_curves() {
        let grid = TimeGrid::try_new([0.0, 0.5, 1.0]).unwrap();
        assert!(matches!(
            CumulativeDividendYieldCurve::try_new(&grid, [0.0, 0.01]),
            Err(InputError::DividendYieldCurveLengthMismatch { .. })
        ));
        assert!(matches!(
            CumulativeDividendYieldCurve::try_new(&grid, [-0.0, 0.01, 0.02]),
            Err(InputError::InvalidCumulativeDividendYieldOrigin { observed })
                if observed.is_sign_negative()
        ));
        assert!(matches!(
            CumulativeDividendYieldCurve::try_new(&grid, [0.0, 0.02, 0.01]),
            Err(InputError::DecreasingCumulativeDividendYield { node: 2, .. })
        ));
        assert!(matches!(
            CumulativeDividendYieldCurve::try_new(&grid, [0.0, f64::INFINITY, f64::INFINITY]),
            Err(InputError::InvalidCumulativeDividendYield { node: 1, .. })
        ));
        assert!(matches!(
            CumulativeDividendYieldCurve::try_new(&grid, [0.0, -0.0, 0.01]),
            Err(InputError::InvalidCumulativeDividendYield { node: 1, observed })
                if observed.is_sign_negative()
        ));
        assert!(matches!(
            CumulativeDividendYieldCurve::try_new(&grid, [0.0, 800.0, 801.0]),
            Err(InputError::UnrepresentableIntervalDividendYield { from_node: 0, .. })
        ));
    }
}
