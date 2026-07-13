//! Node-aligned deterministic discount factors.

use serde::Serialize;

use crate::{InputError, NodeId, TimeGrid};

/// Immutable `log(P(0,t))` values aligned one-for-one with a [`TimeGrid`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LogDiscountCurve {
    log_discounts: Box<[f64]>,
}

impl LogDiscountCurve {
    /// Validates and owns node-aligned log discount factors.
    ///
    /// Negative rates and discount factors above one are permitted. Every adjacent interval
    /// discount must nevertheless be representable as a finite, strictly positive `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`InputError`] for a length mismatch, nonzero origin, non-finite value, or an
    /// interval whose exponentiated discount is not representable.
    pub fn try_new(
        grid: &TimeGrid,
        log_discounts: impl Into<Box<[f64]>>,
    ) -> Result<Self, InputError> {
        let log_discounts = log_discounts.into();
        if log_discounts.len() != grid.len() {
            return Err(InputError::CurveLengthMismatch {
                expected: grid.len(),
                observed: log_discounts.len(),
            });
        }
        if log_discounts[0].to_bits() != 0.0_f64.to_bits() {
            return Err(InputError::InvalidDiscountOrigin {
                observed: log_discounts[0],
            });
        }
        for (node, &observed) in log_discounts.iter().enumerate() {
            if !observed.is_finite() {
                return Err(InputError::NonFiniteLogDiscount { node, observed });
            }
            if node == 0 {
                continue;
            }
            let interval_log_discount = observed - log_discounts[node - 1];
            let discount = interval_log_discount.exp();
            if !discount.is_finite() || discount <= 0.0 {
                return Err(InputError::UnrepresentableIntervalDiscount {
                    from_node: node - 1,
                    to_node: node,
                    interval_log_discount,
                });
            }
        }
        Ok(Self { log_discounts })
    }

    /// Borrows all `log(P(0,t))` values.
    #[must_use]
    pub const fn as_slice(&self) -> &[f64] {
        &self.log_discounts
    }

    /// Returns `log(P(0,t_to) / P(0,t_from))`.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::NodeOutOfBounds`] if either node is outside the curve.
    pub fn interval_log_discount(&self, from: NodeId, to: NodeId) -> Result<f64, InputError> {
        let nodes = self.log_discounts.len();
        let from_value =
            self.log_discounts
                .get(from.get() as usize)
                .ok_or(InputError::NodeOutOfBounds {
                    node: from.get(),
                    nodes,
                })?;
        let to_value =
            self.log_discounts
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
    fn accepts_positive_and_negative_interval_rates() {
        let grid = TimeGrid::try_new([0.0, 0.5, 1.0]).unwrap();
        let curve = LogDiscountCurve::try_new(&grid, [0.0, -0.01, 0.005]).unwrap();
        let from = grid.node_id(1).unwrap();
        let to = grid.node_id(2).unwrap();
        let observed = curve.interval_log_discount(from, to).unwrap();
        assert!((observed - 0.015).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn rejects_wrong_origin_and_nonfinite_values() {
        let grid = TimeGrid::try_new([0.0, 1.0]).unwrap();
        assert!(matches!(
            LogDiscountCurve::try_new(&grid, [0.001, -0.01]),
            Err(InputError::InvalidDiscountOrigin { .. })
        ));
        assert!(matches!(
            LogDiscountCurve::try_new(&grid, [0.0, f64::INFINITY]),
            Err(InputError::NonFiniteLogDiscount { node: 1, .. })
        ));
        assert!(matches!(
            LogDiscountCurve::try_new(&grid, [-0.0, -0.01]),
            Err(InputError::InvalidDiscountOrigin { observed }) if observed.is_sign_negative()
        ));
    }
}
