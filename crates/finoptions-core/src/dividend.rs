//! Explicit deterministic cash-dividend semantics.

use serde::{Deserialize, Serialize};

use crate::{InputError, NodeId};

/// Limited-liability treatment of a declared deterministic cash dividend.
///
/// No variant is a default. Callers must state the financial policy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[non_exhaustive]
pub enum DividendPolicy {
    /// Pays `min(declared, spot_before)` and leaves `max(spot_before - declared, 0)`.
    Liquidator,
    /// Pays the declaration only when `spot_before > declared`; otherwise pays zero.
    Survivor,
}

/// A nonnegative deterministic cash amount attached to an exact event node.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CashDividend {
    node: NodeId,
    amount: f64,
}

impl CashDividend {
    /// Validates a cash dividend without inferring its event phase or policy.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::InvalidCashDividend`] when `amount` is negative or non-finite.
    pub fn try_new(node: NodeId, amount: f64) -> Result<Self, InputError> {
        if !amount.is_finite() || amount < 0.0 || (amount == 0.0 && amount.is_sign_negative()) {
            return Err(InputError::InvalidCashDividend {
                node: node.get(),
                observed: amount,
            });
        }
        Ok(Self { node, amount })
    }

    /// Returns the exact dividend event node.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.node
    }

    /// Returns the declared nonnegative cash amount.
    #[must_use]
    pub const fn amount(self) -> f64 {
        self.amount
    }

    /// Applies the explicitly selected limited-liability policy to a pre-event spot.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::InvalidSpotBeforeEvent`] when `spot_before` is negative or non-finite.
    pub fn apply(self, policy: DividendPolicy, spot_before: f64) -> Result<f64, InputError> {
        if !spot_before.is_finite()
            || spot_before < 0.0
            || (spot_before == 0.0 && spot_before.is_sign_negative())
        {
            return Err(InputError::InvalidSpotBeforeEvent {
                node: self.node.get(),
                observed: spot_before,
            });
        }
        let spot_after = match policy {
            DividendPolicy::Liquidator => (spot_before - self.amount).max(0.0),
            DividendPolicy::Survivor => {
                if spot_before > self.amount {
                    spot_before - self.amount
                } else {
                    spot_before
                }
            }
        };
        Ok(spot_after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeGrid;

    fn dividend(amount: f64) -> CashDividend {
        let grid = TimeGrid::try_new([0.0, 0.5]).unwrap();
        CashDividend::try_new(grid.node_id(1).unwrap(), amount).unwrap()
    }

    fn assert_exact(actual: f64, expected: f64) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn policies_have_distinct_stated_boundary_behavior() {
        let dividend = dividend(5.0);
        assert_exact(
            dividend.apply(DividendPolicy::Liquidator, 5.0).unwrap(),
            0.0,
        );
        assert_exact(dividend.apply(DividendPolicy::Survivor, 5.0).unwrap(), 5.0);
    }

    #[test]
    fn policies_preserve_limited_liability() {
        let dividend = dividend(5.0);
        assert_exact(
            dividend.apply(DividendPolicy::Liquidator, 2.0).unwrap(),
            0.0,
        );
        assert_exact(dividend.apply(DividendPolicy::Survivor, 2.0).unwrap(), 2.0);
        assert_exact(
            dividend.apply(DividendPolicy::Liquidator, 8.0).unwrap(),
            3.0,
        );
        assert_exact(dividend.apply(DividendPolicy::Survivor, 8.0).unwrap(), 3.0);
    }

    #[test]
    fn rejects_invalid_amount() {
        let grid = TimeGrid::try_new([0.0, 0.5]).unwrap();
        let node = grid.node_id(1).unwrap();
        assert!(matches!(
            CashDividend::try_new(node, f64::NAN),
            Err(InputError::InvalidCashDividend { .. })
        ));
        assert!(matches!(
            CashDividend::try_new(node, -f64::EPSILON),
            Err(InputError::InvalidCashDividend { .. })
        ));
        assert!(matches!(
            CashDividend::try_new(node, -0.0),
            Err(InputError::InvalidCashDividend { observed, .. }) if observed.is_sign_negative()
        ));
    }
}
