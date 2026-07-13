//! Explicit deterministic market-model parameters.

use serde::Serialize;

use crate::InputError;

/// Strictly positive finite annualized decimal volatility.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[repr(transparent)]
pub struct ConstantVolatility(f64);

impl ConstantVolatility {
    /// Validates a constant annualized decimal volatility.
    ///
    /// A value such as `0.20` represents 20% volatility. Percent-valued inputs are not inferred or
    /// converted. The deterministic zero-volatility limit is outside the first supported model.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::InvalidConstantVolatility`] for zero, negative, NaN, or infinite input.
    pub fn try_new(annualized_decimal: f64) -> Result<Self, InputError> {
        if !annualized_decimal.is_finite() || annualized_decimal <= 0.0 {
            return Err(InputError::InvalidConstantVolatility {
                observed: annualized_decimal,
            });
        }
        Ok(Self(annualized_decimal))
    }

    /// Returns the annualized decimal volatility.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_decimal_volatility_and_rejects_unsupported_limit() {
        let sigma = ConstantVolatility::try_new(0.20).unwrap();
        assert_eq!(sigma.get().to_bits(), 0.20_f64.to_bits());
        assert!(matches!(
            ConstantVolatility::try_new(0.0),
            Err(InputError::InvalidConstantVolatility { observed: 0.0 })
        ));
        assert!(matches!(
            ConstantVolatility::try_new(f64::NAN),
            Err(InputError::InvalidConstantVolatility { .. })
        ));
    }
}
