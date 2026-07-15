//! Version-pinned Gauss-Legendre nodes and weights.

mod generated;

use crate::AmericanOptionError;

pub(crate) const SUPPORTED_LEGENDRE_ORDERS: &str = "5, 8, 13, 15, 21, 31, 41, 51, 61, 81, or 131";

#[derive(Clone, Copy, Debug)]
pub(crate) struct LegendreRule {
    order: usize,
    nodes_and_weights: &'static [(f64, f64)],
}

impl LegendreRule {
    pub(crate) fn try_new(order: usize) -> Result<Self, AmericanOptionError> {
        let nodes_and_weights = generated::nodes_and_weights(order).ok_or(
            AmericanOptionError::UnsupportedLegendreOrder {
                observed: order,
                supported: SUPPORTED_LEGENDRE_ORDERS,
            },
        )?;
        let rule = Self {
            order,
            nodes_and_weights,
        };
        rule.validate()?;
        Ok(rule)
    }

    pub(crate) const fn nodes_and_weights(self) -> &'static [(f64, f64)] {
        self.nodes_and_weights
    }

    fn validate(self) -> Result<(), AmericanOptionError> {
        if self.nodes_and_weights.len() != self.order || self.nodes_and_weights.is_empty() {
            return Err(self.invalid("node count differs from the requested order"));
        }
        if self.nodes_and_weights.iter().any(|&(node, weight)| {
            !node.is_finite() || !weight.is_finite() || node <= -1.0 || node >= 1.0 || weight <= 0.0
        }) {
            return Err(self.invalid("nodes must be interior and weights finite and positive"));
        }
        if self
            .nodes_and_weights
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(self.invalid("nodes are not strictly increasing"));
        }
        for index in 0..self.order / 2 {
            let mirror = self.order - index - 1;
            let (node, weight) = self.nodes_and_weights[index];
            let (mirror_node, mirror_weight) = self.nodes_and_weights[mirror];
            if node.to_bits() != (-mirror_node).to_bits()
                || weight.to_bits() != mirror_weight.to_bits()
            {
                return Err(self.invalid("stored rule is not exactly symmetric"));
            }
        }
        let mass = compensated_sum(self.nodes_and_weights.iter().map(|&(_, weight)| weight));
        if (mass - 2.0).abs() > 1_024.0 * f64::EPSILON {
            return Err(self.invalid("weights do not integrate constant mass on [-1, 1]"));
        }
        let second = compensated_sum(
            self.nodes_and_weights
                .iter()
                .map(|&(node, weight)| node * node * weight),
        );
        if (second - 2.0 / 3.0).abs() > 4_096.0 * f64::EPSILON {
            return Err(self.invalid("rule does not integrate the second moment"));
        }
        Ok(())
    }

    const fn invalid(self, reason: &'static str) -> AmericanOptionError {
        AmericanOptionError::InvalidLegendreRule {
            order: self.order,
            reason,
        }
    }
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let updated = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - updated) + value
        } else {
            (value - updated) + sum
        };
        sum = updated;
    }
    sum + correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pinned_rule_integrates_admitted_polynomial_moments() {
        for order in [5, 8, 13, 15, 21, 31, 41, 51, 61, 81, 131] {
            let rule = LegendreRule::try_new(order).unwrap();
            for (power, expected) in [(0_i32, 2.0), (1, 0.0), (2, 2.0 / 3.0), (4, 0.4)] {
                let observed = compensated_sum(
                    rule.nodes_and_weights()
                        .iter()
                        .map(|&(node, weight)| node.powi(power) * weight),
                );
                assert!((observed - expected).abs() <= 5.0e-14, "order={order}");
            }
        }
    }
}
