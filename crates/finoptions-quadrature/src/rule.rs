//! Version-pinned standard-normal Gauss-Hermite nodes and weights.

mod generated;

use crate::QuadratureError;

pub(crate) const SUPPORTED_NORMAL_ORDERS: &str = "32, 64, 128, 256, 512, 1024, or 2048";

#[derive(Clone, Debug)]
pub(crate) struct StandardNormalRule {
    requested_order: usize,
    nodes: Box<[f64]>,
    weights: Box<[f64]>,
}

impl StandardNormalRule {
    pub(crate) fn try_new(order: usize) -> Result<Self, QuadratureError> {
        let positive = generated::positive_nodes_and_weights(order).ok_or(
            QuadratureError::UnsupportedQuadratureOrder {
                observed: order,
                supported: SUPPORTED_NORMAL_ORDERS,
            },
        )?;
        let effective_nodes = positive.len() * 2;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(effective_nodes)
            .map_err(|_| QuadratureError::Allocation {
                buffer: "standard-normal nodes",
                elements: effective_nodes,
            })?;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(effective_nodes)
            .map_err(|_| QuadratureError::Allocation {
                buffer: "standard-normal weights",
                elements: effective_nodes,
            })?;
        for &(node, weight) in positive.iter().rev() {
            nodes.push(-node);
            weights.push(weight);
        }
        for &(node, weight) in positive {
            nodes.push(node);
            weights.push(weight);
        }
        let rule = Self {
            requested_order: order,
            nodes: nodes.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
        };
        rule.validate()?;
        Ok(rule)
    }

    pub(crate) fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    pub(crate) fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn validate(&self) -> Result<(), QuadratureError> {
        if self.weights.len() != self.nodes.len() || self.nodes.is_empty() {
            return Err(self.invalid("node and weight counts differ or are empty"));
        }
        if self
            .nodes
            .iter()
            .zip(&self.weights)
            .any(|(&node, &weight)| !node.is_finite() || !weight.is_finite() || weight <= 0.0)
        {
            return Err(
                self.invalid("nodes must be finite and retained weights finite and positive")
            );
        }
        if self.nodes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(self.invalid("nodes are not strictly increasing"));
        }
        for index in 0..self.nodes.len() / 2 {
            let mirror = self.nodes.len() - index - 1;
            if self.nodes[index].to_bits() != (-self.nodes[mirror]).to_bits()
                || self.weights[index].to_bits() != self.weights[mirror].to_bits()
            {
                return Err(self.invalid("stored rule is not exactly symmetric"));
            }
        }

        let weight_sum = compensated_sum(self.weights.iter().copied());
        if (weight_sum - 1.0).abs() > 8_192.0 * f64::EPSILON {
            return Err(self.invalid("weights do not integrate the standard-normal mass"));
        }
        let mean = compensated_sum(
            self.nodes
                .iter()
                .zip(&self.weights)
                .map(|(&node, &weight)| node * weight),
        );
        if mean.abs() > 8_192.0 * f64::EPSILON {
            return Err(self.invalid("nodes and weights do not integrate the normal mean"));
        }
        let variance = compensated_sum(
            self.nodes
                .iter()
                .zip(&self.weights)
                .map(|(&node, &weight)| node * node * weight),
        );
        if (variance - 1.0).abs() > 32_768.0 * f64::EPSILON {
            return Err(self.invalid("nodes and weights do not integrate the normal variance"));
        }
        Ok(())
    }

    fn invalid(&self, reason: &'static str) -> QuadratureError {
        QuadratureError::InvalidNormalRule {
            order: self.requested_order,
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
    fn integrates_standard_normal_moments_at_every_supported_order() {
        for order in [32, 64, 128, 256, 512, 1_024, 2_048] {
            let rule = StandardNormalRule::try_new(order).unwrap();
            for (power, expected) in [(0_i32, 1.0), (1, 0.0), (2, 1.0), (4, 3.0), (6, 15.0)] {
                let observed = compensated_sum(
                    rule.nodes()
                        .iter()
                        .zip(rule.weights())
                        .map(|(&node, &weight)| node.powi(power) * weight),
                );
                assert!((observed - expected).abs() <= 2.0e-11, "order={order}");
            }
        }
    }

    #[test]
    fn rejects_unpinned_orders_instead_of_generating_another_rule() {
        assert!(matches!(
            StandardNormalRule::try_new(255),
            Err(QuadratureError::UnsupportedQuadratureOrder { observed: 255, .. })
        ));
    }
}
