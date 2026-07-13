//! Exact node identity and strictly ordered model time.

use serde::Serialize;

use crate::InputError;

/// Stable zero-based identity of a node in a validated time grid.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(transparent)]
pub struct NodeId(u32);

impl NodeId {
    /// Returns the zero-based integer identity.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    pub(crate) fn from_index(index: usize) -> Result<Self, InputError> {
        let value = u32::try_from(index).map_err(|_| InputError::TooManyNodes {
            nodes: index.saturating_add(1),
        })?;
        Ok(Self(value))
    }
}

/// Immutable model times beginning at the exact valuation origin.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TimeGrid {
    times: Box<[f64]>,
}

impl TimeGrid {
    /// Validates and owns a sequence of strictly increasing finite year fractions.
    ///
    /// # Errors
    ///
    /// Returns [`InputError`] when the grid is empty, does not begin at exactly zero,
    /// contains a non-finite value, is not strictly increasing, or exceeds the node-ID domain.
    pub fn try_new(times: impl Into<Box<[f64]>>) -> Result<Self, InputError> {
        let times = times.into();
        let Some(&origin) = times.first() else {
            return Err(InputError::EmptyTimeGrid);
        };
        if !origin.is_finite() {
            return Err(InputError::NonFiniteTime {
                node: 0,
                observed: origin,
            });
        }
        if origin.to_bits() != 0.0_f64.to_bits() {
            return Err(InputError::InvalidTimeOrigin { observed: origin });
        }
        NodeId::from_index(times.len() - 1)?;

        for (index, &observed) in times.iter().enumerate().skip(1) {
            if !observed.is_finite() {
                return Err(InputError::NonFiniteTime {
                    node: index,
                    observed,
                });
            }
            let previous = times[index - 1];
            if observed <= previous {
                return Err(InputError::NonIncreasingTime {
                    previous_node: index - 1,
                    previous,
                    node: index,
                    observed,
                });
            }
        }
        Ok(Self { times })
    }

    /// Returns the number of nodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns whether the grid has no nodes.
    ///
    /// A validated grid is never empty; this method supports conventional collection APIs.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Borrows all year fractions.
    #[must_use]
    pub const fn as_slice(&self) -> &[f64] {
        &self.times
    }

    /// Returns the exact year fraction attached to a node.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::NodeOutOfBounds`] when the node is not in this grid.
    pub fn time(&self, node: NodeId) -> Result<f64, InputError> {
        self.times
            .get(node.0 as usize)
            .copied()
            .ok_or(InputError::NodeOutOfBounds {
                node: node.0,
                nodes: self.len(),
            })
    }

    /// Returns the canonical identity for a zero-based node index.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::NodeOutOfBounds`] when the index is not in this grid.
    pub fn node_id(&self, index: usize) -> Result<NodeId, InputError> {
        if index >= self.len() {
            return Err(InputError::NodeOutOfBounds {
                node: u32::try_from(index).unwrap_or(u32::MAX),
                nodes: self.len(),
            });
        }
        NodeId::from_index(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_irregular_grid_and_preserves_identity() {
        let grid = TimeGrid::try_new([0.0, 1.0 / 365.0, 0.25, 1.0]).unwrap();
        assert_eq!(grid.len(), 4);
        let node = grid.node_id(2).unwrap();
        assert_eq!(node.get(), 2);
        assert_eq!(grid.time(node).unwrap().to_bits(), 0.25_f64.to_bits());
    }

    #[test]
    fn rejects_nonzero_origin() {
        assert_eq!(
            TimeGrid::try_new([f64::EPSILON, 1.0]).unwrap_err(),
            InputError::InvalidTimeOrigin {
                observed: f64::EPSILON,
            }
        );
        assert!(matches!(
            TimeGrid::try_new([-0.0, 1.0]),
            Err(InputError::InvalidTimeOrigin { observed }) if observed.is_sign_negative()
        ));
    }

    #[test]
    fn rejects_duplicate_or_reversed_nodes() {
        assert!(matches!(
            TimeGrid::try_new([0.0, 0.25, 0.25]),
            Err(InputError::NonIncreasingTime { node: 2, .. })
        ));
        assert!(matches!(
            TimeGrid::try_new([0.0, 0.25, 0.20]),
            Err(InputError::NonIncreasingTime { node: 2, .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_time() {
        assert!(matches!(
            TimeGrid::try_new([0.0, f64::NAN]),
            Err(InputError::NonFiniteTime { node: 1, .. })
        ));
    }
}
