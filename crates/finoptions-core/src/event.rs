//! Ordered exercise, dividend, and terminal events on an exact time grid.

use serde::Serialize;

use crate::{CashDividend, InputError, NodeId, TimeGrid};

/// Event phases attached to one node, interpreted in field order.
///
/// If both `exercise` and `dividend` are present, exercise occurs in the pre-dividend state and the
/// cash jump follows it. `expiry` is structurally exclusive because expiry must be the final node
/// and dividends/exercise decisions must precede it in the first supported slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodePhases<'a> {
    /// Whether the holder may exercise in the node's pre-dividend state.
    pub exercise: bool,
    /// The deterministic cash jump following exercise, if any.
    pub dividend: Option<&'a CashDividend>,
    /// Whether this is the separately ordered terminal payoff node.
    pub expiry: bool,
}

/// A validated finite exercise and cash-dividend schedule ending at exact expiry.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EventSchedule {
    grid: TimeGrid,
    dividends: Box<[CashDividend]>,
    exercise_nodes: Box<[NodeId]>,
    expiry: NodeId,
}

impl EventSchedule {
    /// Validates an event schedule without moving, merging, or dropping events.
    ///
    /// Dividends and exercise nodes must already be strictly increasing. A shared
    /// exercise/dividend node means exercise first and the cash jump second. The final node is
    /// terminal payoff, not another regression decision.
    ///
    /// # Errors
    ///
    /// Returns [`InputError`] when expiry is not the final positive-time node, an event lies outside
    /// its supported temporal domain, or an event type repeats a node.
    pub fn try_new(
        grid: TimeGrid,
        dividends: impl Into<Box<[CashDividend]>>,
        exercise_nodes: impl Into<Box<[NodeId]>>,
        expiry: NodeId,
    ) -> Result<Self, InputError> {
        let expected_index = grid.len().saturating_sub(1);
        let expected = u32::try_from(expected_index).unwrap_or(u32::MAX);
        if expected == 0 || expiry.get() != expected {
            return Err(InputError::InvalidExpiryNode {
                expected,
                observed: expiry.get(),
            });
        }

        let dividends = dividends.into();
        for (index, dividend) in dividends.iter().enumerate() {
            let node = dividend.node().get();
            if node == 0 || node >= expiry.get() {
                return Err(InputError::InvalidDividendNode {
                    node,
                    expiry: expiry.get(),
                });
            }
            if index > 0 {
                let previous = dividends[index - 1].node().get();
                if previous == node {
                    return Err(InputError::DuplicateDividendNode { node });
                }
                if previous > node {
                    return Err(InputError::UnorderedDividendNode { previous, node });
                }
            }
        }

        let exercise_nodes = exercise_nodes.into();
        for (index, node) in exercise_nodes.iter().copied().enumerate() {
            if node.get() >= expiry.get() {
                return Err(InputError::InvalidExerciseNode {
                    node: node.get(),
                    expiry: expiry.get(),
                });
            }
            if index > 0 {
                let previous = exercise_nodes[index - 1].get();
                if previous == node.get() {
                    return Err(InputError::DuplicateExerciseNode { node: node.get() });
                }
                if previous > node.get() {
                    return Err(InputError::UnorderedExerciseNode {
                        previous,
                        node: node.get(),
                    });
                }
            }
        }

        Ok(Self {
            grid,
            dividends,
            exercise_nodes,
            expiry,
        })
    }

    /// Borrows the exact model time grid.
    #[must_use]
    pub const fn grid(&self) -> &TimeGrid {
        &self.grid
    }

    /// Borrows deterministic cash dividends in increasing node order.
    #[must_use]
    pub const fn dividends(&self) -> &[CashDividend] {
        &self.dividends
    }

    /// Borrows exercise opportunities in increasing node order.
    #[must_use]
    pub const fn exercise_nodes(&self) -> &[NodeId] {
        &self.exercise_nodes
    }

    /// Returns the exact terminal expiry node.
    #[must_use]
    pub const fn expiry(&self) -> NodeId {
        self.expiry
    }

    /// Returns the explicitly ordered phases attached to a node.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::NodeOutOfBounds`] when the node is outside this schedule's grid.
    pub fn phases(&self, node: NodeId) -> Result<NodePhases<'_>, InputError> {
        self.grid.time(node)?;
        let exercise = self.exercise_nodes.binary_search(&node).is_ok();
        let dividend = self
            .dividends
            .binary_search_by_key(&node, |candidate| candidate.node())
            .ok()
            .map(|index| &self.dividends[index]);
        Ok(NodePhases {
            exercise,
            dividend,
            expiry: node == self.expiry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> TimeGrid {
        TimeGrid::try_new([0.0, 0.25, 0.5, 1.0]).unwrap()
    }

    #[test]
    fn preserves_explicit_pre_dividend_exercise_order() {
        let grid = grid();
        let dividend_node = grid.node_id(2).unwrap();
        let expiry = grid.node_id(3).unwrap();
        let dividend = CashDividend::try_new(dividend_node, 2.0).unwrap();
        let schedule = EventSchedule::try_new(grid, [dividend], [dividend_node], expiry).unwrap();

        let phases = schedule.phases(dividend_node).unwrap();
        assert!(phases.exercise);
        assert_eq!(phases.dividend.copied(), Some(dividend));
        assert!(!phases.expiry);
        assert_eq!(
            schedule.phases(expiry).unwrap(),
            NodePhases {
                exercise: false,
                dividend: None,
                expiry: true,
            }
        );
    }

    #[test]
    fn requires_strictly_increasing_event_nodes() {
        let grid = grid();
        let one = grid.node_id(1).unwrap();
        let two = grid.node_id(2).unwrap();
        let expiry = grid.node_id(3).unwrap();
        let first = CashDividend::try_new(one, 1.0).unwrap();
        let second = CashDividend::try_new(two, 2.0).unwrap();
        let schedule =
            EventSchedule::try_new(grid.clone(), [first, second], [one, two], expiry).unwrap();
        assert_eq!(schedule.dividends(), &[first, second]);
        assert_eq!(schedule.exercise_nodes(), &[one, two]);

        assert_eq!(
            EventSchedule::try_new(grid.clone(), [second, first], [], expiry).unwrap_err(),
            InputError::UnorderedDividendNode {
                previous: two.get(),
                node: one.get(),
            }
        );
        assert_eq!(
            EventSchedule::try_new(grid.clone(), [], [two, one], expiry).unwrap_err(),
            InputError::UnorderedExerciseNode {
                previous: two.get(),
                node: one.get(),
            }
        );

        assert_eq!(
            EventSchedule::try_new(grid.clone(), [first, first], [], expiry).unwrap_err(),
            InputError::DuplicateDividendNode { node: one.get() }
        );
        assert_eq!(
            EventSchedule::try_new(grid, [], [two, two], expiry).unwrap_err(),
            InputError::DuplicateExerciseNode { node: two.get() }
        );
    }

    #[test]
    fn rejects_events_at_unsupported_boundaries() {
        let grid = grid();
        let origin = grid.node_id(0).unwrap();
        let expiry = grid.node_id(3).unwrap();
        let origin_dividend = CashDividend::try_new(origin, 1.0).unwrap();
        assert!(matches!(
            EventSchedule::try_new(grid.clone(), [origin_dividend], [], expiry),
            Err(InputError::InvalidDividendNode { node: 0, .. })
        ));
        assert!(matches!(
            EventSchedule::try_new(grid.clone(), [], [expiry], expiry),
            Err(InputError::InvalidExerciseNode { .. })
        ));
        assert!(matches!(
            EventSchedule::try_new(grid, [], [], origin),
            Err(InputError::InvalidExpiryNode { .. })
        ));
    }
}
