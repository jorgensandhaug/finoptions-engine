//! Event-aligned simulation primitives.

/// Marker returned while the first production simulation contract is being implemented.
///
/// This crate intentionally exposes no pricing behavior until its financial specification and
/// differential tests are present.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationCrate;
