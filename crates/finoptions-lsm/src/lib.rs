//! Independently fitted and evaluated least-squares Monte Carlo.

/// Marker returned while the production regression solver remains behind its evidence gate.
///
/// No fallback or placeholder estimator is exported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LsmCrate;
