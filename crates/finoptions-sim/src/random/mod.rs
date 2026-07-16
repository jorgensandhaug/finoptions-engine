//! Explicit, reproducible random-stream identity and private generation primitives.

mod normal;
mod normal_tables;
mod philox;

pub(crate) use normal::{NormalError, sample_standard_normal};
pub(crate) use philox::PhiloxPathStream;

/// Reproducible random namespace for one simulation role.
///
/// `seed` selects the Philox key. `domain` occupies a distinct counter lane and must differ for
/// path sets that must be statistically independent, such as LSM policy fitting and evaluation.
/// Path identity occupies another counter lane, so worker count and scheduling cannot change a
/// path's random stream.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RandomStream {
    seed: u64,
    domain: u32,
}

impl RandomStream {
    /// Creates an explicit `(seed, domain)` random namespace without inferring either value.
    #[must_use]
    pub const fn new(seed: u64, domain: u32) -> Self {
        Self { seed, domain }
    }

    /// Returns the exact 64-bit Philox key seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Returns the exact 32-bit counter-domain identifier.
    #[must_use]
    pub const fn domain(self) -> u32 {
        self.domain
    }
}
