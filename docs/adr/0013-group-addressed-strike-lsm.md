# ADR 0013: Group-addressed strike LSM with shared path populations

- Status: accepted for financially identical constant-parameter call states
- Date: 2026-07-14

## Context

Listed-option research tables commonly contain several strikes for the same underlying, quote
date, and expiry. Once spot, maturity, discount curve, volatility, and dividend dynamics are fixed,
the simulated risk-neutral state process does not depend on strike. Simulating that process again
for every strike wastes the dominant path-generation work and gives a memory-constrained research
machine less useful iteration throughput.

Reuse is valid only across contracts with exactly the same simulated state. Strike changes the
payoff, in-the-money fitting population, continuation regression, stopping rule, and evaluation
payoffs. Those strike-specific objects cannot be shared. A compact API must make this distinction
explicit and must not infer approximate equality from tabular data.

## Decision

Add a grouped call boundary with contiguous uint32 strike-row offsets. Spot, maturity, constant
continuously compounded rate, constant volatility, and the complete dividend representation are
stored once per group; strike remains row-aligned. Empty groups, nonzero initial offsets, reversed
boundaries, a terminal offset different from the strike count, and every compact-axis mismatch fail
before simulation.

Each group receives one training random domain and one disjoint evaluation random domain. All
strikes in that group intentionally use these common random numbers. Every strike independently
fits its declared LSM stopping policy on the shared training paths and independently evaluates that
policy on the shared evaluation paths. Cross-strike Monte Carlo errors are therefore correlated,
but each contract retains its own valid out-of-sample estimator and diagnostics.

Training paths are simulated once, all strike policies are fitted, and the training matrix is
released before the evaluation matrix is allocated. Evaluation paths are then simulated once and
used by each fitted policy. Before parallel dispatch, the row output is safely partitioned into one
disjoint mutable slice per group. Rayon's indexed scheduler operates on those slice descriptors;
no unsafe aliasing or bespoke leaf-one task tree is used. Row order is preserved, the lowest failed
group and strike wins deterministic serial error replay, partition-allocation failure is typed, and
no partial result escapes.

The Python type `LsmStrikeGroups` accepts only plain, one-dimensional, C-contiguous NumPy arrays of
the exact declared dtype. It does not sort, copy, collapse, or validate equality of source rows.
Polars remains outside the engine: a research adapter must prove exact constancy of every simulated
state field before materializing compact group columns. Pricing is a batch terminal operation, not
an opaque row-wise lazy expression.

## Evidence

- A one-contract-per-group request matches the existing row-addressed implementation bit for bit
  for continuous yields, exact cash-dividend jumps, and escrowed cash dividends.
- Every distinct strike in a shared group matches an independent one-row run on that group's exact
  training and evaluation population in all eleven result and diagnostic fields.
- Duplicate strikes in one group produce bitwise-identical prices, sampling statistics, regression
  diagnostics, and stopping counts; different strikes retain different results.
- Serial and parallel grouped execution are bitwise identical for all three dividend
  representations.
- The indexed scheduler completes a 74,215-group boundary matching the reconstructed project
  cohort's exact group count, and a separate test proves that parallel failures replay the same
  lowest contract row as serial execution.
- Rust and Python tests reject malformed offsets and compact axes without repair or fallback.
- The project-shaped Apple Silicon measurements are recorded in
  [`../performance/2026-07-14-grouped-strike-lsm.md`](../performance/2026-07-14-grouped-strike-lsm.md).

## Consequences

- Random-domain identity is group-addressed, not strike-row-addressed. Multi-strike grouped output
  need not equal a legacy row batch that intentionally used different random paths for each strike.
- Common random numbers improve the comparability of cross-strike estimates but do not make LSM
  exact or remove regression-support requirements.
- Callers own group construction. Approximate floats, missing state fields, or silently selected
  representatives are outside this contract and must never be passed as exact groups.
- The engine stays independent of Arrow and Polars while remaining compatible with compact buffers
  materialized from a verified lazy plan.
