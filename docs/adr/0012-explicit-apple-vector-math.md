# ADR 0012: Explicit Apple vector-math backend

- Status: accepted
- Date: 2026-07-14

## Context

Profiles of the submitted-study LSM workload first identified GBM exponential evaluation and then
continuation-regression reductions as dominant CPU consumers. Apple Accelerate exposes
double-precision vForce exponentials and vDSP vector reductions on every supported macOS deployment
target. Apple Silicon also guarantees the AArch64 NEON ISA, which can fuse the three reductions and
their required squared-basis scratch write into one pass. Each implementation has a distinct
numerical contract; vForce/vDSP require a C ABI boundary and NEON requires an isolated intrinsic
safety proof. Automatic platform selection would make an execution manifest incomplete, while
retrying through the scalar implementation would hide a backend failure.

## Decision

`CpuExecutor` owns an explicit `CpuMathBackend`: portable `rust_std_f64` or macOS-only
`apple_accelerate`. The default is the named portable backend; Apple execution must be requested.
The Apple backend uses vForce for GBM exponentials. On AArch64 macOS, it uses a fused NEON kernel for
the three reductions and squared-basis scratch required by each empirical orthogonal-polynomial
basis column. For every later column, a second fused NEON kernel constructs the next three-term-
recurrence basis and accumulates those same moments in one traversal. Other macOS architectures
retain vDSP for the initial moments and a checked scalar fused recurrence/moment traversal.
Architecture dispatch is fixed at compile time inside the already explicit backend; unavailable
backends, malformed operands, and kernel failures return typed errors and are never substituted.

Apple-specific unsafe vector code lives alone in `finoptions-accelerate`, whose
[safety contract](../../crates/finoptions-accelerate/SAFETY.md) covers buffer lifetime,
non-aliasing, vector-block bounds, length conversion, target-feature availability, platform
linking, and empty input. Seeded simulation retains the same Philox `(seed, domain, path)` assignment
and evaluates GBM transitions in measured 64-path vector blocks. Regression retains the same
standardized degree-zero-to-four polynomial space, unregularized least-squares definition, rank
threshold, condition gate, and exercise population; only reduction order varies by backend and
architecture. Contract-parallel LSM uses one serial vector kernel per row; single-population
parallel simulation distributes fixed vector blocks across the requested Rayon pool.

Backend identity is part of execution identity. Equivalent runs on one backend and numerical build
must remain bitwise thread-invariant. Different backends are checked against the same financial
model, path streams, bounds, and predeclared tolerances; LSM results are not required to match bits
because last-bit path differences can alter a fitted stopping decision.

## Consequences

- macOS research runs can use the measured vector path without changing financial semantics.
- Portable runs remain available and explicit on every supported platform.
- The runtime owns backend dispatch; financial algorithms never call vendor FFI directly.
- The workspace contains a very small audited unsafe boundary instead of duplicating vendor FFI in
  simulation kernels.
- The stable `apple_accelerate` identity means the explicitly selected Apple-platform vector
  backend, not a promise that every internal instruction is dispatched by the Accelerate framework;
  engine revision remains part of reproducible execution identity.
- Adding another backend requires a distinct identifier, availability contract, differential tests,
  financial validation, and end-to-end performance evidence.
