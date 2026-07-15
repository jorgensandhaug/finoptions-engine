# ADR 0012: Explicit Apple vector-math backend

- Status: accepted
- Date: 2026-07-14

## Context

Profiles of the submitted-study LSM workload first identified GBM exponential evaluation and then
continuation-regression reductions as dominant CPU consumers. Apple Accelerate exposes
double-precision vForce exponentials and vDSP vector reductions on every supported macOS deployment
target, but each has a distinct numerical contract and requires a C ABI boundary. Automatic
platform selection would make an execution manifest incomplete, while retrying through the scalar
implementation would hide a backend failure.

## Decision

`CpuExecutor` owns an explicit `CpuMathBackend`: portable `rust_std_f64` or macOS-only
`apple_accelerate`. The default is the named portable backend; Apple execution must be requested.
The Apple backend uses vForce for GBM exponentials and vDSP for the three reductions required by
each empirical orthogonal-polynomial basis column. Unavailable backends, malformed operands, and
kernel failures return typed errors and are never substituted.

The Accelerate ABI lives alone in `finoptions-accelerate`, whose
[safety contract](../../crates/finoptions-accelerate/SAFETY.md) covers buffer lifetime,
non-aliasing, length conversion, platform linking, and empty input. Seeded simulation retains the
same Philox `(seed, domain, path)` assignment and evaluates GBM transitions in measured 64-path
vector blocks. Regression retains the same standardized degree-zero-to-four polynomial space,
unregularized least-squares definition, rank threshold, condition gate, and exercise population;
only reduction order varies by backend. Contract-parallel LSM uses one serial vector kernel per
row; single-population parallel simulation distributes fixed vector blocks across the requested
Rayon pool.

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
- Adding another backend requires a distinct identifier, availability contract, differential tests,
  financial validation, and end-to-end performance evidence.
