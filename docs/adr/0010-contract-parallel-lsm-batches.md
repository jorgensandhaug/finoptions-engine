# ADR 0010: Contract-parallel independent-policy LSM batches

- Status: accepted for constant-parameter Bermudan calls
- Date: 2026-07-14

## Context

Research workloads price many option rows with a shared estimator specification. Repeated Python
calls add boundary overhead, repeatedly enter a path-parallel runtime, and make row-level random
identity and failure ordering harder to audit. A batch interface must improve throughput without
changing any contract, dividend event, exercise opportunity, regression, path population, or
random stream.

The batch boundary must also preserve the distinction between exact equivalence and independent
numerical validation. Scalar and batch executions of the same declared algorithm may be required
to agree bit for bit. A different mathematically valid method is evidence through bounds,
convergence, and justified tolerances; it is not an exact numerical oracle.

## Decision

Expose one borrowed, columnar call-pricing request with equal-length `f64` vectors for initial spot,
strike, maturity, constant continuously compounded risk-free rate, and constant volatility. One
request selects exactly one dividend representation:

- a row-aligned constant nonnegative proportional-dividend yield; or
- row-aligned ragged deterministic cash schedules under one explicit limited-liability jump policy;
  or
- the separately named escrowed prepaid-forward approximation from
  [ADR 0011](0011-explicit-escrowed-cash-dividends.md), with an explicit volatility convention.

The exercise identity is a uniform Bermudan base mesh. Every cash-event year fraction is inserted
exactly into its contract's mesh, exercise observes the pre-dividend state, and expiry remains the
separate terminal node. Exercise at valuation origin is explicitly included or excluded. Events at
or after maturity, duplicate event times, invalid scalar columns, and unrepresentable meshes fail;
the implementation never rounds, shifts, aggregates, or drops an event.

Each contract row receives `training_domain_start + row` and
`evaluation_domain_start + row`. Both complete half-open domain intervals are validated before
result allocation; overflow or overlap fails. Policy fitting and valuation use separate path
populations. One explicit CPU executor parallelizes contracts, while each contract uses serial
path-local simulation and policy kernels. This avoids nested parallel execution and bounds
per-worker scratch storage independently of total batch rows.

Parallel execution preserves row order. If any row fails, workers record the lowest failed row and
that row is replayed serially to return its complete typed error. No partially valid result escapes.
Successful output is a compact set of immutable, row-aligned NumPy columns containing price,
sampling uncertainty, aggregated regression-support diagnostics, and the complete early/expiry
stopping partition.

The initial Python boundary accepts only plain one-dimensional C-contiguous NumPy arrays with the
exact declared dtype. It performs no implicit cast or copy. The synchronous call borrows these
arrays, so callers must not mutate them concurrently. A future Polars adapter may borrow compatible
physical buffers, but it must preserve these exact validation and ownership semantics rather than
embedding pricing as an opaque row-wise expression.

## Evidence

- One-row continuous-yield and off-grid cash-dividend batches match independently assembled scalar
  plans bit for bit in price, sampling statistics, aggregated diagnostics, and stopping counts.
- Serial and parallel batch execution are bitwise identical for both continuous-yield and exact
  cash-dividend contracts.
- Tests cover row-addressed domains, interval overflow/collision, deterministic lowest-row failure,
  one-ULP cash-event insertion, duplicate and expiry-time events, stopping partitions, immutable
  outputs, and native revalidation after Python construction.
- The project-shaped performance checkpoint is recorded in
  [`../performance/2026-07-14-lsm-batch.md`](../performance/2026-07-14-lsm-batch.md).

## Consequences

- Contract batches are the primary CPU throughput unit; scalar pricing remains the audit and
  bespoke-model interface.
- Worker count is an execution resource, not part of the financial or stochastic result identity.
- Exact cash dividends, escrowed cash dividends, and continuous proportional yields remain
  different request variants and cannot be mixed or silently converted.
- Failure is atomic at the request boundary. Continuing past invalid rows requires an explicit
  higher-level partitioning policy outside the engine.
- The accepted slice does not yet include time-varying rate/volatility columns, puts, stochastic
  dividends, heterogeneous estimators within one request, GPU execution, or a Polars lazy plugin.
