# ADR 0020: Additive-panel stationary bootstrap

- Status: accepted
- Date: 2026-07-16

## Context

Financial panels often contain many dependent rows per date. A row-level bootstrap is both the
wrong dependence model and unnecessarily expensive when the statistic is a transformation of
additive sufficient statistics. Reproducible research also cannot leave the random generator,
counter assignment, discrete-uniform mapping, geometric inverse CDF, or execution scheduling
implicit.

## Decision

`finoptions-sim` exposes a generic circular stationary bootstrap over a finite, ordered,
row-major matrix of additive panel statistics. One bootstrap replication always contains exactly
the observed number of panel positions. A checked prefix matrix evaluates a nonwrapping or
once-wrapping block in `O(statistics)` work, so expected work is
`O(replications * panels / mean_block_length * statistics)` rather than revisiting every selected
panel row.

The request owns an explicit Philox4x32-10 `(seed, domain)` namespace and a nonwrapping starting
path lane. Replication `b` occupies path lane `start + b`; its sequential block counter occupies
the first two Philox counter words. Circular origins use Lemire unsigned-64 multiply-high rejection.
Each accepted origin is followed by one length word. The top 52 bits map to the exactly
representable open-unit midpoint `(bits + 0.5) / 2^52`; the geometric inverse CDF is
`1 + floor(libm::log1p(-u) / libm::log1p(-1/L))`. `L = 1` returns one without consuming a length
word. The terminal block is truncated to the remaining sequence length.

Inputs are borrowed without conversion. Empty shapes, nonfinite values, overflowing shapes or
counter ranges, invalid mean lengths, allocation failures, and nonfinite prefix or result sums are
typed errors. A dedicated `CpuExecutor` parallelizes independent replications; path addressing
makes output bitwise invariant to worker count on one numerical build. No alternative generator,
sampler, precision, block rule, or scalar fallback is selected implicitly.

## Consequences

- Python callers can bootstrap panel-aggregated sufficient statistics without materializing row
  indices or copying the input matrix.
- Nonlinear metrics such as RMSE remain the caller's explicit transformation of resampled additive
  sums; the engine does not hard-code a paper's estimand or multiplicity procedure.
- Changing any digital random mapping is a new method identity, not an implementation detail.
- Prefix subtraction fixes a fast Float64 accumulation route; callers requiring a different
  summation identity must request and validate a distinct kernel.
