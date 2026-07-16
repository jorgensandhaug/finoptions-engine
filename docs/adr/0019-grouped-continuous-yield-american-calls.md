# ADR 0019: Grouped continuous-yield American calls

- Status: accepted
- Date: 2026-07-15

## Context

Continuous-yield project variants do not need Monte Carlo paths. Under constant Black–Scholes
parameters, American call-put symmetry gives
`C_A(S,K,r,q) = P_A(K,S,q,r)`. Normalizing the transformed put strike to one makes its exercise
boundary independent of the original call strike, so every strike sharing one market state can use
one boundary solve.

The implementation must cover the recovered study's nonnegative call yields and both positive and
negative call rates. It must not route a failing group to LSM, refine a configuration implicitly, or
clip an interpolated boundary. The published Andersen–Lake–Offengenden algorithms are mathematical
references, not an obligation to reproduce a particular accelerator or library implementation.

## Decision

Add a dependency-isolated `finoptions-american` crate and a small typed Python facade. One grouped
request borrows Arrow-style `i64` offsets, group-aligned spot/maturity/rate/yield/volatility vectors,
and row-aligned strikes without copying.

For every admitted group:

1. If the original call has `q = 0` and `r >= 0`, use the exact no-early-exercise theorem and the
   European Black–Scholes call formula.
2. Otherwise apply call-put symmetry, normalize the transformed put strike to one, and solve
   fixed-point system B at a caller-declared number of Chebyshev collocation nodes.
3. Perform exactly the declared number of ordinary fixed-point iterations. The published partial
   Jacobi–Newton first step is deliberately not used: it left the admissible boundary domain on a
   recovered group, while the ordinary map remained stable. There is no runtime choice between
   them.
4. Reconstruct `H = ln(B/B(0+))²` with a local shape-preserving PCHIP cubic. Endpoint values of `H`
   are nonnegative, and each interval remains inside its endpoint range even when the full boundary
   is nonmonotone. No negative-value clipping is present.
5. Use checked-in Gauss–Legendre rules for both boundary and price integrals. The generator pins
   Python, NumPy, and SciPy versions; runtime rule generation is absent.
6. Use the exact stopping payoff inside the solved origin exercise region. If finite-precision
   price quadrature produces a continuation-side value below the Snell-envelope obstacle, restore
   the obstacle only when the required correction divided by original call spot is within a
   separately declared limit. Report the origin decision per contract and maximum correction per
   group. A larger correction is a typed error. Setting the limit to zero disables projection.

Every numerical count and both convergence limits are required constructor arguments. Negative
call yields fail before pricing because they do not belong to this implementation's proved
single-lower-bound transformed-put domain. A boundary-domain violation, unsupported quadrature
order, excessive fixed-point residual, excessive obstacle projection, invalid financial bound, or
allocation failure stops the complete request. Parallel execution deterministically returns the
lowest failing group and is bitwise equal to serial execution.

The implementation follows the integral price and fixed-point-B equations in Andersen, Lake, and
Offengenden, but its ordinary iteration and PCHIP reconstruction are owned choices. Healy's
negative-rate analysis supplies the theorem boundary for call early exercise. The interpolation
formula follows the local shape-preserving construction of Fritsch and Butland.

- Andersen, Lake, and Offengenden: <https://doi.org/10.21314/JCF.2016.312>
- Healy: <https://arxiv.org/abs/2109.15157>
- Fritsch and Butland: <https://doi.org/10.1137/0905021>

## Evidence

Independent gates cover exact zero-yield theorem reduction, published three-year put values through
symmetry, dense 8,192-step CRR comparisons including a negative call rate, Gauss–Legendre moments,
nonmonotone PCHIP interval bounds, exact origin stopping payoff, bounded obstacle projection,
malformed domains, and bitwise serial/parallel equality.

The independent-tree cases also include rounded synthetic envelopes for the study's unusual but
valid short-maturity/high-yield and high-volatility regimes: annualized yield `3.10` at three days,
and volatility `0.84` at yield `0.105`. The original cases retain a `$0.0003` absolute gate; the
high-volatility envelope uses `$0.0005` because the finite 8,192-step tree itself differs by about
`$0.00039`. No licensed row or option-price target is embedded in the test.

The target-free recovered-cohort pilot used 96,666 natural groups and 678,122 contracts. The paper
profile `(boundary order=13, iterations=16, collocation nodes=6, price order=21, boundary
residual=1e-4, obstacle projection=1e-6)` completed in 0.563 seconds for forecast yields and 0.891
seconds for historical yields on eight M4 Pro threads. Maximum observed residuals were `1.30e-5`
and `1.50e-6`; maximum obstacle projections were `2.62e-7` and `3.02e-7` of spot.

Against an independently declared over-resolved profile `(81,48,32,131,1e-8,1e-8)`, maximum price
differences were $0.000401 forecast and $0.000400 historical; RMSE was about $0.000035 for both, and
no contract differed by $0.001. Option targets were not read when choosing the profile. Full
evidence is recorded in
[`../performance/2026-07-15-grouped-continuous-yield-american.md`](../performance/2026-07-15-grouped-continuous-yield-american.md).

## Consequences

- Continuous-yield publication variants become deterministic and eliminate simulation error.
- The engine remains framework-neutral; Polars owns lazy preparation and hands contiguous arrays to
  the explicit terminal operation.
- One boundary per natural state makes multi-strike pricing the intended fast path.
- The paper owns its selected profile. The engine supplies no hidden `fast`, `accurate`, or host-
  selected default.
- Discrete cash-dividend variants remain separate methods; this decision does not reinterpret a cash
  schedule as a yield.
