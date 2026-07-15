# ADR 0005: Checked low-degree polynomial regression

- Status: accepted for degrees zero through four
- Date: 2026-07-13

## Context

The first target estimator uses ordinary least squares on in-the-money paths with a fourth-degree
continuation-value polynomial. Repeated regressions are on the hot path, often with only 500
observations, so a general dense solver pays substantial dispatch, allocation, and factorization
overhead. Normal equations are not acceptable: they square the design condition and make a
plausible result too easy to return from a numerically unsupported population.

Agreement with another implementation is not a definition of financial or numerical correctness.
Independent pivoted-QR/SVD calculations are validation oracles for the declared polynomial
projection, numerical rank, and condition diagnostics; their internal coefficients and last bits
are not replication targets.

## Decision

Support unregularized, equally weighted polynomial OLS for exactly degrees zero through four. A
positive-degree fit standardizes the predictor with its sample population mean and standard
deviation, then constructs monic empirical orthogonal polynomials with the three-term recurrence.
Projection onto that basis spans the same declared polynomial space while requiring `O(n d)` work,
four reusable `f64` vectors, and fixed-size model storage.

The solver reconstructs the triangular transformation from standardized monomials to the
orthonormal empirical basis. A one-sided Jacobi SVD of that at-most `5 x 5` transformation computes
the two-norm condition without forming a Gram matrix. The transformation and standardized
Vandermonde design have reciprocal singular values and therefore the same condition number. The
relative rank tolerance and condition limit are explicit estimator inputs. A condition limit may
not exceed the reciprocal rank tolerance. The validated one-to-five-column dimension dispatches to
fixed-size Jacobi implementations so the compiler can specialize the tiny matrix loops; unsupported
dimensions remain unreachable through the validated public configuration rather than selecting a
different solver.

Fitted-policy execution similarly dispatches once per exercise rule to the validated degree-zero-
through-four recurrence. The path loop keeps the same standardization, multiply-add sequence, and
finite-value checks without repeating runtime degree control for every prediction. This is an
execution specialization of the fitted model, not a different polynomial estimator.

All predictor and response values are validated before workspace reuse. Unsupported degree,
insufficient observations, non-finite input/arithmetic, degenerate predictor scale, numerical rank
loss, excessive condition, and Jacobi non-convergence are typed failures. There is no automatic
degree reduction, ridge term, pseudoinverse, population expansion, tolerance change, or forced
continuation rule.

The LSM policy's reusable in-the-money population owns its predictor and response buffers together
with the first-invalid-value state and ordered Welford location/scale summary. Accumulating that
summary during the already required population-selection pass avoids rescanning private owned
values. Arbitrary-slice test and benchmark entry points retain full validation. Tests require the
owned-population and fully validating routes to produce bitwise-identical models and the same error
precedence; the optimization removes memory passes, not checks or estimator work.

## Evidence

Tests compare accepted projections and condition numbers with nalgebra 0.35.0 SVD calculations for
every supported degree on a representative 500-row population and for degree four on 65,536 rows,
verify affine state rescaling and exact polynomial recovery, and compare adversarial rank/condition
classifications. The oracle is a development-only dependency; the production LSM dependency graph
remains limited to engine crates and `thiserror`.

On an Apple M4 Pro, Criterion 0.8.2 measured the complete allocation-reusing degree-four kernel,
including validation and condition diagnostics, at about `9.54 us` for 500 rows and `1.05 ms` for
65,536 rows. The independent SVD benchmark measured about `16.11 us` and `2.40 ms`, respectively.
These are local engineering measurements, not publication claims or permanent regression limits.

At the 150-interval, 500-training/500-evaluation grouped paper workload, fusing population
selection with validation and ordered Welford statistics reduced a stable one-thread Apple run by
6.35%--8.13% after the type-safe ownership refactor. See the
[fused-population checkpoint](../performance/2026-07-15-fused-regression-population.md) for the
profile, rejected candidates, exact A/B scope, and invalidated scalar timing.

## Consequences

- The paper's degree-four estimator has a small, auditable, fail-fast kernel.
- A fitted model stores its sample-derived recurrence and can evaluate the same fixed policy on an
  independent path population without refitting.
- Higher degree, multiple state variables, weights, or regularization require a new named estimator
  and a separate evidence gate.
- LSM correctness still requires explicit payoff, discounting, exercise ordering, independent
  fitting/evaluation, convergence studies, and financial reference tests; this ADR establishes only
  the regression component.

## Sources

- [NIST DLMF: general orthogonal polynomials and three-term recurrences](https://dlmf.nist.gov/18.2)
- [LAPACK `DGELSD`: relative singular-value rank and condition semantics](https://www.netlib.org/lapack/explore-html/d9/d67/group__gelsd_ga0bee7e1b9e7e43f59ecf2419b2759c42.html)
- [Demmel et al.: SVD with high relative accuracy](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1997/5373.html)
- [Longstaff and Schwartz: least-squares Monte Carlo](https://doi.org/10.1093/rfs/14.1.113)
