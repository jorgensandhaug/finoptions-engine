# ADR 0006: Independently fitted and evaluated call LSM

- Status: accepted for the first call/GBM slice
- Date: 2026-07-13

## Context

Least-squares Monte Carlo constructs a stopping policy from simulated paths. Valuing that policy on
the same paths reuses training information and can bias the reported result. The supported estimator
therefore needs separately identified fitting and evaluation populations, plus explicit event
ordering, discounting, exercise support, regression diagnostics, and sampling uncertainty.

## Decision

Support a fixed-strike call with a caller-declared finite exercise schedule, exact node log discount
factors, constant-volatility risk-neutral GBM, and the dividend semantics already owned by the path
plan. Exercise observes the pre-dividend state. Terminal expiry is not another regression node.

At each positive-time exercise node, backward fitting discounts the current future-policy cashflow
to that node and fits the accepted polynomial OLS estimator only on in-the-money paths. Zero ITM
training paths is recorded as absent support. A positive population that cannot support the declared
basis fails through the regression contract. A valuation-origin exercise opportunity is one common
decision: intrinsic value is compared with the mean discounted training continuation rather than
fitting a degenerate cross-sectional regression.

The fitted policy is evaluated forward on a distinct population without refitting. An active
evaluation path entering an ITM region with no training support is an error. Exercise/continuation
ties follow an explicit `strictly greater` or `greater or equal` rule. Discounted path payoffs produce
the independent mean, sample standard deviation, standard error, and stopping counts.

Seeded fitting/evaluation must use different 32-bit Philox domains, and the exact seeded stream is
stored in each path population. Externally supplied normal matrices receive a BLAKE3 1.8.5 content
identity so the exact same matrix cannot be used for both roles. Seeded and external identities are
not mixed implicitly. There is no same-path mode, degree fallback, regularization fallback, support
expansion, or automatic exercise-schedule optimization.

## Evidence

- Serial and dedicated-pool runs are bitwise identical on the same build.
- Exact same seeded streams and exact same external path matrices are rejected.
- Missing support, underdetermined/degenerate regression, condition failure, non-finite prediction,
  simulation/valuation-curve mismatch, training/evaluation GBM-plan mismatch, observation mismatch,
  initial-state mismatch, and unrepresentable observation discount are typed errors.
- A European-only slice discounts terminal path payoffs directly, and an origin-exercise hand case
  produces one deterministic common decision rather than a regression.
- Twelve predeclared 65,536-training/65,536-evaluation replications of the published
  `S=K=100`, `r=6%`, `sigma=30%`, one `D=7` dividend at `t=0.5` liquidator case produced mean
  `11.64337996`, replication standard deviation `0.07366459`, standard error `0.02126514`, and a
  two-sided Student-t 99% interval `[11.57733456, 11.70942536]`. The independent reference
  `11.65644895` lies inside the interval. This validates the narrow policy mechanism statistically;
  it does not make LSM an exact method.

Local Criterion measurements on an Apple M4 Pro include simulation, fitting, evaluation,
allocation, validation, and diagnostics. A 150-step, 500-training/500-evaluation, degree-four case
with all 150 exercise opportunities measured about `5.44 ms` with one worker and `3.15 ms` with four.
The 32-step, 65,536/65,536 one-exercise reference case measured about `101.31 ms` and `30.11 ms`.
These are engineering measurements, not publication results or stable performance thresholds.

## Consequences

- The Python surface can expose one small `Lsm` specification and one `price_lsm_call` operation.
- Every reported value estimates the value of an out-of-sample feasible fixed policy. Its
  population value is a lower bound on the value of the declared optimal-stopping problem, while
  the finite-sample Monte Carlo estimate has sampling error and can lie on either side of either
  population value. It is not an exact American-option price.
- Exercise-grid, path-count, basis, and randomized-replication convergence remain paper-level study
  responsibilities and must be predeclared in immutable manifests.
- The engine does not infer a pre-dividend-only theorem optimization. If a study selects that
  opportunity set, its financial domain must be established outside this generic kernel.
- Batch contract scheduling, workspace pooling, and variance-reduction designs can optimize this
  estimator without changing its identity, but require their own evidence gates.

## Sources

- [Longstaff and Schwartz: least-squares Monte Carlo](https://doi.org/10.1093/rfs/14.1.113)
- [Haug, Haug, and Lewis: back to basics for discrete dividends](https://doi.org/10.1002/wilm.42820030514)
- [Merton: option pricing restrictions](https://doi.org/10.2307/3003143)
