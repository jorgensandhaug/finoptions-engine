# Hoisted policy-prediction dispatch checkpoint: 2026-07-15

This checkpoint specializes the already validated degree-zero-to-four polynomial recurrence once
per exercise rule instead of branching on degree for every in-the-money path. It does not change
the regression, fitted coefficients, predictor normalization, exercise population, tie rule, or
payoff. Every prediction retains the same finite-input and finite-output checks, and unsupported
degrees remain impossible through the validated model constructor rather than selecting another
method.

## Profile and rejected candidates

A 20-second macOS `sample` profile of engine revision
`f81d93d14cc2de01a61014b396175bd8e610669a` used the complete Apple grouped benchmark at 128
states, 896 contracts, 150 intervals, 500 training paths, 500 evaluation paths, and eight threads.
After the vDSP and fixed-size Jacobi work, policy fitting remained the dominant top-of-stack
symbol; fixed-policy evaluation was the next largest engine leaf, followed by Philox generation,
vDSP reductions, path stepping, and vector exponential evaluation. That made recurrence dispatch a
measured target shared by fitting and evaluation.

Two candidates were rejected and removed from production code:

- dispatching to a fixed-degree recurrence inside every scalar prediction made the isolated
  500-prediction kernel about 24.7% slower, even with forced inlining;
- replacing the remaining scaled Jacobi `hypot` with an explicit magnitude branch produced no
  complete-fit speedup (scalar slightly slower, Apple statistically unchanged).

The accepted design hoists the degree match outside each path loop. A retained benchmark-only
fitted-model harness prevents the rejected per-prediction dispatch from being rediscovered.

## Results

On the same Apple M4 Pro environment as the preceding checkpoints, the adjacent isolated
500-prediction comparison measured 1.2821 us for runtime-degree dispatch and 1.0790 us after the
degree-four dispatch was hoisted, a 15.8% reduction in that kernel.

Sustained eight-thread measurements ran alongside material foreground CPU load while both backend
baselines fell by 18%--35%; they were invalidated and are deliberately not used as evidence. An
adjacent one-thread Apple grouped A/B remained stable: the hoisted implementation measured 1.6764 s
and the restored runtime-degree implementation measured 1.7383 s. Criterion found the old
implementation 3.32%--4.11% slower with a 3.70% mean difference; restoring the candidate reproduced
its baseline with no statistically detectable change.

## Correctness gates

- For every supported degree, the specialized recurrence is bitwise identical to the checked
  runtime-degree recurrence across the representative SVD-oracle population.
- Both routes retain predictor and result finiteness checks and the identical multiply-add order.
- The policy dispatch occurs once per fitted exercise rule and has no fallback arm; the unreachable
  arm represents the private validated-model invariant.
- Scalar, grouped, parallel, dividend-model, random-domain, financial-bound, and independent SVD
  tests continue to exercise the production path.

## Reproduce

```shell
cargo bench -p finoptions-lsm --features bench-internals \
  --bench polynomial_regression -- \
  'polynomial_prediction_degree_4/(runtime_degree|hoisted_degree_four)_recurrence_500'
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_apple_accelerate_1t/128'
```
