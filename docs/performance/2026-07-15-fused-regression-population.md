# Fused regression-population checkpoint: 2026-07-15

This checkpoint accumulates regression validation and Welford predictor location/scale while the
LSM policy performs its unavoidable in-the-money population selection. The private reusable
population owns its predictor and response buffers together with their summary, so statistics
cannot be paired with different slices. The regression, population membership, weighting,
standardization, orthogonal basis, condition diagnostic, coefficients, and stopping rule are
unchanged.

## Profile and rejected candidates

A 20-second macOS `sample` profile of engine revision
`4a1f3aea41d0241cf0bc4dc696f579bd89a0232b` used the grouped 128-state, 896-contract workload with
150 intervals, 500 training paths, 500 evaluation paths, Apple math, and eight workers. Fitting was
the dominant engine leaf. Disassembly showed that every positive in-the-money population was first
collected, then scanned twice for predictor/response finiteness, once for Welford location/scale,
and once for standardization.

Three narrower candidates were measured and completely removed:

- multiplying discounted values in one pass and checking finiteness in a second made the full
  one-thread Apple workload 0.83%--1.69% slower, with a 1.24% mean regression;
- invoking out-of-place `vDSP_vsmulD` for every 500-path discount slice made the workload
  109%--236% slower, with a 166% mean regression, because roughly 134,000 small kernel calls could
  not amortize dispatch cost;
- replacing path-level moneyness division with reciprocal multiplication produced no detectable
  change (−0.99% to +1.82%, `p = 0.55`) and would have complicated the extreme valid-strike domain.

A later target-free probe also rejected fusing location, scale, and standardization through
Apple's `vDSP_normalizeD`. Although the primitive returned the expected population standard
deviation for `[0, 2, 4, 6]`, it returned zero scale and NaN standardized outputs for 500 distinct
representable values `100 + i * 1e-10`. A high-precision calculation over those exact binary64
inputs gives population standard deviation `1.4433727911925833e-8`. It likewise returned zero for
32,768 values `1e16 + 2i`, whose exact-input population standard deviation is
`18918.613611995992`. That cancellation behavior would misclassify valid near-degenerate
regressions, so no normalization wrapper or runtime branch was added.

No API, FFI wrapper, alternate arithmetic route, or dead benchmark from those candidates remains.

## Accepted design

The LSM selector now pushes each accepted `(moneyness, discounted continuation)` pair into one
private `RegressionPopulation`. The push records the first non-finite predictor and response and,
for positive degree, performs the identical ordered Welford update. The population is cleared and
reused at the next exercise node without reallocating. Its immutable estimator specification and
owned buffers make the summarized fit internally self-consistent by construction.

Dimension and minimum-population checks still occur before non-finite and degenerate-scale errors.
The independent benchmark/test entry point retains complete slice validation and the original
location/scale pass. Production fitting therefore removes redundant scans only when the values
were already observed by the owned population; it never trusts arbitrary caller slices.

## Results

Against the untouched `4a1f3ae` baseline, the final type-safe implementation measured
`1.6537--1.6761 s` for the complete one-thread Apple grouped workload, compared with the saved
baseline's `1.7825--1.8073 s`. Criterion reports a 6.35%--8.13% elapsed-time reduction with a 7.23%
mean reduction (`p < 0.05`). Two earlier forms of the same fusion reproduced 6.91%--9.98% and
8.90%--10.66% reductions before the type-safe ownership refactor.

The retained isolated 500-row benchmark measures the summarized pass separately rather than fused
with selection. Scalar full-validation and summarized means were `9.8149 us` and `9.5404 us`;
Apple means were `7.1540 us` and `7.2771 us`. The latter correctly shows that a standalone summary
pass is not an Apple optimization: the production benefit comes from merging it into the already
required ITM selection pass. A scalar end-to-end A/B was invalidated after foreground load widened
the old estimate to `2.25--3.90 s` and made Criterion forecast an 87-second candidate sample; those
numbers are not evidence.

### Indexed stopping-policy application

A later profile showed that the training stopping rule rescanned every simulated path, repeated
the in-the-money comparison, and divided spot by strike even though population selection had
already produced exactly the ordered in-the-money predictors. The retained extension records each
accepted path index beside its predictor. After fitting, policy application traverses that same
population and updates the indexed value slots. Population selection, predictor values, fitted
model, prediction order, exercise tie rule, and path values are otherwise unchanged. A
degree-zero-through-four differential test proves bitwise-equal resulting value vectors against
the original full-path scan.

Three prebuilt binaries isolated the accepted `d7f8a6b` baseline, the indexed candidate, and an
indexed candidate that also replaced evaluation Welford moments. Their SHA-256 digests were
`11f4021308f1d04d8025abd906a7a06238e234a959d068990bc15c2e02bf7fdd`,
`1a1e39bfa0f5c9ea146a5bb272ba8ccb6447442189f26b91fd1c3e7a239f8e66`, and
`966e27b5f417b5c3ba6ffd4a06651733cc7bdf2e3d124ea88f875cc544c88a78`, respectively. The complete
one-thread Apple workload ran in the order old, indexed, indexed-plus-moments,
indexed-plus-moments, indexed, old. Center estimates were `1.4512`, `1.3823`, `1.3864`, `1.3876`,
`1.3799`, and `1.4797` seconds. The indexed candidate reduced the paired centers by 4.75% and
6.75%, or 5.76% across the two center means.

The alternate moment reducer was removed. Its one-thread centers were 0.30% and 0.56% slower than
the adjacent indexed candidate. Eight-thread runs had material scheduler variance: repeated
centers for the same binaries moved between roughly `237` and `283` ms and reversed the apparent
moment-kernel ordering. Those measurements cannot support a production change. The original
single-pass Welford evaluation remains; no alternate moment route or test-only production branch
is retained.

## Correctness gates

- For degrees zero through four, every fitted model field from the summarized population is
  bitwise identical to the fully validating route on both scalar and Apple backends.
- Length/minimum-population ordering remains at the boundary; insufficient population,
  non-finite predictor, non-finite response, and degenerate scale retain the same error precedence
  and payload.
- Reuse after a population containing NaNs is tested to clear values, statistics, and validation
  state before the next fit.
- Existing independent SVD, affine-rescaling, rank, condition, grouped/row, dividend, random-domain,
  thread-invariance, and financial-reference tests continue to exercise the production fit.
- There is no fallback, imputation, degree change, regularization, row removal, or relaxed finite
  check.

## Reproduce

```shell
cargo bench -p finoptions-lsm --features bench-internals \
  --bench polynomial_regression -- \
  'polynomial_regression_degree_4/(empirical_orthogonal|summarized_population)_(rust_std_f64|apple_accelerate)/500'
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_apple_accelerate_1t/128'
```
