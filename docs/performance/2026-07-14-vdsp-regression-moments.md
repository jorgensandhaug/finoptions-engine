# Apple vDSP regression-reduction checkpoint: 2026-07-14

This checkpoint extends the explicit macOS backend from vForce path exponentials to the dominant
continuation-regression reductions. It does not change the financial model or estimator: both
backends use the same standardized degree-zero-to-four polynomial space, unregularized empirical
orthogonal least squares, in-the-money training population, rank threshold, condition limit, and
fixed-policy out-of-sample evaluation. Reduction order is an explicit numerical implementation
choice and exact cross-backend bits are not promised.

## Profile and decision

A release `sample` profile of the verified grouped project workload attributed about 77% of direct
pricing samples to policy fitting, 11% to policy evaluation, and 12% to the two path simulations.
Within fitting, regression construction dominated; design diagnosis and prediction were the next
largest components. This made the three reductions for every orthogonal basis column the first
measured target.

Rejected candidates were removed rather than retained as dormant complexity:

- reusing additional regression/value scratch vectors had no statistically detectable grouped
  benefit;
- folding the finite check into the recurrence write had no measurable benefit;
- fusing basis generation with scalar moments improved an isolated loop but not the grouped run;
- three vDSP calls for the basis recurrence took roughly 306 ns at 500 rows versus 102 ns scalar.

The accepted kernel uses one `vDSP_vsqD` and three `vDSP_dotprD` calls. Backend dispatch remains
outside vendor FFI, malformed operand shapes fail before scratch mutation, unavailable platforms
fail during executor construction, and no scalar retry exists.

## Environment

- Apple M4 Pro, 14 physical/logical CPU cores, 48 GiB unified memory
- macOS 26.2 build 25C5048a, arm64
- Rust 1.97.0, Criterion 0.8.2, release benchmark profile
- no other intentional research kernel running

## Results

| Boundary | Portable scalar mean | Apple Accelerate mean | Apple elapsed reduction |
|---|---:|---:|---:|
| Three moments, 500 rows | 591.84 ns | 240.88 ns | 59.3% (2.46× throughput) |
| Complete degree-four fit, 500 rows | 9.6727 µs | 7.4909 µs | 22.6% (1.29× throughput) |
| Grouped LSM, 128 states / 896 contracts / 8 threads | 343.32 ms | 285.31 ms | 16.9% (1.20× throughput) |

The grouped comparison measures the complete named backends, so its Apple advantage includes both
vForce path exponentials and vDSP regression reductions. In the earlier paired candidate A/B run
with Apple path simulation held fixed, adding vDSP moments reduced grouped median time by about
6.2%; its bootstrap interval excluded zero. The final table is the authoritative reproducible
checkpoint for the integrated implementation.

## Correctness gates

- The unsafe wrapper is isolated under the crate safety contract and checks all four operand
  lengths before entering vDSP.
- A 65,536-row deterministic population compares each vDSP reduction with a compensated sum under
  a fixed forward-error bound and checks the squared-basis workspace.
- The complete Apple degree-four regression agrees with an independent SVD oracle on predictions,
  rank, and condition number at 500 and 65,536 observations.
- Seeded Apple simulation is bitwise invariant across worker counts and remains within its declared
  vForce tolerance of the scalar transition.
- Apple batch LSM is bitwise invariant across worker counts, satisfies call bounds, and preserves
  the complete fitted-policy diagnostics. No test requires independent backends to match bits.

## Reproduce

```shell
cargo bench -p finoptions-accelerate --bench polynomial_moments
cargo bench -p finoptions-lsm --features bench-internals \
  --bench polynomial_regression -- \
  'polynomial_regression_degree_4/empirical_orthogonal_.*500'
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_(rust_std_f64|apple_accelerate)_8t/128'
```

Primary ABI and operation references:
[vDSP dot product](https://developer.apple.com/documentation/accelerate/vdsp_dotprd),
[vDSP vector square](https://developer.apple.com/documentation/accelerate/vdsp_vsqd), and
[vDSP vector arithmetic](https://developer.apple.com/documentation/accelerate/using-vdsp-for-vector-based-arithmetic).
