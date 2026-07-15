# Fused AArch64 polynomial-moment checkpoint: 2026-07-15

This checkpoint replaces four separate vDSP traversals of each continuation-regression basis
column on Apple Silicon with one fused AArch64 NEON traversal. It writes the same required
`basis^2` scratch and computes `sum(basis^2)`, `sum(response * basis)`, and
`sum(predictor * basis^2)`. The polynomial space, standardization, estimator, population, rank and
condition gates, coefficients, exercise rule, path streams, and financial model are unchanged.

## Design and safety

The AArch64 kernel processes eight observations through four independent two-lane accumulators,
then a two-lane tail and at most one scalar observation. The safe public boundary validates all four
slice lengths before dispatch, and Rust borrowing proves that mutable scratch cannot alias the
inputs. Every intrinsic load/store has an explicit remaining-length proof. NEON is part of the
baseline AArch64 macOS ISA; other macOS architectures retain the accepted vDSP implementation.
There is no runtime retry or scalar fallback.

The scalar differential test covers lengths `0..10`, every residue around 16 and 32, 127--129,
499--501, and 1,023--1,025. All three reductions meet the pre-existing compensated-sum forward-
error bound and every squared-basis scratch element is bitwise equal to scalar multiplication. The
65,536-row compensated oracle, independent SVD regression checks, degrees zero through four,
financial reference cases, grouped/row equivalence, and thread invariance also exercise the
production route.

## Controlled performance decision

The baseline is accepted revision `46183aa9a1a26eb13b4130d8a74816fa683de463` in a detached
worktree. Both revisions were compiled before measurement and their already-built binaries were
run sequentially in the order old, candidate, candidate, old. No benchmark ran concurrently. The
host reported no thermal or performance warning; CPU idle was sampled around 63%--72% at the end
of the sequence. An earlier measurement taken during a concurrent release-LTO Polars build and a
later one during a video render are invalid and excluded.

| Boundary | Accepted vDSP centers | Fused NEON centers | Mean-center reduction |
|---|---:|---:|---:|
| 500-row moment kernel | 234.73 ns, 241.85 ns | 106.29 ns, 101.62 ns | 56.37% |
| Complete degree-four summarized fit | 9.6116 us, 9.0288 us | 8.7740 us, 8.8498 us | 5.45% |
| 128-state/896-contract grouped LSM, one thread | 1.6140 s, 1.6503 s | 1.5431 s, 1.5688 s | 4.67% |

For the fit boundary, the highest candidate 95% interval endpoint was 8.9238 us and the lowest
baseline endpoint was 9.0135 us. For grouped LSM, the corresponding endpoints were 1.5785 s and
1.6065 s. The end-to-end improvement therefore survives outside the isolated kernel and justifies
the unsafe specialization under the repository policy.

## Backend identity

The stable public identifier remains `apple_accelerate`: it names the explicitly requested
Apple-platform vector backend, which still uses Accelerate vForce for GBM exponentials. Repository
documentation now states that continuation reductions use NEON on Apple Silicon and vDSP on other
macOS architectures. Engine revision is mandatory execution provenance, so this architecture-
specific arithmetic route is never silent.

## Reproduce

```shell
cargo bench -p finoptions-accelerate --bench polynomial_moments -- \
  'polynomial_moments_f64/(apple_production|apple_vdsp_reference)'
cargo bench -p finoptions-lsm --features bench-internals \
  --bench polynomial_regression -- \
  'polynomial_regression_degree_4/summarized_population_apple_accelerate/500'
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_apple_accelerate_1t/128'
```
