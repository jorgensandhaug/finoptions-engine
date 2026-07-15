# Fused orthogonal-recurrence moment checkpoint: 2026-07-15

This checkpoint fuses construction of each nonconstant empirical orthogonal-polynomial basis
column with the three reductions immediately required to fit it. The former implementation first
traversed every in-the-money path to compute
`next = (predictor - alpha) * current - beta * previous`, then traversed the same population again
for norm, response projection, and predictor-weighted squared norm. The accepted kernel performs
that exact recurrence, stores the next basis, and accumulates all three moments in one pass.

## Numerical and safety contract

- The standardized degree-zero-to-four polynomial space, three-term recurrence, response
  population, coefficients, relative-rank threshold, condition limit, diagnostics, exercise rule,
  and prediction recurrence are unchanged.
- The portable backend uses one safe scalar fused traversal. AArch64 macOS uses one isolated NEON
  traversal with four independent two-lane accumulators, a two-lane tail, and at most one scalar
  value. The explicit `apple_accelerate` identity is unchanged and the engine revision records the
  internal numerical build.
- The safe runtime validates all five vector lengths before output is written. Rust borrowing proves
  that mutable next-basis storage cannot alias predictors, responses, or either input basis.
- Differential tests cover lengths 0--3, every relevant residue around 8/16/128/500/1,024, exact
  scalar-recurrence bits, and compensated moment oracles. Independent SVD tests, all degrees and
  backends, fail-fast error precedence, grouped/row equivalence, thread invariance, and financial
  reference tests exercise the production fit. There is no retry, alternate degree, regularizer,
  population change, or backend fallback.

## Controlled measurements

The retained workload uses an Apple M4 Pro, Rust 1.97.0, Criterion 0.8.2, 128 natural states,
seven strikes per state, 150 intervals, 500 training plus 500 independent evaluation paths,
degree four, and complete request validation, simulation, fitting, and evaluation. The baseline is
clean engine commit `dd7b493cd388c5e4f53cf9c369b8d9114e113d89` in a detached sibling worktree.
Already-built binaries ran sequentially; no benchmark ran concurrently.

| Boundary | Baseline center(s) | Fused center(s) | Mean-center reduction |
|---|---:|---:|---:|
| 500-row recurrence plus moments | 197.68 ns | 187.74 ns | 5.03% |
| Complete Apple workload, 1 thread | 1.5759 s, 1.5588 s | 1.4309 s, 1.4885 s | 6.87% |
| Complete Apple workload, 8 threads | 297.81 ms | 283.99 ms | 4.64% |
| Complete portable scalar workload, 1 thread | 2.0220 s | 1.8917 s | 6.44% |

The one-thread host varied between adjacent repetitions, but the highest candidate interval endpoint
was 1.4904 seconds and the lowest baseline endpoint was 1.5557 seconds. The eight-thread intervals
were 282.57--285.44 ms and 293.87--302.34 ms. Both end-to-end comparisons remain separated rather
than relying on the isolated kernel alone.

## Reproduce

```shell
cargo bench -p finoptions-accelerate --bench polynomial_moments -- \
  'polynomial_recurrence_moments_f64' --noplot
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_apple_accelerate_(1|8)t/128' --noplot
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_rust_std_f64_1t/128' --noplot
```
