# Exact exercise-time batch overhead

Date: 2026-07-15  
Engine revision: `15b1cca75d7ad3e74fe991a6fa35ca5bf67d6d21`

## Question

Does the generic row-aligned explicit exercise API add material runtime overhead relative to the
theorem-specialized cash-dividend-date mesh when both construct exactly the same event grid?

## Workload

- 128 independent simulation groups and seven strikes per group: 896 option prices.
- One-year maturity, one cash event and one exercise opportunity at `t = 0.5`.
- 500 policy-training paths plus 500 independent valuation paths per group.
- Degree-four polynomial policy, Apple Accelerate math backend.
- Fixed seeds and random domains; identical numerical work in both methods.
- Criterion: 2 s warm-up, 5 s measurement, 10 samples.
- Apple M4 Pro, arm64, macOS 26.2, Rust/Cargo 1.97.0.

The `grouped` route uses `CashDividendExerciseMesh`. The `explicit` route supplies the same `0.5`
time through a zero-copy `RaggedExerciseTimes` buffer. Times are wall-clock means; intervals are
Criterion 95% confidence intervals.

| Workers | Specialized grouped | Generic explicit | Difference |
|---:|---:|---:|---:|
| 1 | 16.008 ms [15.892, 16.119] | 16.084 ms [15.965, 16.229] | +0.47% |
| 8 | 2.724 ms [2.704, 2.745] | 2.734 ms [2.706, 2.766] | +0.36% |

The confidence intervals overlap strongly. This run detects no material cost from the generic
schedule representation; the small point-estimate differences are benchmark noise, not evidence of
a regression. Eight workers provide about 5.88x speedup for both routes in this workload.

## Reproduction

```console
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_cash_dividend_dates_500_lsm/(grouped|explicit)_apple_accelerate_(1|8)t/128' \
  --noplot
```

Criterion's machine-readable estimates are under
`target/criterion/project_grouped_cash_dividend_dates_500_lsm/*/128/new/estimates.json`.
