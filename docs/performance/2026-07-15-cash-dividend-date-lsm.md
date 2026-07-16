# Cash-dividend-date LSM performance checkpoint: 2026-07-15

This is a reproducible local comparison, not a cross-machine guarantee. It measures the explicit
`CashDividendExerciseMesh` specialization accepted in ADR 0016 against the general 150-interval
uniform mesh. Both target the same American fixed-strike call problem for this theorem-domain
workload, but they are separately identified finite-path LSM estimators: the specialized mesh fits
only at the cash-event left limit, whereas the general mesh also fits at economically redundant
intermediate dates.

## Environment and workload

- Apple M4 Pro, 14 CPU cores, 48 GiB unified memory
- macOS 26.2, arm64
- Rust 1.97.0, Criterion 0.8.2, `--profile bench`
- 128 exact shared simulation states and seven strikes per state: 896 contracts
- one exact liquidator cash dividend at year fraction 0.5 and maturity 1.0
- positive constant rates, so the cash-dividend-date theorem domain holds
- 500 policy-fitting paths and 500 disjoint evaluation paths
- checked degree-four continuation regression with fail-on-insufficient-support policy
- explicit `apple_accelerate` math backend
- complete timed request validation, plan construction, fitting, and evaluation

## Results

| Executor | Exercise mesh | Criterion center | Interval | Throughput |
|---:|---|---:|---:|---:|
| 1 thread | Uniform, 150 intervals | 1.5665 s | 1.5530–1.5827 s | 571.98 contracts/s |
| 1 thread | Cash-dividend dates | 22.901 ms | 22.835–22.985 ms | 39,125 contracts/s |
| 8 threads | Uniform, 150 intervals | 314.50 ms | 308.98–320.61 ms | 2,848.9 contracts/s |
| 8 threads | Cash-dividend dates | 5.7256 ms | 5.2773–6.3265 ms | 156,490 contracts/s |

The specialized request is **68.4 times faster on one thread** and **54.9 times faster with eight
threads** at the reported centers. The eight-thread specialized sample contained two high severe
outliers, so its interval is reported rather than hidden. The one-thread comparison most directly
measures removed path transitions and regressions; the eight-thread comparison additionally
includes scheduler and machine-contention effects.

This gain does not come from reducing path counts, sharing a strike policy, changing the regression,
using in-sample valuation, or falling back after a failed fit. It comes from representing the
financially complete exercise-opportunity set for this restricted call domain. Proportional-yield,
negative-rate, and terminal-cash requests fail rather than entering this benchmarked path.

Reproduce the retained measurements with:

```shell
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_cash_dividend_dates_500_lsm/grouped_apple_accelerate_1t/128' --noplot
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_cash_dividend_dates_500_lsm/grouped_apple_accelerate_8t/128' --noplot
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_apple_accelerate_1t/128' --noplot
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/grouped_apple_accelerate_8t/128' --noplot
```
