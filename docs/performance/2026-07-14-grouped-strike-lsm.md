# Grouped-strike LSM performance checkpoint: 2026-07-14

This is a reproducible local comparison, not a cross-machine guarantee. Both paths use the same
financial method, Apple vForce backend, eight-thread executor, contract order, and complete timed
validation/plan construction. Their stochastic identities intentionally differ: the row path uses
one random domain per contract, while the grouped path uses one domain per shared simulation state.

## Environment and workload

- Apple M4 Pro, 14 CPU cores, 48 GiB unified memory
- macOS 26.2, arm64
- Rust 1.97.0, Criterion 0.8.2, `--profile bench`
- 150 Bermudan base intervals
- one exact liquidator cash dividend at year fraction 0.5
- 500 policy-fitting paths and 500 disjoint evaluation paths
- checked degree-four continuation regression
- seven supported in-/at-the-money strikes per exact spot/maturity/rate/volatility/dividend state
- explicit `apple_accelerate_vforce` math backend and eight threads

## Results

| States | Contracts | Execution | Mean elapsed | Throughput |
|---:|---:|---|---:|---:|
| 32 | 224 | Grouped, 32 path pairs | 86.663 ms | 2,584.7 contracts/s |
| 32 | 224 | Row, 224 path pairs | 147.21 ms | 1,521.6 contracts/s |
| 128 | 896 | Grouped, 128 path pairs | 375.68 ms | 2,385.0 contracts/s |
| 128 | 896 | Row, 896 path pairs | 595.16 ms | 1,505.5 contracts/s |

The grouped boundary was 1.70 times as fast in the 224-contract measurement and 1.58 times as fast
in the larger 896-contract measurement. The larger Criterion intervals were 359.88–393.44 ms for
grouped execution and 589.79–601.40 ms for row execution. Grouping eliminates six of seven path
pairs in this workload; strike-specific regression and policy evaluation remain necessary and
therefore bound the end-to-end speedup below seven times. No other research kernel was running
during these measurements.

Grouped execution schedules each simulation state independently so the explicit executor can use
all requested workers. Serial and parallel requests retain bitwise-identical outputs.

Reproduce the release measurements with:

```shell
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/.*apple_accelerate_vforce_8t/32'
cargo bench -p finoptions-lsm --bench lsm_batch -- \
  'project_grouped_strikes_150x500_lsm/.*apple_accelerate_vforce_8t/128'
```

An unsupported 15%-out-of-the-money benchmark draft produced only two in-the-money training paths
at an early node for a degree-four regression. It failed with the declared typed insufficient-
observations error. The accepted performance cohort was narrowed rather than changing degree,
exercise nodes, fitting population, or fallback policy.
