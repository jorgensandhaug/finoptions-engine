# Grouped continuous-yield American-call benchmark

- Date: 2026-07-15
- Machine: Apple M4 Pro, macOS arm64
- Build: release PyO3 extension
- Input: receipt-verified private project cohort; aggregate evidence only
- Shape: 96,666 natural states, 678,122 call strikes
- Selection: target-free; observed option prices were not used

## Frozen paper profile

The selected profile is `(boundary quadrature=13, ordinary iterations=16, collocation nodes=6,
price quadrature=21, maximum relative boundary residual=1e-4, maximum relative obstacle
projection=1e-6)`.

| Yield input | 1 thread | 8 threads | Speedup | Max residual | Max obstacle projection |
|---|---:|---:|---:|---:|---:|
| Forecast | 3.741 s | 0.563 s | 6.64× | 1.298e-5 | 2.622e-7 |
| Historical | 6.435 s | 0.891 s | 7.22× | 1.504e-6 | 3.016e-7 |

Serial and parallel outputs were bitwise identical, including origin stopping decisions and group
diagnostics. The forecast theorem route priced 38,340 zero-yield/nonnegative-rate groups directly
as European calls; historical yields required boundary solves for every group.

## Target-free nested comparison

The over-resolved reference was `(81,48,32,131,1e-8,1e-8)`. It took 49.75 seconds forecast and
76.67 seconds historical on eight threads.

| Comparison | Forecast | Historical |
|---|---:|---:|
| Maximum absolute price difference | $0.0004012 | $0.0003998 |
| Mean absolute price difference | $0.00001385 | $0.00001618 |
| RMSE | $0.00003459 | $0.00003493 |
| 99.9th percentile absolute difference | $0.0003135 | $0.0003155 |
| Maximum relative difference | 0.00536% | 0.00372% |
| Differences above $0.001 | 0 | 0 |
| Origin-decision disagreements | 188 / 678,122 | 430 / 678,122 |

All decision disagreements occur at the discretized free boundary; both configurations satisfy the
same financial bounds and explicit convergence gates.

An extreme four-node profile `(8,16,4,13,1e-3,1e-4)` completed in 0.250/0.409 seconds, but added up
to $0.00070 versus the six-node profile. The paper profile retains six nodes because its sub-second
absolute saving is no longer material to whole-study iteration, while its direct over-resolved
margin is stronger. The engine does not select between profiles at runtime.
