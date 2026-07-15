# Event-aligned direct-quadrature checkpoint

- Date: 2026-07-15
- Host: Apple M4 Pro, macOS arm64, 48 GB unified memory
- Runtime: CPython 3.14, Rust 1.97.0, release profile
- Workload: `S=K=100`, `r=6%`, `sigma=30%`, one liquidator cash dividend `D=7` at
  `t=0.5`, exercise only at the dividend left limit, expiry `T=1`, upper stock boundary 600,
  20,001 uniform stock points
- Samples: one unmeasured warm-up followed by 15 complete Python calls; table reports median and
  observed endpoint range

| Normal order | Threads | Exponential backend | Median | Endpoint range |
| ---: | ---: | --- | ---: | ---: |
| 256 | 1 | portable `f64` | 15.099 ms | 14.767–15.583 ms |
| 256 | 4 | portable `f64` | 3.897 ms | 3.812–5.213 ms |
| 256 | 4 | Apple Accelerate | 3.925 ms | 3.841–4.153 ms |
| 1,024 | 1 | portable `f64` | 42.389 ms | 42.096–43.624 ms |
| 1,024 | 4 | portable `f64` | 10.980 ms | 10.702–11.985 ms |
| 1,024 | 4 | Apple Accelerate | 10.951 ms | 10.668–11.235 ms |
| 2,048 | 1 | portable `f64` | 62.542 ms | 61.718–65.232 ms |
| 2,048 | 4 | portable `f64` | 15.836 ms | 15.701–16.824 ms |
| 2,048 | 4 | Apple Accelerate | 15.915 ms | 15.658–17.523 ms |

The portable four-thread kernel scales by about 3.86x at order 1,024 and 3.95x at order 2,048
relative to one thread. The selected exponential backend has little effect because a reusable method
owns the version-pinned normal rule and each price exponentiates only the retained rule nodes; the
dominant work is independent stock-grid integration. The explicitly requested Apple backend remains
valuable for model consistency with other engine kernels, but this workload does not justify making
it a default.

Numerical output is intentionally not monotone in Gauss-Hermite order at payoff kinks. Orders 256,
1,024, and 2,048 produced 11.6587279882, 11.6565487171, and 11.6559918905, respectively, while the
independent adaptive reference is 11.6564489477. Therefore order is not a performance-only knob and
no value in this report is a production default. Publication workflows must predeclare joint stock
grid, boundary, and quadrature refinements and apply cross-method envelopes rather than selecting the
fastest or closest observed result.
