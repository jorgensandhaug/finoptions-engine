# ADR 0002: Exact adjacent-node GBM transition

Status: accepted for the zero-yield case; extended by ADR 0009

Between ordered financial events, the first engine slice uses risk-neutral GBM with zero continuous
yield/borrow, deterministic node discount factors, and strictly positive constant annualized
decimal volatility. For adjacent nodes, with
`log_D = log(P(0,t_next) / P(0,t_current))`, the transition is

```text
S_next- = S_current+ * exp(
    -log_D - 0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z
)
```

The step precomputes drift and diffusion from the curve's own validated time grid. It rejects
skipped/reversed nodes, negative or non-finite state, non-finite normal variates, and non-finite
results. Canonical positive zero is absorbing. Event jumps and exercise decisions remain separate
ordered operations; the transition never applies or skips them implicitly.

The scalar implementation is the semantic reference for later vectorized kernels. Reordering,
lower precision, approximate exponentials, or fused multi-event steps require separate measured
execution choices and differential evidence.
