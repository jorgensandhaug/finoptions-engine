# finoptions-engine

Private pre-release repository for a high-performance, financially explicit options research
engine. It presents one small typed Python package backed by safe, dependency-isolated Rust crates.

The first supported slice is deliberately narrow: event-aligned deterministic cash dividends,
node-aligned deterministic discount factors, constant-volatility risk-neutral GBM, explicit finite
exercise schedules, and independently fitted/evaluated LSM for calls. Unsupported semantics fail;
the engine never substitutes a nearby model, algorithm, precision, event time, or backend.

## Development

Prerequisites are `uv`, Rust 1.97.0, and `just`.

```shell
just sync
just test
just check
```

The repository is private, not yet licensed for redistribution, and no package publication is
authorized.

