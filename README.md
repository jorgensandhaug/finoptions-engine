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

## Validated specifications

Financial schedules are immutable and explicit. Construction fails before pricing if an event is
ambiguous, duplicated, reordered, off-grid, non-finite, or not canonically represented:

```python
from finoptions import CashDividend, EventSchedule, LogDiscountCurve

schedule = EventSchedule(
    times=(0.0, 0.5, 1.0),
    dividends=(CashDividend(node=1, amount=2.0),),
    exercise_nodes=(1,),
    expiry_node=2,
)
curve = LogDiscountCurve(
    times=schedule.times,
    log_discounts=(0.0, -0.01, -0.025),
)
```

Exercise at a dividend node occurs in the pre-dividend state; expiry is a separate terminal node.
No pricing method is exported until its complete estimator and independent validation gate pass.

The repository is private, not yet licensed for redistribution, and no package publication is
authorized.
