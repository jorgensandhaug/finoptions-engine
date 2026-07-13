# finoptions-engine

Private pre-release repository for a high-performance, financially explicit options research
engine. It presents one small typed Python package backed by safe, dependency-isolated Rust crates.

The first supported slice is deliberately narrow: event-aligned deterministic cash dividends,
node-aligned deterministic discount factors and cumulative continuous proportional-dividend
yields, constant-volatility risk-neutral GBM, explicit finite exercise schedules, and exact
serial/parallel path dynamics from either explicit normal variates or path-addressable seeded
streams, and independently fitted/evaluated LSM for fixed-strike calls. Its
checked degree-zero-through-four polynomial regression kernel has passed the solver gate.
Unsupported semantics fail; the engine never substitutes a nearby model, algorithm, precision,
event time, random stream, or backend.

Exact variable-length dividend schedules also cross the native boundary as Arrow-compatible offsets
plus `f64` child values. Padded sequences and time bins are created only when requested.

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
from finoptions import (
    CashDividend,
    CumulativeDividendYieldCurve,
    EventSchedule,
    LogDiscountCurve,
)

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
dividend_yield_curve = CumulativeDividendYieldCurve.zero(times=schedule.times)
```

Exercise at a dividend node occurs in the pre-dividend state; expiry is a separate terminal node.
Execution resources are equally explicit and reusable:

```python
from finoptions import (
    ConstantVolatility,
    CpuExecutor,
    DividendPolicy,
    RandomStream,
    simulate_gbm_seeded,
)

paths = simulate_gbm_seeded(
    initial_spot=100.0,
    schedule=schedule,
    curve=curve,
    dividend_yield_curve=dividend_yield_curve,
    volatility=ConstantVolatility(0.20),
    dividend_policy=DividendPolicy.LIQUIDATOR,
    executor=CpuExecutor(threads=4),
    paths=100_000,
    random_stream=RandomStream(seed=7, domain=1),
)
```

The seeded kernel fuses random generation and path evolution without allocating an interval-by-path
normal matrix. Seed, domain, and path index define stream identity; use distinct domains for path
sets that must be independent. The explicit-normal `simulate_gbm` route remains available for
audits and externally generated variance-reduction designs. A dedicated executor never falls back
to the global thread pool or another backend.

The same executor builds model inputs without storing a second padded dataset:

```python
import numpy as np

from finoptions import (
    CashAmountScale,
    MaturityBoundary,
    PaddedSequenceSpec,
    RaggedCashDividends,
    pack_padded_cash_dividends,
)

dividends = RaggedCashDividends(
    offsets=np.array([0, 2, 2], dtype=np.int64),
    times=np.array([0.25, 0.75], dtype=np.float64),
    amounts=np.array([1.0, 1.0], dtype=np.float64),
)
tensor = pack_padded_cash_dividends(
    dividends=dividends,
    maturities=np.array([1.0, 0.5], dtype=np.float64),
    spots=np.array([100.0, 80.0], dtype=np.float64),
    spec=PaddedSequenceSpec(
        capacity=2,
        maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
        amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
    ),
    executor=CpuExecutor(threads=4),
)
```

`tensor.times`, `tensor.amounts`, `tensor.mask`, and `tensor.lengths` have explicit chronological
semantics. Float32 conversion, capacity, maturity inclusion, scale, and worker count are validated;
none is inferred or substituted.

The first pricing method fits a call stopping policy and values it only on an independently seeded
population:

```python
from finoptions import Lsm, RandomStream, price_lsm_call

result = price_lsm_call(
    initial_spot=100.0,
    strike=100.0,
    schedule=schedule,
    curve=curve,
    dividend_yield_curve=dividend_yield_curve,
    volatility=ConstantVolatility(0.20),
    dividend_policy=DividendPolicy.LIQUIDATOR,
    executor=CpuExecutor(threads=4),
    method=Lsm(
        training_paths=65_536,
        evaluation_paths=65_536,
        training_stream=RandomStream(seed=7, domain=1),
        evaluation_stream=RandomStream(seed=7, domain=2),
    ),
)
```

`result` includes the independent mean, standard error, stopping counts, and every exercise-node
regression diagnostic. Reused random domains, unsupported degree, deficient rank/support, excessive
condition, or non-finite arithmetic raise without changing the estimator.

The repository is private, not yet licensed for redistribution, and no package publication is
authorized.
