# finoptions-engine

Private pre-release repository for a high-performance, financially explicit options research
engine. It presents one small typed Python package backed by dependency-isolated Rust crates. Unsafe
code is confined to one audited Apple Accelerate FFI wrapper; every public interface remains safe.

The first supported slice is deliberately narrow: event-aligned deterministic cash dividends,
node-aligned deterministic discount factors and cumulative continuous proportional-dividend
yields, constant-volatility risk-neutral GBM, explicit finite exercise schedules, and exact
serial/parallel path dynamics from either explicit normal variates or path-addressable seeded
streams, independently fitted/evaluated LSM for fixed-strike calls, and deterministic
event-aligned direct integration for the same finite call schedules. Its
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
    CpuMathBackend,
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
    executor=CpuExecutor(threads=4, math_backend=CpuMathBackend.RUST_STD_F64),
    paths=100_000,
    random_stream=RandomStream(seed=7, domain=1),
)
```

The seeded kernel fuses random generation and path evolution without allocating an interval-by-path
normal matrix. Seed, domain, and path index define stream identity; use distinct domains for path
sets that must be independent. The explicit-normal `simulate_gbm` route remains available for
audits and externally generated variance-reduction designs. A dedicated executor never falls back
to the global thread pool or another backend.

`CpuExecutor` defaults to the portable `CpuMathBackend.RUST_STD_F64`. On macOS, callers may request
`CpuMathBackend.APPLE_ACCELERATE` explicitly; unsupported platforms fail during executor
construction. The Apple backend uses Accelerate vForce for GBM exponentials. On Apple Silicon it
uses the baseline AArch64 NEON ISA for fused continuation-regression recurrences and reductions;
other macOS architectures retain vDSP. The selected backend is inspectable and belongs in every
reproducible execution manifest because vector and scalar implementations need not return identical
bits.

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
from finoptions import (
    InsufficientExerciseSupportPolicy,
    Lsm,
    RandomStream,
    price_lsm_call,
)

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
        insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
    ),
)
```

`result` includes the independent mean, standard error, stopping counts, and every exercise-node
regression diagnostic. Reused random domains, unsupported degree, deficient rank/support, excessive
condition, or non-finite arithmetic raise without changing the estimator.

The independent deterministic route carries its own explicit numerical identity:

```python
from finoptions import EventAlignedQuadrature, price_event_aligned_call

reference = price_event_aligned_call(
    initial_spot=100.0,
    strike=100.0,
    schedule=schedule,
    curve=curve,
    dividend_yield_curve=dividend_yield_curve,
    volatility=ConstantVolatility(0.20),
    dividend_policy=DividendPolicy.LIQUIDATOR,
    executor=CpuExecutor(threads=4),
    method=EventAlignedQuadrature(
        upper_stock=600.0,
        stock_points=20_001,
        normal_order=1_024,
    ),
)
```

This method applies exercise before a same-node cash jump and an admitted terminal cash jump before
expiry payoff. It uses version-pinned standard-normal rules, linear interpolation, and an explicit
affine upper tail. `reference.maximum_upper_boundary_gap` exposes the largest computed-grid/tail
discontinuity. The engine does not select a grid, boundary, order, or refinement sequence; research
workflows must predeclare convergence gates and retain every level. The production implementation
is checked against a separate adaptive one-dividend formula and an independently interpolated cash
tree in the non-production Rust testkit.

For research tables, the same estimator prices strict contiguous columns with contract-level
parallelism and no implicit dtype or layout conversion:

```python
import numpy as np

from finoptions import (
    ContinuousDividendYieldBatch,
    OriginExercise,
    UniformExerciseMesh,
    price_lsm_call_batch,
)

batch_result = price_lsm_call_batch(
    initial_spots=np.array([145.0, 155.0], dtype=np.float64),
    strikes=np.array([100.0, 105.0], dtype=np.float64),
    maturities=np.array([1.0, 1.0], dtype=np.float64),
    rates=np.array([0.04, 0.05], dtype=np.float64),
    volatilities=np.array([0.20, 0.25], dtype=np.float64),
    dividends=ContinuousDividendYieldBatch(
        annualized_decimals=np.array([0.02, 0.03], dtype=np.float64),
    ),
    mesh=UniformExerciseMesh(
        base_intervals=150,
        origin_exercise=OriginExercise.EXCLUDED,
    ),
    executor=CpuExecutor(threads=4),
    method=Lsm(
        training_paths=500,
        evaluation_paths=500,
        training_stream=RandomStream(seed=7, domain=0),
        evaluation_stream=RandomStream(seed=11, domain=1_000_000),
        insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
    ),
)
```

Rows receive disjoint, addressable random domains. Exact cash schedules use `CashDividendBatch`.
For fixed-strike calls with deterministic cash dividends, nonnegative rates, and no cash event at
expiry, the explicit `CashDividendExerciseMesh()` admits only the financially complete cash-event
left limits and expiry. Requests outside that theorem domain fail before simulation; the engine
never substitutes a uniform mesh. Continuous-yield and other general-domain runs use the declared
`UniformExerciseMesh`.

Arbitrary finite Bermudan contracts use exact row-aligned times without retaining a redundant
uniform grid:

```python
from finoptions import ExplicitExerciseMesh, RaggedExerciseTimes

explicit_mesh = ExplicitExerciseMesh(
    schedules=RaggedExerciseTimes(
        offsets=np.array([0, 1, 2], dtype=np.int64),
        times=np.array([255.0 / 256.0, 0.5], dtype=np.float64),
    ),
    origin_exercise=OriginExercise.EXCLUDED,
)
```

The first row declares one predecessor before a terminal event; the second declares one half-year
opportunity. Cash times are added as transition nodes but do not become exercisable unless listed.
Times must be positive, strictly increasing within each row, and strictly before that row's expiry.
The engine never derives a prior business day, subtracts an epsilon, sorts, broadcasts, or clips a
schedule. Real calendar semantics belong to the data adapter.

The distinct prepaid-forward approximation uses `EscrowedCashDividendBatch` and requires either
`EscrowedVolatilityConvention.PREPAID_FORWARD` or
`EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN`; no dividend representation or volatility
meaning is inferred. Both cash representations also require `MaturityBoundary`: `STRICTLY_BEFORE`
rejects an event equal to expiry, while `AT_OR_BEFORE` orders an equal-time cash jump before the
terminal payoff. Events are never clipped or shifted. Parallel and serial execution of an identical
request are bitwise equivalent, while independent numerical methods are validated by financial
identities, bounds, convergence, and justified tolerances rather than forced equality.

When several strikes share one exact simulated state, use `LsmStrikeGroups` and
`price_lsm_call_grouped_batch`. Group offsets and compact state columns are explicit NumPy buffers;
the engine never guesses groups from approximate values or shares a strike-specific stopping
policy. Each group simulates one training/evaluation path pair, while every strike fits and evaluates
its own policy on those common random numbers. The architecture and measured project-shaped speedup
are recorded in [ADR 0013](docs/adr/0013-group-addressed-strike-lsm.md) and the
[grouped-strike performance checkpoint](docs/performance/2026-07-14-grouped-strike-lsm.md).

The current project-shaped CPU and memory measurements are recorded as a reproducible
[performance checkpoint](docs/performance/2026-07-14-lsm-batch.md).

The repository is private, not yet licensed for redistribution, and no package publication is
authorized.
