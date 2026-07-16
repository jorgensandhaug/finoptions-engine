# finoptions-engine

A high-performance, typed options-research engine written in Rust and exposed through a small
Python API. Financial semantics, numerical methods, random streams, execution resources, and data
layouts are explicit. Unsupported requests fail before pricing; the engine never substitutes a
nearby model, mesh, precision, event time, backend, or fallback algorithm.

This repository is private and pre-release. Package publication and redistribution are not yet
authorized.

## Supported scope

- Event-aligned deterministic cash dividends and continuous proportional-dividend yields.
- Node-aligned continuously compounded discount curves and constant-volatility risk-neutral GBM.
- Independently fitted and evaluated Longstaff--Schwartz calls on explicit exercise schedules.
- Deterministic event-aligned quadrature for finite cash-dividend call schedules.
- Grouped Andersen--Lake American calls under continuous yield.
- Contract- and strike-group-addressed batch pricing with explicit random domains.
- Zero-copy Arrow-compatible ragged cash and exercise schedules across the Python/Rust boundary.
- Optional Polars lazy projection and explicit materialized pricing adapters.
- Portable scalar CPU execution and an explicitly requested Apple Accelerate/NEON backend.

The complete method and domain contract is in the [capability reference](docs/reference.md).

## Quick start

Prerequisites are `uv`, Rust 1.97.0, and `just`.

```shell
just test
just check
```

The commands synchronize the locked Python environment automatically. `just test` builds the
native extension and runs the Python boundary suite; `just check` runs formatting, Clippy with
warnings denied, all Rust unit and documentation tests, Ruff, and ty.

```python
from finoptions import (
    CashDividend,
    ConstantVolatility,
    CpuExecutor,
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EventSchedule,
    InsufficientExerciseSupportPolicy,
    LogDiscountCurve,
    Lsm,
    RandomStream,
    price_lsm_call,
)

schedule = EventSchedule(
    times=(0.0, 0.5, 1.0),
    dividends=(CashDividend(node=1, amount=2.0),),
    exercise_nodes=(1,),
    expiry_node=2,
)
curve = LogDiscountCurve(times=schedule.times, log_discounts=(0.0, -0.01, -0.025))

result = price_lsm_call(
    initial_spot=100.0,
    strike=100.0,
    schedule=schedule,
    curve=curve,
    dividend_yield_curve=CumulativeDividendYieldCurve.zero(times=schedule.times),
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

Training and valuation populations are distinct by construction. The result exposes price,
standard error, stopping counts, and every exercise-node regression diagnostic.

Install the optional Polars surface with `uv sync --extra polars`. It keeps projection lazy but
requires the researcher to make collection visible before stateful pricing:

```python
import finoptions.polars as fpl

columns = fpl.LsmCallColumns(
    initial_spot="spot",
    strike="strike",
    maturity="maturity",
    rate="rate",
    volatility="volatility",
)
dividends = fpl.ContinuousYieldColumn("forecast_yield")
selected = fpl.select_lsm_call_inputs(lazy_quotes, columns=columns, dividends=dividends)
materialized = selected.collect()
result_table = fpl.price_lsm_call_frame(
    materialized,
    columns=columns,
    dividends=dividends,
    mesh=mesh,
    executor=executor,
    method=method,
)
```

The adapter never collects a `LazyFrame`, casts a financial column, guesses a group, or hides LSM
inside a chunk-dependent dataframe UDF. Cash schedules use paired `List(Float64)` columns and an
explicit zero-copy or rechunk policy.

Measure lazy projection, explicit collection and strict native preparation independently on a
synthetic project-scale frame:

```shell
just bench-polars-boundary 1000000 continuous_yield
just bench-polars-boundary 1000000 cash
```

The benchmark verifies that scalar arrays still borrow the materialized Polars buffers and reports
cold-path timings, steady-state medians, phase throughput, and incremental process high-water
memory as canonical JSON.

## Architecture

- `finoptions-core`: schedules, curves, market inputs, and financial event ordering.
- `finoptions-sim`: Philox streams, GBM stepping, path storage, and panel bootstrap kernels.
- `finoptions-lsm`: checked regression, stopping policies, and grouped LSM pricing.
- `finoptions-quadrature`: deterministic event-aligned cash-dividend pricing.
- `finoptions-american`: deterministic continuous-yield American pricing.
- `finoptions-batch`: validated contiguous and ragged native input boundaries.
- `finoptions-runtime`: explicit executors and backend dispatch.
- `finoptions-accelerate`: the isolated audited Apple vector specialization.
- `finoptions-python`: the typed Python surface.
- `finoptions.polars`: the optional Python-only lazy-plan/materialization adapter.

Safe Rust owns all public boundaries. Architecture decisions live in [docs/adr](docs/adr), and
retained controlled measurements live in [docs/performance](docs/performance).

## Reproducibility contract

Thread count, math backend, random seed/domain, path counts, exercise mesh, regression limits, and
method parameters belong to the request identity. Invalid support, rank, conditioning, financial
bounds, convergence, layout, or arithmetic is an error. Serial and parallel execution of an
identical request is tested for bitwise equality where the declared backend is unchanged.
