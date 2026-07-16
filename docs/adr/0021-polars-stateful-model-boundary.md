# ADR 0021: Polars lazy-plan and stateful-model boundary

- Status: accepted
- Date: 2026-07-16

## Context

Research tables benefit from Polars scans, projection and predicate pushdown, streaming sinks, and
typed expressions. Pricing and training kernels have different semantics. Grouped LSM owns a
complete simulation group, globally addressed random streams, and one stopping policy per strike.
Rolling neural estimation owns complete training, validation, and test populations. Arbitrary
dataframe chunks are therefore not independent execution units for either operation.

Polars expression plugins are attractive for native elementwise work, while `LazyFrame.map_batches`
can place a Python callback inside a lazy plan. Both interfaces can be misused to make a stateful
estimator appear streamable. A batch callback is streamable only when every partitioning produces
the same result as executing on the complete input. LSM groups can cross chunk boundaries, row-based
random identities change when chunks change, and model fitting depends on the complete declared
population. Marking these operations as streamable would silently change the estimator.

The current native engine uses PyO3 and Rust NumPy 0.29. The current `pyo3-polars` 0.27 graph uses
PyO3 0.28 and Rust Polars 0.54.4. Pulling that graph into the numerical extension would couple the
engine to a second native dataframe ABI, materially increase cold build cost, and either downgrade
or duplicate the Python binding stack. A measured project-shaped spike also found that a built-in
Polars expression was faster than an equivalent Rust plugin for a simple financial transform.

## Decision

Polars is the primary lazy columnar engine for research-data preparation, but it is not a dependency
of any numerical Rust crate. Lazy scans, joins, filters, projections, and stateless feature
expressions remain lazy for as long as possible. Operations expressible from built-in Polars
expressions use those expressions rather than a plugin.

Stateful simulation, pricing, inference, and training execute after one explicit, named
materialization boundary. The engine never implicitly collects a `LazyFrame`. A Polars adapter may
accept a materialized `DataFrame`, validate its complete schema and financial domain, and borrow
single-chunk primitive or Arrow-compatible list buffers. Any cast, rechunk, sort, grouping, or copy
requires an explicit caller policy. Results preserve the declared source-row order and return as a
separate, typed result table that can be joined only by an explicit stable identity or verified
row-order contract.

Grouped pricing has two explicit materialized axes. One state table contains exactly one row per
independently simulated market state, a second strike table contains one row per output contract,
and a caller-supplied `UInt32` offset vector maps states to contiguous strike ranges. The adapter
does not repeat state columns per strike or infer groups from value equality, keys, sorting, or
dataframe partitioning.

LSM, simulation, calibration, bootstrap, and neural training are never exposed as elementwise
plugins or streamable batch UDFs. A non-streamable `map_batches` wrapper is also rejected: it hides
the important execution boundary, obstructs resource and failure control, and cannot improve
pushdown beyond a projection constructed before collection. All requested rows are validated before
native execution, and a failing row fails the complete request without a partial dataframe.

An optional Python-only Polars adapter may depend on the current stable Python Polars API without
changing the Rust dependency graph or the base NumPy-only installation. A Rust expression-plugin
adapter must remain a separate optional crate and distribution component. It is admitted only for a
complex stateless transform whose benchmark beats a built-in expression or avoids a material memory
cost, and only when its exact Polars/PyO3 ABI graph is compatible with the main extension.

## Consequences

- Research pipelines retain lazy pushdown and streaming until the smallest complete model input is
  materialized.
- Stateful estimator identity cannot depend on an optimizer-selected chunk layout.
- The numerical core remains usable without Polars, Arrow, or a dataframe ABI.
- Simple feature formulas stay visible and optimizable in Python Polars expressions.
- Model boundaries must expose their allocation, rechunking, ordering, grouping, random-domain, and
  resource policies instead of relying on dataframe defaults.
- GPU or streaming fallback selected by a dataframe runtime is outside the pricing method. A
  publication workflow that requires a specific dataframe engine must verify that engine
  separately rather than treating a transparent fallback as equivalent execution.

## Re-evaluation triggers

- Polars gains a native stateful operator contract with stable whole-group partitioning and explicit
  resource and failure semantics.
- A direct Arrow C-data boundary measurably reduces model-boundary time or peak memory.
- `pyo3-polars` shares the engine's PyO3 line and a justified stateless plugin outperforms the
  built-in expression on representative data.
- A supported backend can prove exact group and random-domain invariance under distributed
  execution.
