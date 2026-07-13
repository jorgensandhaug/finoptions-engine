# ADR 0007: Shared explicit CPU runtime

- Status: accepted
- Date: 2026-07-13

## Context

Simulation, regression, and model-input kernels all need bounded multicore execution. Independent
implicit pools can oversubscribe a laptop and make both speed and memory use unpredictable.
Execution policy is not a simulation concept and should not force unrelated kernels to depend on a
simulation crate.

## Decision

Own one `CpuExecutor` in the dependency-independent `finoptions-runtime` crate. Construction requires
an exact positive worker count. One worker dispatches a separately implemented serial closure;
larger counts use one reusable isolated Rayon pool with exactly the requested workers. The executor
never consults the global Rayon pool, changes the count, or selects another backend.

Simulation re-exports the Rust type for compatibility, while Python exposes one immutable executor
specification shared by every native CPU operation. Pool-construction failures use a distinct typed
execution error.

## Consequences

- Simulation and batch kernels share one resource policy without depending on each other.
- A caller can coordinate Polars, ML, BLAS, and engine thread counts explicitly.
- Serial kernels avoid parallel scheduling overhead, while a requested parallel kernel cannot
  silently run through a global or differently sized pool.
- Hardware-aware planning remains an outer, persisted decision; it may propose a worker count but
  cannot silently alter a submitted execution specification.
