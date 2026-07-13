# ADR 0008: Ragged cash-dividend model tensors

- Status: accepted for the first model-input slice
- Date: 2026-07-13

## Context

Dividend schedules are short and variable length. Persisting a padded table repeats zeros, fixes an
arbitrary capacity, and duplicates the exact source schedule for every downstream model. Native
kernels instead need contiguous values, exact row boundaries, explicit units, and bounded outputs.

## Decision

Use Arrow `LargeList`-compatible `i64` offsets and aligned `f64` event-time/cash-amount child buffers
as the durable and audit representation. A validated Rust view borrows these buffers without copying;
nonzero first offsets are supported for sliced Arrow arrays. Future times must be finite, positive,
and nondecreasing within a row. Cash amounts must be finite and canonically nonnegative. Duplicate
times remain distinct components.

Create dense data only for a named consumer. The first operations produce either chronological
right-padded sequences with a mandatory Boolean mask and `u32` lengths, or chronological fixed-width
cash-amount bins. The request must declare capacity or bin count, event-at-maturity policy, amount
units, and—when binning—horizon and interval closure. Spot scaling requires an explicit positive
per-row spot vector. There is no default convention.

Model tensors are explicitly `f32`, while source schedules remain `f64`. Every narrowing rejects
overflow, underflow to zero, non-finite results, or relative conversion error beyond float32 epsilon.
Output shapes use checked arithmetic and fallible reservation. A single explicit `CpuExecutor`
selects the serial or dedicated-pool kernel; both preserve row order and produce identical values.

The Python package initially accepts contiguous NumPy views so the Rust core has no Arrow, Polars,
or ML-framework dependency. A Polars-side adapter may materialize bounded offsets and borrow or copy
primitive child buffers according to an explicit batch contract.

## Consequences

- Exact source schedules are stored once; padding, masks, and bins are ephemeral.
- Consumers allocate only the representation they request.
- Missing masks, ambiguous edge closure, implicit amount normalization, and silent precision changes
  are structurally unavailable through the typed public API.
- Simulation can later consume the same exact borrowed view without adopting model-tensor precision.
- A direct Arrow C-data boundary remains possible if measured conversion cost warrants its additional
  dependency and safety surface.
