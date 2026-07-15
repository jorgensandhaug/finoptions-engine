# ADR 0018: Exact row-aligned exercise-time batches

- Status: accepted
- Date: 2026-07-15

## Context

`UniformExerciseMesh` defines a numerical Bermudan approximation and
`CashDividendExerciseMesh` is a theorem-domain optimization for nonnegative-rate, zero-yield calls
with strictly pre-expiry deterministic cash. Neither can express a terminal-cash contract whose
economically relevant dividend-capture decision occurs on an earlier eligible trading date.

A date-only equity dataset may place an ex-dividend event on the option's expiry date. The stock is
already ex-dividend before the terminal option observation, so the engine's cash-then-expiry event
order remains correct. The Options Industry Council states that dividend entitlement requires call
exercise before the ex-dividend date. The prior eligible decision is therefore contract/calendar
data, not a numeric phase the engine may invent at expiry.

The needed abstraction is broader than terminal cash. Research callers need exact finite Bermudan
opportunities per contract or per natural strike group, including continuous-yield and negative-rate
models where the cash-event theorem specialization does not apply.

## Decision

Add zero-copy `RaggedExerciseTimes` over Arrow `LargeList`-compatible `i64` offsets and `f64` child
values, plus `ExplicitExerciseMesh` with an explicit origin convention. The batch method borrows
these schedules. Row batches require one schedule per contract; grouped batches require one per
simulation group. No schedule is broadcast.

Construction rejects malformed offsets, non-finite/nonpositive values, duplicates, and unordered
times. Complete batch validation rejects every time greater than or equal to its row maturity before
path allocation. Expiry remains the final stopping opportunity and is never listed. Origin is a
separate enum rather than the numeric value zero.

For each row, the simulation grid is the exact sorted union of origin, explicit exercise times,
cash-event times, and expiry. Cross-source equal times are coalesced bit-exactly. An explicit time at
a cash node observes the pre-jump state. A cash node absent from the exercise schedule is a
transition only; terminal cash retains cash-then-payoff order. The engine does not keep unrelated
uniform transition nodes, infer a calendar predecessor, subtract an epsilon, snap, sort, deduplicate,
clip, retry, or select another mesh.

The explicit mesh defines a different finite-sample random identity from a redundant uniform grid
because exact GBM transitions are coalesced. Mathematical contract equivalence does not imply
bitwise Monte Carlo equality across those two meshes. Publication workflows must identify the exact
schedule buffers and rerun their frozen numerical pilot before adopting the compact method.

## Evidence

- Rust and Python row tests are bitwise identical to a uniform mesh when the two requests contain
  exactly the same transition and exercise times, including terminal cash.
- Explicit terminal-predecessor tests fit exactly one regression and are invariant to worker count.
- Row and grouped continuous-yield routes are bitwise identical to their exactly equal uniform
  opportunity sets.
- Invalid duplicates, order, positivity, maturity, row count, dtype, and layout fail without
  sorting, casting, broadcasting, or partial output.
- On the Apple M4 Pro publication-shaped benchmark, the generic explicit schedule is within the
  overlapping 95% confidence intervals of the theorem-specialized cash-date schedule at both one
  and eight workers. See
  [`2026-07-15-exact-exercise-times.md`](../performance/2026-07-15-exact-exercise-times.md).
