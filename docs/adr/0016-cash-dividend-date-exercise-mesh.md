# ADR 0016: Explicit cash-dividend-date exercise mesh

- Status: accepted for theorem-domain fixed-strike calls
- Date: 2026-07-15

## Context

The uniform 150-interval Bermudan mesh inherited from the submitted project dominates the cost of
exact-cash and escrowed-cash LSM. Most of those exercise regressions are unnecessary for a
fixed-strike call in the relevant analytical domain.

Between deterministic cash events, the stock pays no dividend. At any time before the next event,
continuing at least until that event's left limit gives the value of a European call over a
dividend-free interval. With a nonnegative discount rate, that value is no less than immediate call
exercise: the usual lower bound is
`C >= S - K exp(-r delta) >= S - K`. Exercise inside the interval is therefore dominated. The only
early-exercise opportunities required by the American call problem are the left limits of strictly
pre-expiry cash events; expiry remains the final stopping opportunity. This is the interval-wise
application of the no-early-exercise result in Merton's
[theory of rational option pricing](https://doi.org/10.2307/3003143) to a deterministic discrete
dividend schedule.

The conclusion does not hold from this argument under negative rates. A cash event represented at
the same numeric time as expiry also needs two terminal phases—exercise on the cum-dividend left
limit, followed by the cash jump and expiry payoff—which the current terminal event type does not
represent.

## Decision

Add `CashDividendExerciseMesh` as a distinct Rust and Python method identity. It constructs an
event-aligned grid containing origin, every exact cash-event time, and expiry, while admitting
exercise only immediately before strictly pre-expiry cash events. Valuation origin is excluded.
An empty cash schedule consequently has expiry as its only stopping opportunity.

Before allocating or simulating any path in the request, row and grouped batches validate the
complete specialization domain:

- the dividend representation is exact cash or escrowed cash, never continuous yield;
- every constant continuously compounded rate is nonnegative; and
- no cash event has a bit-identical time equal to maturity.

Any violation returns its typed row and event error. The engine never changes the mesh, moves an
event, partitions a request, or retries with a uniform method. A research adapter may explicitly
partition a heterogeneous table into separately declared theorem-domain and general-domain runs;
that partition and the merge order are study methodology, not runtime recovery.

The specialization changes only the exercise-opportunity geometry. Cash-jump policy, escrowed
volatility convention, regression basis, support policy, tie rule, training and independent
evaluation populations, random streams, math backend, and diagnostics remain explicit and
unchanged. The resulting finite-path LSM estimate is not described as an exact analytical price.

## Evidence

- The specialized mesh is bitwise identical in every price and diagnostic field to a uniform
  two-interval mesh when one cash event at one half is their identical opportunity set.
- Empty cash schedules produce zero regression nodes and expiry-only stopping.
- Rust and Python tests reject proportional yield, negative rate, and terminal cash without
  fallback; validation covers the entire request before path execution.
- On one exact 256-interval terminal-cash schedule, direct quadrature with all 256 pre-expiry
  exercise nodes agrees within `0.002` with the same grid restricted to its explicitly declared
  final predecessor node, and with an independent 16,384-step interpolated CRR tree. This validates
  the theorem diagnostic; it does not relax the production mesh's terminal-cash rejection.
- Retained project-shaped measurements are recorded in
  [`../performance/2026-07-15-cash-dividend-date-lsm.md`](../performance/2026-07-15-cash-dividend-date-lsm.md).

## Consequences

- Exact- and escrowed-cash publication runs in the proven domain avoid economically redundant
  simulations and continuation regressions.
- Negative-rate and terminal-cash rows require a separately named general mesh until a broader
  theorem or an explicit terminal left-limit phase is implemented and tested.
- Continuous-yield models retain their general uniform mesh.
- A result manifest must identify the mesh variant; two meshes are not silently treated as the
  same finite-sample estimator even when they target the same American contract value.
