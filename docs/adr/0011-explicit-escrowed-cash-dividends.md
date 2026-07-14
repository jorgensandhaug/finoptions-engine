# ADR 0011: Explicit escrowed cash-dividend approximation

- Status: accepted for constant-rate, constant-volatility call batches
- Date: 2026-07-14

## Context

The recovered project study compared an exact cash-jump-style baseline with a distinct escrowed
cash-dividend approximation. Treating those as interchangeable would erase the model identity and
make the volatility input ambiguous. The escrowed construction is well known as an approximation
that subtracts the present value of deterministic future dividends from spot; Haug, Haug, and
Lewis, *Back to Basics: A New Approach to the Discrete Dividend Problem* (2003), discuss both this
construction and origin spot-volatility rescaling.

## Decision

Represent the observed pre-event stock as

\[
S_t = F_t + A_t, \qquad
A_t = \sum_{\tau_k \ge t} D_k P(t,\tau_k),
\]

where the risky prepaid forward `F` follows risk-neutral GBM and `A` is deterministic prepaid cash.
At an exact dividend node, exercise observes `A` including the same-node cash amount; the amount
then leaves the stock while `F` remains continuous. This is an escrowed observation model, not a
limited-liability spot jump.

The request must name one volatility convention:

- `prepaid_forward`: the supplied value is already the volatility of `F`; or
- `spot_matched_at_origin`: a supplied origin spot volatility is transformed as
  `sigma_F = sigma_S * S_0 / F_0`, matching initial dollar volatility.

`F_0 = S_0 - A_0` must be finite and strictly positive. Non-representable prepaid cash or
volatility fails before simulation. The engine never changes the cash schedule, chooses a different
convention, or falls back to the exact cash-jump model.

## Evidence

- Exact deterministic path tests verify origin identity, pre-event observation, the cash-component
  drop, and continuity of the risky prepaid forward.
- Discounted terminal call payoffs converge within a predeclared sampling band to the independent
  Black–Scholes closed form with underlying `F_0`.
- A batch row matches an independently assembled scalar escrowed plan bit for bit.
- Serial and contract-parallel batches are bitwise identical.
- Python tests prove that `spot_matched_at_origin` is exactly equivalent to supplying its explicitly
  transformed `prepaid_forward` volatility and that nonpositive `F_0` fails early.

## Consequences

- Results can name the recovered approximation without implying equivalence to an exact cash-jump
  process.
- The approximation and volatility convention become immutable run-manifest inputs.
- Alternative time-weighted volatility adjustments, stochastic dividends, and calibration from
  option-implied volatility are outside this accepted slice and cannot be inferred by the engine.
