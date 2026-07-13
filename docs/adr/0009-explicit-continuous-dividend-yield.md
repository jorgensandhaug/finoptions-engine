# ADR 0009: Explicit integrated continuous dividend yield

- Status: accepted for nonnegative proportional payouts
- Date: 2026-07-14

## Context

The study needs cash-dividend, forecast-yield, historical-yield, and no-dividend GBM baselines.
Continuous proportional dividend yield is not a risk-free rate, a deterministic cash jump, or a
generic borrow/carry adjustment. A scalar called `rate` or nodewise spot-yield observations would
leave annualization, interpolation, compounding, and interval integration ambiguous.

## Decision

Represent continuous proportional dividend yield by a node-aligned cumulative curve
`Q(t) = integral_0^t q(u) du`. The curve is always supplied explicitly, including the identically
zero curve. Its origin is canonical positive zero, values are finite and nondecreasing, and every
adjacent factor `exp(-(Q(t_next) - Q(t_current)))` must be finite and strictly positive. This type
therefore admits only nonnegative proportional payouts. Financing, stock borrow, convenience yield,
and other signed carry require separate future types.

For adjacent nodes, let
`log_D_r = log(P_r(0,t_next) / P_r(0,t_current))` and
`Delta_Q = Q(t_next) - Q(t_current)`. The exact pre-event risk-neutral GBM transition is

```text
S_next- = S_current+ * exp(
    -log_D_r - Delta_Q - 0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z
)
```

Only the risk-free curve discounts option cashflows. The dividend-yield curve changes stock
dynamics; it never discounts the strike or payoff. Deterministic cash dividends remain separately
ordered jumps, so an explicitly declared hybrid model can contain both concepts without converting
one into the other.

Python exposes the same cumulative values and two exact constructors: `constant`, which computes
`Q(t)=q*t` from a nonnegative annualized decimal yield, and `zero`. Neither constructor interpolates
or infers a vendor convention.

## Evidence

- Scalar transition tests recover the declared `r - q - sigma^2 / 2` drift for constant curves.
- Irregular adjacent steps telescope to the exact terminal expression containing separate
  risk-free and integrated-yield terms.
- A seeded Monte Carlo diagnostic recovers the theoretical terminal mean
  `S(0) * exp(integral_0^T (r(u)-q(u)) du)` within predeclared sampling uncertainty.
- Grid mismatch, negative interval yield, non-finite input, noncanonical origin, and an
  unrepresentable proportional payout factor are typed failures.

## Consequences

- Project forecast/historical scalar-yield adapters can state exactly which recovered feature is
  treated as a constant continuous payout yield; the engine does not endorse that empirical proxy.
- Cash-dividend and yield baselines remain different financial models even if their prices happen
  to be close.
- Rate and yield curves may share times but remain distinct validated identities.
- Time-varying volatility, stochastic dividends, signed borrow/carry, and implied-yield calibration
  remain outside this accepted slice.

## Sources

- [Merton, “Theory of Rational Option Pricing”](https://doi.org/10.2307/3003143)
- [Longstaff and Schwartz, “Valuing American Options by Simulation”](https://doi.org/10.1093/rfs/14.1.113)
