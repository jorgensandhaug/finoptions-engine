# ADR 0014: Terminal cash-event order

## Decision

Cash-dividend LSM requests declare a `MaturityBoundary`. `STRICTLY_BEFORE` rejects an event whose
time equals contract maturity. `AT_OR_BEFORE` admits equality and gives the terminal node the phase
order cash jump, then expiry payoff. A terminal node is never an exercise-regression node.
Nonterminal nodes retain exercise, then cash jump, so American exercise observes the cum-dividend
state.

The spot simulator therefore records nonterminal exercise states before applying same-node cash,
but applies terminal cash before recording the expiry state. The escrowed approximation includes
terminal cash in prepaid cash at every earlier node and excludes the already-paid amount from its
terminal reconstructed stock. Serial, parallel, scalar-exponential, vector-exponential, and seeded
kernels implement the same ordering.

## Rationale

Equity becomes ex-dividend before trading on the ex-date, whereas an expiring equity option's
terminal value is determined later. The Options Industry Council also states that a call must be
exercised before the ex-dividend date to receive that dividend. A date-only dataset can collapse
the ex-date and expiry-day close to one year fraction; `AT_OR_BEFORE` preserves their economic
order without inventing an epsilon or moving the source event. See the official
[Options Exercise FAQ](https://www.optionseducation.org/referencelibrary/faq/options-exercise).

## Consequences

The caller must choose the boundary explicitly for each cash representation. No default, silent
drop, time shift, or fallback exists. Events after maturity always fail before simulation. The
boundary is part of the typed Python request and the native Rust dividend-model identity.
