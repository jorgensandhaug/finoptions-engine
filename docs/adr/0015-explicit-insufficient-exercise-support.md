# ADR 0015: Explicit insufficient-exercise-support policy

- Status: accepted
- Date: 2026-07-15

## Context

A degree-`d` polynomial continuation regression has `d + 1` basis columns. Some simulated exercise
nodes, especially for out-of-the-money contracts and small training populations, contain fewer than
`d + 1` in-the-money paths. Those rows cannot identify the declared regression. Changing degree,
adding regularization, expanding the fitting population, or reusing evaluation paths would silently
change the estimator.

There are nevertheless two coherent stopping-policy definitions. A strict numerical-domain method
can reject the fit. A reconstruction or predeclared sparse-support method can make continuation
mandatory at any node where it cannot fit the exercise rule. The latter is an admissible restricted
stopping policy, not an estimate of the missing continuation function. Its population value remains
no greater than the value of the unrestricted declared optimal-stopping problem.

## Decision

Every scalar, row-batch, and grouped-batch LSM method must carry an
`InsufficientExerciseSupportPolicy`; Python provides no default.

- `Fail` preserves the strict support-domain behavior. A positive population smaller than the basis
  raises the typed regression error. Zero training support is recorded, and evaluation fails if an
  active path later enters that unsupported in-the-money state.
- `Continue` fits no regression whenever the ITM training count is smaller than the basis-column
  count, including zero. It records both counts and requires continuation for every evaluation path
  at that node.

The choice is fixed before simulation and is part of the estimator identity. `Continue` does not
retry a failed regression. Rank deficiency, degenerate predictors, excessive condition number,
non-finite arithmetic, and all simulation or event-contract violations remain fatal under both
policies. No partial batch is returned.

## Evidence

- A degree-four, four-ITM-row fixture fails with `observed=4, required=5` under `Fail`.
- The identical fixture under `Continue` reports the exact counts, fits no regression, makes zero
  exercise decisions at that node, and values the independent population normally.
- Python scalar integration verifies the required typed field and the same observable diagnostic.
- Rust scalar, row-batch, and grouped-batch construction all carry the same immutable policy value.

## Consequences

- Sparse-support behavior is reviewable in manifests and cannot be introduced by runtime recovery.
- A study using `Continue` must describe the resulting restricted stopping rule; it must not claim
  that a continuation regression was fitted at the affected nodes.
- Increasing training paths remains the preferred way to reduce sparse nodes when the study intends
  to approximate the full exercise policy, but the engine never changes path counts automatically.
