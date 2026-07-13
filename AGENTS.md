# Agent working agreement

## Mission

Build the fastest practical engine among implementations that pass the financial correctness
contract. Keep the Python surface tiny and intuitive and the Rust implementation idiomatic, safe,
measured, and explicit.

## Non-negotiable rules

- Never silently change a financial model, event, unit, estimator, basis, precision, or backend.
- Never add fallback algorithms. Return a typed contextual error when a configured method is
  unavailable or outside its domain.
- Do not describe simulated, regressed, discretized, or approximate output as exact.
- Do not require independent numerical methods to match digits; use method-appropriate identities,
  convergence, and uncertainty criteria fixed before inspecting candidate outputs.
- Do not add Polars, Python-framework, vendor, or paper dependencies to `finoptions-core`.
- Do not add unsafe Rust without an isolated safety proof, scalar differential tests, and measured
  end-to-end benefit.
- Do not publish the package or make the repository public.

## Supported workflow

- `just fmt` formats Rust and Python.
- `just check` runs formatting, release-quality lint, Rust tests, Ruff, and strict `ty` checking.
- `just test` builds the native extension through the locked environment and runs integration tests.
- `just full` runs the complete current package gate.

Commit exact locks. Use current stable mutually compatible dependencies and record any compatibility
pin with a removal trigger.

## Architecture

`runtime` and `core` are independent foundations; `sim` depends on both, `batch` depends only on
`runtime`, and `lsm` depends on `sim`. Python is the outer typed facade; `testkit` is non-production
validation. Resolve runtime choices outside hot loops. Keep native entry points private behind the
typed Python facade. Add abstractions only for variation required by a validated workload.
