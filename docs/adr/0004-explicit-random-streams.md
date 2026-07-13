# ADR 0004: Explicit path-addressable random streams

Status: accepted

Seeded simulation uses Random123-compatible Philox4x32-10 with a 128-bit counter partitioned into
three injective coordinates: a 64-bit local block, a 32-bit path ID, and a 32-bit domain. The 64-bit
seed is the Philox key. The caller must provide both seed and domain; fitting and evaluation path
sets use distinct domains. Path identity therefore does not depend on worker count, scheduling, or
the total number of paths requested.

Standard normals use the Doornik ZIGNOR transform adapted from Rust `rand_distr`. Rejection is
bounded at 2^20 iterations and returns a typed error at the limit. Each path stream receives an
exact normal budget equal to the model interval count. The bounded outer and tail loops consume at
most 2^21 Philox blocks per requested normal; combined with the engine's 32-bit interval domain,
that is below 2^53 blocks per path and proves its reserved 64-bit lane cannot wrap. Another path or
domain can never be entered. There is no RNG, transform, precision, seed, domain, or backend
fallback.

The fast kernel generates and evolves a complete path in one worker task, keeping the state and RNG
hot and entering the worker pool once. It first writes path-major observation states, then performs
one explicit transpose to the public time-major regression layout. The caller-supplied-normal
kernel remains available as an independent audit/reference route.

Random words and normal outputs are bitwise invariant across worker counts on one numerical build.
Cross-platform bitwise identity is not promised because accepted tail draws and GBM evolution use
platform floating-point transcendental implementations; run manifests must record engine/compiler/
target identity. Correctness is instead gated by Random123 known-answer vectors, transform
differential tests, stream-prefix identity, independent distribution diagnostics, exact GBM
identities, and deterministic failure reporting.
