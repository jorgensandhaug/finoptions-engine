# ADR 0001: Exact event identities and node-aligned log discounts

Status: accepted

The engine accepts strictly increasing finite year fractions beginning at canonical positive zero,
integer node identities, and strictly increasing event schedules. Exercise at a dividend node is an
explicit pre-jump phase; expiry is the final, separately ordered terminal node. Events are never
rounded, shifted, merged, sorted, dropped, or inferred from floating-point ratios.

Deterministic curves enter as `log(P(0,t))` values aligned one-for-one with the model grid. The
origin is canonical positive zero, negative interval rates are supported, and every interval
discount must remain finite and strictly positive when exponentiated. Vendor rates and compounding
conventions belong in source-specific adapters rather than this core.

Validated Rust types cannot be deserialized around their constructors. Python specifications are
immutable, enforce exact runtime container/scalar types, and call the same Rust validators used by
future pricing kernels. Unsupported or malformed input raises the public `InputError`; there is no
coercion or fallback.
