# Apple Accelerate FFI safety contract

This crate is the only place where the workspace permits unsafe Rust. Its public API accepts an
immutable `f64` input slice and a distinct mutable `f64` output slice of equal length. Safe Rust
therefore guarantees live, aligned, non-overlapping buffers for the complete native call.

The wrapper calls the macOS SDK declaration
`vvexp(double *output, const double *input, const int *count)`. Inputs larger than `i32::MAX` are
split into equal contiguous chunks before conversion of the element count. The count and both
slice chunks remain live for each call. Empty slices make no FFI call. The module links only on
macOS; other targets return a typed unavailable-platform error and never substitute another
implementation.

The FFI safety contract is separate from numerical equivalence. Differential tests require vForce
to remain within one ULP of Rust standard-library `f64::exp` over the declared pricing exponent
domain. Pricing code records the selected backend because exact bits are not promised across the
two implementations.
