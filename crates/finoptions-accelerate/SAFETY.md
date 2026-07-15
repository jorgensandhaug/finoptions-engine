# Apple Accelerate FFI safety contract

This crate is the only place where the workspace permits unsafe Rust. Every native call is behind
a safe slice API that validates the complete operand shape before entering the FFI module. Safe
Rust guarantees live, aligned buffers and prevents the mutable output or scratch slice from
aliasing any immutable input for the duration of a call.

## vForce exponential

The wrapper calls the macOS SDK declaration
`vvexp(double *output, const double *input, const int *count)`. Input and output have equal lengths.
Requests larger than `i32::MAX` are split into equal contiguous chunks before count conversion; the
count and both chunks remain live for each call. Empty slices make no FFI call.

## vDSP polynomial reductions

The regression wrapper calls `vDSP_vsqD` and `vDSP_dotprD` with unit strides and the SDK's
`vDSP_Length`/`vDSP_Stride` ABI widths (`usize`/`isize` on the supported 64-bit targets). Predictors,
responses, basis, and squared-basis scratch must have exactly the same length. The mutable scratch
is disjoint from all three immutable inputs; scalar results and every slice remain live for each
call. Empty slices make no FFI call. The safe runtime boundary validates the same contract before
choosing either scalar or Accelerate execution.

The module links only on macOS; other targets return a typed unavailable-platform error and never
substitute an implementation. Numerical evidence is separate from memory safety: vForce is checked
against Rust `f64::exp` over the declared pricing exponent domain, and vDSP reductions are checked
against compensated large-population sums under a fixed forward-error bound. The production
regression is independently checked against an SVD oracle. Every research run records its selected
backend because exact bits are not promised across implementations.
