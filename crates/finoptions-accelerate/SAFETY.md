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

## Polynomial reductions

Predictors, responses, basis, and squared-basis scratch must have exactly the same length. The safe
public wrapper proves that contract before dispatch, and Rust's borrowing rules keep the mutable
scratch disjoint from all three immutable inputs. Empty slices return the three zero reductions
without entering a native kernel. The runtime validates the same contract before choosing either
the portable or explicitly requested Apple backend.

The fused recurrence kernel similarly requires equal predictor, response, previous-basis,
current-basis, and mutable next-basis lengths. The next-basis output is a distinct mutable borrow,
so it cannot alias any input. Each lane computes
`(predictor - alpha).mul_add(current, -beta * previous)`, stores that value, and accumulates its
norm, response projection, and predictor-weighted squared norm. The same common-length proof bounds
every access; an empty request writes nothing and returns three zero reductions.

On AArch64 macOS, NEON is part of the baseline ISA. The implementation loads and stores two `f64`
lanes only under a remaining-length proof, processes eight elements through four independent
accumulators, then handles a two-element vector tail and at most one scalar element. Pointer
addition is therefore bounded by the common slice length. Horizontal reduction occurs only after
all vector loads and stores complete. The `#[target_feature(enable = "neon")]` function is called
only from the AArch64-specific safe wrapper. Both the basis-only and recurrence/moment kernels use
this block/tail structure.

On non-AArch64 macOS, the wrapper retains `vDSP_vsqD` and `vDSP_dotprD` with unit strides and the
SDK's `vDSP_Length`/`vDSP_Stride` ABI widths (`usize`/`isize` on supported 64-bit targets). Scalar
results and every slice remain live for each call. The benchmark also retains the former vDSP
implementation as an explicit measured reference under the same shape and aliasing proof; it is
not a production fallback.

The module links only on macOS; other targets return a typed unavailable-platform error and never
substitute an implementation. Numerical evidence is separate from memory safety: vForce is checked
against Rust `f64::exp` over the declared pricing exponent domain. Polynomial reductions are checked
against compensated sums at every vector-block residue around small, 500-row, and power-of-two
boundaries, plus a 65,536-row population, under a fixed forward-error bound. Squared-basis scratch
is checked bitwise. The production regression is independently checked against an SVD oracle.
Every research run records its selected backend because exact bits are not promised across
implementations.
