//! Small audited safe wrappers around Apple platform vector math.

use thiserror::Error;

/// A safe wrapper request violates the exact Accelerate buffer contract.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AccelerateError {
    /// Input and output vectors have different element counts.
    #[error("Accelerate input/output length mismatch: input {input}, output {output}")]
    LengthMismatch {
        /// Input elements.
        input: usize,
        /// Output elements.
        output: usize,
    },

    /// Apple Accelerate is not present on the compilation target.
    #[cfg(not(target_os = "macos"))]
    #[error("Apple Accelerate is unavailable on this platform")]
    PlatformUnavailable,
}

/// A safe multi-vector Accelerate request violates its exact operand contract.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VectorKernelError {
    /// One operand has a different element count.
    #[error("Accelerate {operation} operand {operand} has length {observed}; expected {expected}")]
    OperandLengthMismatch {
        /// Stable kernel identifier.
        operation: &'static str,
        /// Stable operand identifier.
        operand: &'static str,
        /// Required element count.
        expected: usize,
        /// Observed element count.
        observed: usize,
    },

    /// Apple Accelerate is not present on the compilation target.
    #[cfg(not(target_os = "macos"))]
    #[error("Apple Accelerate is unavailable on this platform")]
    PlatformUnavailable,
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod ffi {
    const MAX_VECTOR_LENGTH: usize = 2_147_483_647;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub(super) fn vvexp(output: *mut f64, input: *const f64, count: *const i32);
        #[cfg(not(target_arch = "aarch64"))]
        pub(super) fn vDSP_dotprD(
            first: *const f64,
            first_stride: isize,
            second: *const f64,
            second_stride: isize,
            result: *mut f64,
            count: usize,
        );
        #[cfg(not(target_arch = "aarch64"))]
        pub(super) fn vDSP_vsqD(
            input: *const f64,
            input_stride: isize,
            output: *mut f64,
            output_stride: isize,
            count: usize,
        );
    }

    pub(super) fn exp_f64(input: &[f64], output: &mut [f64]) {
        for (input_chunk, output_chunk) in input
            .chunks(MAX_VECTOR_LENGTH)
            .zip(output.chunks_mut(MAX_VECTOR_LENGTH))
        {
            let Ok(count) = i32::try_from(input_chunk.len()) else {
                unreachable!("chunks are bounded to Accelerate's signed 32-bit count")
            };
            // SAFETY: safe Rust guarantees that the immutable input and mutable output slices do
            // not overlap. Both chunks are contiguous, live for the call, and have exactly
            // `count` f64 elements. The count pointer is also live for the complete call.
            unsafe {
                vvexp(
                    output_chunk.as_mut_ptr(),
                    input_chunk.as_ptr(),
                    &raw const count,
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn fused_neon_polynomial_moments_f64_impl(
        predictors: &[f64],
        responses: &[f64],
        basis: &[f64],
        squared_basis_output: &mut [f64],
    ) -> (f64, f64, f64) {
        use core::arch::aarch64::{
            vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64, vmulq_f64, vst1q_f64,
        };

        debug_assert_eq!(predictors.len(), responses.len());
        debug_assert_eq!(predictors.len(), basis.len());
        debug_assert_eq!(predictors.len(), squared_basis_output.len());
        let mut norm = [vdupq_n_f64(0.0); 4];
        let mut projection = [vdupq_n_f64(0.0); 4];
        let mut weighted_predictor = [vdupq_n_f64(0.0); 4];
        let mut index = 0;
        while index + 8 <= basis.len() {
            for accumulator in 0..4 {
                let offset = index + 2 * accumulator;
                // SAFETY: the safe public wrapper proved all four slices have one common length.
                // The eight-element loop bound therefore contains each two-lane load, and Rust's
                // slice borrowing proves the mutable output cannot alias any immutable input.
                let (predictor, response, basis_value) = unsafe {
                    (
                        vld1q_f64(predictors.as_ptr().add(offset)),
                        vld1q_f64(responses.as_ptr().add(offset)),
                        vld1q_f64(basis.as_ptr().add(offset)),
                    )
                };
                let squared_basis = vmulq_f64(basis_value, basis_value);
                // SAFETY: the same common-length proof and loop bound contain this two-lane store.
                unsafe {
                    vst1q_f64(squared_basis_output.as_mut_ptr().add(offset), squared_basis);
                }
                norm[accumulator] = vfmaq_f64(norm[accumulator], basis_value, basis_value);
                projection[accumulator] = vfmaq_f64(projection[accumulator], response, basis_value);
                weighted_predictor[accumulator] =
                    vfmaq_f64(weighted_predictor[accumulator], predictor, squared_basis);
            }
            index += 8;
        }
        let mut norm = vaddq_f64(vaddq_f64(norm[0], norm[1]), vaddq_f64(norm[2], norm[3]));
        let mut projection = vaddq_f64(
            vaddq_f64(projection[0], projection[1]),
            vaddq_f64(projection[2], projection[3]),
        );
        let mut weighted_predictor = vaddq_f64(
            vaddq_f64(weighted_predictor[0], weighted_predictor[1]),
            vaddq_f64(weighted_predictor[2], weighted_predictor[3]),
        );
        while index + 2 <= basis.len() {
            // SAFETY: the common-length proof and two-element loop bound contain every load.
            let (predictor, response, basis_value) = unsafe {
                (
                    vld1q_f64(predictors.as_ptr().add(index)),
                    vld1q_f64(responses.as_ptr().add(index)),
                    vld1q_f64(basis.as_ptr().add(index)),
                )
            };
            let squared_basis = vmulq_f64(basis_value, basis_value);
            // SAFETY: the same proof contains this two-lane output store.
            unsafe {
                vst1q_f64(squared_basis_output.as_mut_ptr().add(index), squared_basis);
            }
            norm = vfmaq_f64(norm, basis_value, basis_value);
            projection = vfmaq_f64(projection, response, basis_value);
            weighted_predictor = vfmaq_f64(weighted_predictor, predictor, squared_basis);
            index += 2;
        }
        let mut norm = vaddvq_f64(norm);
        let mut projection = vaddvq_f64(projection);
        let mut weighted_predictor = vaddvq_f64(weighted_predictor);
        while index < basis.len() {
            let basis_value = basis[index];
            let squared_basis = basis_value * basis_value;
            squared_basis_output[index] = squared_basis;
            norm += squared_basis;
            projection = responses[index].mul_add(basis_value, projection);
            weighted_predictor = predictors[index].mul_add(squared_basis, weighted_predictor);
            index += 1;
        }
        (norm, projection, weighted_predictor)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fused_neon_polynomial_recurrence_moments_f64_impl(
        predictors: &[f64],
        responses: &[f64],
        previous_basis: &[f64],
        current_basis: &[f64],
        next_basis_output: &mut [f64],
        alpha: f64,
        beta: f64,
    ) -> (f64, f64, f64) {
        use core::arch::aarch64::{
            vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64, vmulq_f64, vnegq_f64,
            vst1q_f64, vsubq_f64,
        };

        debug_assert_eq!(predictors.len(), responses.len());
        debug_assert_eq!(predictors.len(), previous_basis.len());
        debug_assert_eq!(predictors.len(), current_basis.len());
        debug_assert_eq!(predictors.len(), next_basis_output.len());
        let alpha_vector = vdupq_n_f64(alpha);
        let beta_vector = vdupq_n_f64(beta);
        let mut norm = [vdupq_n_f64(0.0); 4];
        let mut projection = [vdupq_n_f64(0.0); 4];
        let mut weighted_predictor = [vdupq_n_f64(0.0); 4];
        let mut index = 0;
        while index + 8 <= predictors.len() {
            for accumulator in 0..4 {
                let offset = index + 2 * accumulator;
                // SAFETY: the safe wrapper proved a common length for every slice, so the
                // eight-element loop bound contains all two-lane loads and the output store.
                let (predictor, response, previous, current) = unsafe {
                    (
                        vld1q_f64(predictors.as_ptr().add(offset)),
                        vld1q_f64(responses.as_ptr().add(offset)),
                        vld1q_f64(previous_basis.as_ptr().add(offset)),
                        vld1q_f64(current_basis.as_ptr().add(offset)),
                    )
                };
                let centered = vsubq_f64(predictor, alpha_vector);
                let negative_beta_previous = vnegq_f64(vmulq_f64(beta_vector, previous));
                let next = vfmaq_f64(negative_beta_previous, centered, current);
                // SAFETY: the same common-length and disjoint-borrow proof contains this store.
                unsafe {
                    vst1q_f64(next_basis_output.as_mut_ptr().add(offset), next);
                }
                let squared = vmulq_f64(next, next);
                norm[accumulator] = vfmaq_f64(norm[accumulator], next, next);
                projection[accumulator] = vfmaq_f64(projection[accumulator], response, next);
                weighted_predictor[accumulator] =
                    vfmaq_f64(weighted_predictor[accumulator], predictor, squared);
            }
            index += 8;
        }
        let mut norm = vaddq_f64(vaddq_f64(norm[0], norm[1]), vaddq_f64(norm[2], norm[3]));
        let mut projection = vaddq_f64(
            vaddq_f64(projection[0], projection[1]),
            vaddq_f64(projection[2], projection[3]),
        );
        let mut weighted_predictor = vaddq_f64(
            vaddq_f64(weighted_predictor[0], weighted_predictor[1]),
            vaddq_f64(weighted_predictor[2], weighted_predictor[3]),
        );
        while index + 2 <= predictors.len() {
            // SAFETY: the common-length proof and two-element bound contain every access.
            let (predictor, response, previous, current) = unsafe {
                (
                    vld1q_f64(predictors.as_ptr().add(index)),
                    vld1q_f64(responses.as_ptr().add(index)),
                    vld1q_f64(previous_basis.as_ptr().add(index)),
                    vld1q_f64(current_basis.as_ptr().add(index)),
                )
            };
            let centered = vsubq_f64(predictor, alpha_vector);
            let negative_beta_previous = vnegq_f64(vmulq_f64(beta_vector, previous));
            let next = vfmaq_f64(negative_beta_previous, centered, current);
            // SAFETY: the same proof contains this two-lane output store.
            unsafe {
                vst1q_f64(next_basis_output.as_mut_ptr().add(index), next);
            }
            let squared = vmulq_f64(next, next);
            norm = vfmaq_f64(norm, next, next);
            projection = vfmaq_f64(projection, response, next);
            weighted_predictor = vfmaq_f64(weighted_predictor, predictor, squared);
            index += 2;
        }
        let mut norm = vaddvq_f64(norm);
        let mut projection = vaddvq_f64(projection);
        let mut weighted_predictor = vaddvq_f64(weighted_predictor);
        while index < predictors.len() {
            let next = (predictors[index] - alpha)
                .mul_add(current_basis[index], -beta * previous_basis[index]);
            next_basis_output[index] = next;
            let squared = next * next;
            norm += squared;
            projection = responses[index].mul_add(next, projection);
            weighted_predictor = predictors[index].mul_add(squared, weighted_predictor);
            index += 1;
        }
        (norm, projection, weighted_predictor)
    }

    #[cfg(target_arch = "aarch64")]
    fn fused_neon_polynomial_moments_f64(
        predictors: &[f64],
        responses: &[f64],
        basis: &[f64],
        squared_basis_output: &mut [f64],
    ) -> (f64, f64, f64) {
        // SAFETY: NEON is part of the baseline AArch64 ISA on macOS. The safe public wrapper has
        // already proved the equal, disjoint slice contract required by the implementation.
        unsafe {
            fused_neon_polynomial_moments_f64_impl(
                predictors,
                responses,
                basis,
                squared_basis_output,
            )
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    fn fused_neon_polynomial_recurrence_moments_f64(
        predictors: &[f64],
        responses: &[f64],
        previous_basis: &[f64],
        current_basis: &[f64],
        next_basis_output: &mut [f64],
        alpha: f64,
        beta: f64,
    ) -> (f64, f64, f64) {
        // SAFETY: NEON is baseline AArch64 on macOS. The public wrapper proved equal lengths and
        // Rust borrowing proves that the mutable output cannot alias an immutable operand.
        unsafe {
            fused_neon_polynomial_recurrence_moments_f64_impl(
                predictors,
                responses,
                previous_basis,
                current_basis,
                next_basis_output,
                alpha,
                beta,
            )
        }
    }

    pub(super) fn polynomial_moments_f64(
        predictors: &[f64],
        responses: &[f64],
        basis: &[f64],
        squared_basis: &mut [f64],
    ) -> (f64, f64, f64) {
        if predictors.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        #[cfg(target_arch = "aarch64")]
        {
            fused_neon_polynomial_moments_f64(predictors, responses, basis, squared_basis)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut norm = 0.0;
            let mut projection = 0.0;
            let mut weighted_predictor = 0.0;
            // SAFETY: validation in the safe wrapper proves that all four contiguous slices contain
            // exactly `count` f64 elements. The mutable squared-basis output is disjoint from all
            // immutable inputs, every stride is one, and each scalar result lives for its call.
            unsafe {
                vDSP_vsqD(
                    basis.as_ptr(),
                    1,
                    squared_basis.as_mut_ptr(),
                    1,
                    basis.len(),
                );
                vDSP_dotprD(
                    basis.as_ptr(),
                    1,
                    basis.as_ptr(),
                    1,
                    &raw mut norm,
                    basis.len(),
                );
                vDSP_dotprD(
                    responses.as_ptr(),
                    1,
                    basis.as_ptr(),
                    1,
                    &raw mut projection,
                    basis.len(),
                );
                vDSP_dotprD(
                    predictors.as_ptr(),
                    1,
                    squared_basis.as_ptr(),
                    1,
                    &raw mut weighted_predictor,
                    basis.len(),
                );
            }
            (norm, projection, weighted_predictor)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn polynomial_recurrence_moments_f64(
        predictors: &[f64],
        responses: &[f64],
        previous_basis: &[f64],
        current_basis: &[f64],
        next_basis_output: &mut [f64],
        alpha: f64,
        beta: f64,
    ) -> (f64, f64, f64) {
        if predictors.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        #[cfg(target_arch = "aarch64")]
        {
            fused_neon_polynomial_recurrence_moments_f64(
                predictors,
                responses,
                previous_basis,
                current_basis,
                next_basis_output,
                alpha,
                beta,
            )
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scalar_polynomial_recurrence_moments_f64(
                predictors,
                responses,
                previous_basis,
                current_basis,
                next_basis_output,
                alpha,
                beta,
            )
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    fn scalar_polynomial_recurrence_moments_f64(
        predictors: &[f64],
        responses: &[f64],
        previous_basis: &[f64],
        current_basis: &[f64],
        next_basis_output: &mut [f64],
        alpha: f64,
        beta: f64,
    ) -> (f64, f64, f64) {
        let mut norm = 0.0;
        let mut projection = 0.0;
        let mut weighted_predictor = 0.0;
        for ((((predictor, response), previous), current), next) in predictors
            .iter()
            .zip(responses)
            .zip(previous_basis)
            .zip(current_basis)
            .zip(next_basis_output)
        {
            *next = (*predictor - alpha).mul_add(*current, -beta * *previous);
            let squared = *next * *next;
            norm += squared;
            projection = response.mul_add(*next, projection);
            weighted_predictor = predictor.mul_add(squared, weighted_predictor);
        }
        (norm, projection, weighted_predictor)
    }
}

/// Computes the double-precision natural exponential using Apple vForce `vvexp`.
///
/// # Errors
///
/// Returns [`AccelerateError::LengthMismatch`] without writing output when slice lengths differ.
pub fn exp_f64(input: &[f64], output: &mut [f64]) -> Result<(), AccelerateError> {
    if input.len() != output.len() {
        return Err(AccelerateError::LengthMismatch {
            input: input.len(),
            output: output.len(),
        });
    }
    #[cfg(target_os = "macos")]
    {
        ffi::exp_f64(input, output);
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(AccelerateError::PlatformUnavailable)
    }
}

/// Computes the three reductions required by one empirical orthogonal-polynomial basis vector.
///
/// The result is `(sum(basis^2), sum(responses * basis),
/// sum(predictors * basis^2))`. `squared_basis` is explicit caller-owned scratch and is overwritten.
///
/// # Errors
///
/// Returns a typed error before writing scratch unless every operand has the same length.
pub fn polynomial_moments_f64(
    predictors: &[f64],
    responses: &[f64],
    basis: &[f64],
    squared_basis: &mut [f64],
) -> Result<(f64, f64, f64), VectorKernelError> {
    const OPERATION: &str = "polynomial_moments_f64";
    let expected = predictors.len();
    for (operand, observed) in [
        ("responses", responses.len()),
        ("basis", basis.len()),
        ("squared_basis", squared_basis.len()),
    ] {
        if observed != expected {
            return Err(VectorKernelError::OperandLengthMismatch {
                operation: OPERATION,
                operand,
                expected,
                observed,
            });
        }
    }
    #[cfg(target_os = "macos")]
    {
        Ok(ffi::polynomial_moments_f64(
            predictors,
            responses,
            basis,
            squared_basis,
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(VectorKernelError::PlatformUnavailable)
    }
}

/// Constructs the next empirical orthogonal basis vector and computes its three moments in one
/// pass. The recurrence is `next = (predictor - alpha) * current - beta * previous`.
///
/// # Errors
///
/// Returns a typed error before writing output unless every vector has the same length.
#[allow(clippy::too_many_arguments)]
pub fn polynomial_recurrence_moments_f64(
    predictors: &[f64],
    responses: &[f64],
    previous_basis: &[f64],
    current_basis: &[f64],
    next_basis_output: &mut [f64],
    alpha: f64,
    beta: f64,
) -> Result<(f64, f64, f64), VectorKernelError> {
    const OPERATION: &str = "polynomial_recurrence_moments_f64";
    let expected = predictors.len();
    for (operand, observed) in [
        ("responses", responses.len()),
        ("previous_basis", previous_basis.len()),
        ("current_basis", current_basis.len()),
        ("next_basis_output", next_basis_output.len()),
    ] {
        if observed != expected {
            return Err(VectorKernelError::OperandLengthMismatch {
                operation: OPERATION,
                operand,
                expected,
                observed,
            });
        }
    }
    #[cfg(target_os = "macos")]
    {
        Ok(ffi::polynomial_recurrence_moments_f64(
            predictors,
            responses,
            previous_basis,
            current_basis,
            next_basis_output,
            alpha,
            beta,
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (alpha, beta);
        Err(VectorKernelError::PlatformUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    fn compensated_sum(terms: impl IntoIterator<Item = f64>) -> (f64, f64) {
        let mut sum = 0.0;
        let mut correction = 0.0;
        let mut absolute_sum = 0.0;
        for term in terms {
            absolute_sum += term.abs();
            let next = sum + term;
            correction += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
        }
        (sum + correction, absolute_sum)
    }

    #[cfg(target_os = "macos")]
    fn assert_reduction_within_declared_bound(observed: f64, expected: f64, absolute_sum: f64) {
        let tolerance = 128.0 * f64::EPSILON * absolute_sum.max(f64::MIN_POSITIVE);
        assert!(
            (observed - expected).abs() <= tolerance,
            "observed={observed:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rejects_length_mismatch_before_writing() {
        let mut output = [7.0; 1];
        assert_eq!(
            exp_f64(&[0.0, 1.0], &mut output),
            Err(AccelerateError::LengthMismatch {
                input: 2,
                output: 1,
            })
        );
        assert_eq!(output[0].to_bits(), 7.0_f64.to_bits());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn agrees_with_rust_std_f64_within_one_ulp_on_the_pricing_domain() {
        let input = (0..100_001)
            .map(|index| -700.0 + 1_400.0 * f64::from(index) / 100_000.0)
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        exp_f64(&input, &mut output).unwrap();
        for (&exponent, &observed) in input.iter().zip(&output) {
            let expected = exponent.exp();
            assert!(expected.to_bits().abs_diff(observed.to_bits()) <= 1);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn polynomial_moments_match_the_declared_reductions() {
        let predictors = [0.5, 0.75, 1.0, 1.25, 1.5];
        let responses = [2.0, 3.0, 5.0, 7.0, 11.0];
        let basis = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut squared_basis = [f64::NAN; 5];
        let observed =
            polynomial_moments_f64(&predictors, &responses, &basis, &mut squared_basis).unwrap();
        let expected = (
            basis.iter().map(|value| value * value).sum::<f64>(),
            responses
                .iter()
                .zip(basis)
                .map(|(response, basis)| response * basis)
                .sum::<f64>(),
            predictors
                .iter()
                .zip(basis)
                .map(|(predictor, basis)| predictor * basis * basis)
                .sum::<f64>(),
        );
        assert_eq!(observed, expected);
        assert_eq!(
            squared_basis.map(f64::to_bits),
            [4.0, 1.0, 0.0, 1.0, 4.0].map(f64::to_bits)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn polynomial_moments_match_a_compensated_large_population_oracle() {
        let observations = 65_536_u32;
        let predictors = (0..observations)
            .map(|index| {
                let unit = f64::from(index) / f64::from(observations);
                0.82 + 0.56 * unit + 0.015 * (17.0 * unit).sin()
            })
            .collect::<Vec<_>>();
        let responses = predictors
            .iter()
            .map(|&predictor| {
                2.0 - 0.7 * predictor + 0.35 * predictor.powi(2) - 0.08 * predictor.powi(3)
                    + 0.015 * predictor.powi(4)
                    + 0.002 * (31.0 * predictor).cos()
            })
            .collect::<Vec<_>>();
        let basis = predictors
            .iter()
            .map(|&predictor| {
                let z = (predictor - 1.1) / 0.2;
                0.05 + z * (-0.2 + z * (-0.1 + z))
            })
            .collect::<Vec<_>>();
        let mut squared_basis = vec![f64::NAN; basis.len()];
        let observed =
            polynomial_moments_f64(&predictors, &responses, &basis, &mut squared_basis).unwrap();
        let expected = [
            compensated_sum(basis.iter().map(|value| value * value)),
            compensated_sum(
                responses
                    .iter()
                    .zip(&basis)
                    .map(|(response, basis)| response * basis),
            ),
            compensated_sum(
                predictors
                    .iter()
                    .zip(&basis)
                    .map(|(predictor, basis)| predictor * basis * basis),
            ),
        ];
        for (observed, (expected, absolute_sum)) in [observed.0, observed.1, observed.2]
            .into_iter()
            .zip(expected)
        {
            assert_reduction_within_declared_bound(observed, expected, absolute_sum);
        }
        assert!(
            squared_basis
                .iter()
                .zip(&basis)
                .all(|(observed, basis)| observed.to_bits() == (basis * basis).to_bits())
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn polynomial_moments_match_compensated_oracles_across_vector_block_residues() {
        let lengths = [
            0_usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 127, 128, 129, 499,
            500, 501, 1_023, 1_024, 1_025,
        ];
        for length in lengths {
            let predictors = (0..length)
                .map(|index| {
                    let index = f64::from(u32::try_from(index).unwrap());
                    0.73 + 0.000_31 * index + 0.017 * (0.13 * index).sin()
                })
                .collect::<Vec<_>>();
            let responses = predictors
                .iter()
                .enumerate()
                .map(|(index, &predictor)| {
                    let index = f64::from(u32::try_from(index).unwrap());
                    1.7 - 0.43 * predictor
                        + 0.09 * predictor * predictor
                        + 0.004 * (0.19 * index).cos()
                })
                .collect::<Vec<_>>();
            let basis = predictors
                .iter()
                .enumerate()
                .map(|(index, &predictor)| {
                    let index = f64::from(u32::try_from(index).unwrap());
                    let centered = predictor - 0.91;
                    0.03 + centered * (-0.2 + centered * 0.7) + 0.002 * (0.07 * index).sin()
                })
                .collect::<Vec<_>>();
            let mut squared_basis = vec![f64::NAN; length];
            let observed =
                polynomial_moments_f64(&predictors, &responses, &basis, &mut squared_basis)
                    .unwrap();
            let expected = [
                compensated_sum(basis.iter().map(|value| value * value)),
                compensated_sum(
                    responses
                        .iter()
                        .zip(&basis)
                        .map(|(response, basis)| response * basis),
                ),
                compensated_sum(
                    predictors
                        .iter()
                        .zip(&basis)
                        .map(|(predictor, basis)| predictor * basis * basis),
                ),
            ];
            for (observed, (expected, absolute_sum)) in [observed.0, observed.1, observed.2]
                .into_iter()
                .zip(expected)
            {
                assert_reduction_within_declared_bound(observed, expected, absolute_sum);
            }
            assert!(
                squared_basis
                    .iter()
                    .zip(&basis)
                    .all(|(observed, basis)| observed.to_bits() == (basis * basis).to_bits()),
                "squared-basis scratch differs at population length {length}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn polynomial_recurrence_moments_match_scalar_recurrence_and_compensated_oracles() {
        let lengths = [
            0_usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 499, 500, 501, 1_023, 1_024,
            1_025,
        ];
        let alpha = -0.17;
        let beta = 0.23;
        for length in lengths {
            let predictors = (0..length)
                .map(|index| {
                    let index = f64::from(u32::try_from(index).unwrap());
                    -1.4 + 0.003_1 * index + 0.07 * (0.11 * index).sin()
                })
                .collect::<Vec<_>>();
            let responses = predictors
                .iter()
                .enumerate()
                .map(|(index, predictor)| {
                    let index = f64::from(u32::try_from(index).unwrap());
                    1.3 + 0.41 * predictor - 0.08 * predictor * predictor
                        + 0.003 * (0.07 * index).cos()
                })
                .collect::<Vec<_>>();
            let previous = predictors
                .iter()
                .map(|predictor| 0.3 - 0.2 * predictor)
                .collect::<Vec<_>>();
            let current = predictors
                .iter()
                .map(|predictor| 0.7 + predictor * (-0.1 + 0.04 * predictor))
                .collect::<Vec<_>>();
            let expected_next = predictors
                .iter()
                .zip(&previous)
                .zip(&current)
                .map(|((predictor, previous), current)| {
                    (*predictor - alpha).mul_add(*current, -beta * *previous)
                })
                .collect::<Vec<_>>();
            let mut observed_next = vec![f64::NAN; length];
            let observed = polynomial_recurrence_moments_f64(
                &predictors,
                &responses,
                &previous,
                &current,
                &mut observed_next,
                alpha,
                beta,
            )
            .unwrap();
            assert!(
                observed_next
                    .iter()
                    .zip(&expected_next)
                    .all(|(observed, expected)| observed.to_bits() == expected.to_bits()),
                "recurrence differs at population length {length}"
            );
            let mut squared = vec![f64::NAN; length];
            let separate =
                polynomial_moments_f64(&predictors, &responses, &expected_next, &mut squared)
                    .unwrap();
            assert_eq!(
                [observed.0, observed.1, observed.2].map(f64::to_bits),
                [separate.0, separate.1, separate.2].map(f64::to_bits),
                "fused moments differ from the accepted separate kernel at length {length}"
            );
            let expected = [
                compensated_sum(expected_next.iter().map(|value| value * value)),
                compensated_sum(
                    responses
                        .iter()
                        .zip(&expected_next)
                        .map(|(response, basis)| response * basis),
                ),
                compensated_sum(
                    predictors
                        .iter()
                        .zip(&expected_next)
                        .map(|(predictor, basis)| predictor * basis * basis),
                ),
            ];
            for (observed, (expected, absolute_sum)) in [observed.0, observed.1, observed.2]
                .into_iter()
                .zip(expected)
            {
                assert_reduction_within_declared_bound(observed, expected, absolute_sum);
            }
        }
    }

    #[test]
    fn polynomial_moments_reject_misaligned_operands_without_writing_scratch() {
        let mut scratch = [7.0; 2];
        assert_eq!(
            polynomial_moments_f64(&[1.0, 2.0], &[3.0], &[4.0, 5.0], &mut scratch),
            Err(VectorKernelError::OperandLengthMismatch {
                operation: "polynomial_moments_f64",
                operand: "responses",
                expected: 2,
                observed: 1,
            })
        );
        assert_eq!(scratch.map(f64::to_bits), [7.0, 7.0].map(f64::to_bits));

        assert_eq!(
            polynomial_recurrence_moments_f64(
                &[1.0, 2.0],
                &[3.0, 4.0],
                &[0.0],
                &[1.0, 1.0],
                &mut scratch,
                0.0,
                0.0,
            ),
            Err(VectorKernelError::OperandLengthMismatch {
                operation: "polynomial_recurrence_moments_f64",
                operand: "previous_basis",
                expected: 2,
                observed: 1,
            })
        );
        assert_eq!(scratch.map(f64::to_bits), [7.0, 7.0].map(f64::to_bits));
    }
}
