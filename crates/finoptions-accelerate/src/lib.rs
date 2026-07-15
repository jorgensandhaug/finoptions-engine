//! Small audited safe wrappers around the Apple Accelerate ABI.

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
        pub(super) fn vDSP_dotprD(
            first: *const f64,
            first_stride: isize,
            second: *const f64,
            second_stride: isize,
            result: *mut f64,
            count: usize,
        );
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

    pub(super) fn polynomial_moments_f64(
        predictors: &[f64],
        responses: &[f64],
        basis: &[f64],
        squared_basis: &mut [f64],
    ) -> (f64, f64, f64) {
        if predictors.is_empty() {
            return (0.0, 0.0, 0.0);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
