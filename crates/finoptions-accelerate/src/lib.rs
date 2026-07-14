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

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod ffi {
    const MAX_VECTOR_LENGTH: usize = 2_147_483_647;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub(super) fn vvexp(output: *mut f64, input: *const f64, count: *const i32);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
