//! Fallible bounded ZIGNOR standard-normal sampling over a path-local Philox stream.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use super::normal_tables::{ZIG_NORM_F, ZIG_NORM_R, ZIG_NORM_X};
use super::philox::PhiloxPathStream;

const MAX_REJECTION_ITERATIONS: usize = 1_048_576;
const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
const EXPONENT_0: u64 = 1023_u64 << 52;
const EXPONENT_1: u64 = 1024_u64 << 52;

/// Explicit standard-normal generation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NormalError {
    BudgetExhausted,
    RejectionLimit,
}

impl Display for NormalError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExhausted => formatter.write_str("standard-normal path budget exhausted"),
            Self::RejectionLimit => formatter.write_str("standard-normal rejection limit reached"),
        }
    }
}

impl Error for NormalError {}

#[inline]
fn symmetric_unit(bits: u64) -> f64 {
    f64::from_bits(((bits >> 12) & FRACTION_MASK) | EXPONENT_1) - 3.0
}

#[inline]
fn open_unit(stream: &mut PhiloxPathStream) -> f64 {
    let bits = stream.next_u64();
    let fraction = (bits >> 12) & FRACTION_MASK;
    f64::from_bits(fraction | EXPONENT_0) - (1.0 - f64::EPSILON / 2.0)
}

#[inline]
#[allow(clippy::cast_precision_loss)]
fn standard_unit(stream: &mut PhiloxPathStream) -> f64 {
    (stream.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn sample_tail(stream: &mut PhiloxPathStream, signed_unit: f64) -> Result<f64, NormalError> {
    for _ in 0..MAX_REJECTION_ITERATIONS {
        let x = open_unit(stream).ln() / ZIG_NORM_R;
        let y = open_unit(stream).ln();
        if -2.0 * y >= x * x {
            return Ok(if signed_unit < 0.0 {
                x - ZIG_NORM_R
            } else {
                ZIG_NORM_R - x
            });
        }
    }
    Err(NormalError::RejectionLimit)
}

#[cold]
fn sample_rejected(
    stream: &mut PhiloxPathStream,
    mut index: usize,
    mut unit: f64,
    mut value: f64,
) -> Result<f64, NormalError> {
    for _ in 0..MAX_REJECTION_ITERATIONS {
        if index == 0 {
            return sample_tail(stream, unit);
        }
        let ordinate = ZIG_NORM_F[index + 1]
            + (ZIG_NORM_F[index] - ZIG_NORM_F[index + 1]) * standard_unit(stream);
        if ordinate < (-0.5 * value * value).exp() {
            return Ok(value);
        }

        let bits = stream.next_u64();
        index = usize::from(u8::try_from(bits & 0xff).unwrap());
        unit = symmetric_unit(bits);
        value = unit * ZIG_NORM_X[index];
        if value.abs() < ZIG_NORM_X[index + 1] {
            return Ok(value);
        }
    }
    Err(NormalError::RejectionLimit)
}

/// Samples one `N(0, 1)` variate or returns an explicit bounded-generation failure.
#[allow(clippy::inline_always)] // Measured hot-path requirement; rejected draws stay cold.
#[inline(always)]
pub(crate) fn sample_standard_normal(stream: &mut PhiloxPathStream) -> Result<f64, NormalError> {
    if !stream.begin_normal() {
        return Err(NormalError::BudgetExhausted);
    }
    let bits = stream.next_u64();
    let index = usize::from(u8::try_from(bits & 0xff).unwrap());
    let unit = symmetric_unit(bits);
    let value = unit * ZIG_NORM_X[index];
    if value.abs() < ZIG_NORM_X[index + 1] {
        Ok(value)
    } else {
        sample_rejected(stream, index, unit, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_the_standard_normal_transform_sequence() {
        let mut stream = PhiloxPathStream::new(0x01d6_328f_7a4c_915b, 0x1729_0042, 37, 16);
        let values = (0..16)
            .map(|_| sample_standard_normal(&mut stream).unwrap().to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            [
                0xbff6_05fe_3b0d_63ee,
                0xbff9_dadd_dd00_d869,
                0xbfd7_f54d_70f5_61a5,
                0xbfcb_7694_d4ed_0321,
                0x3ff2_3a33_29d7_b8a1,
                0x3fa2_b88c_2817_f076,
                0x3fec_2765_8b88_49ad,
                0x3fe0_b765_1e54_0e58,
                0xbff1_da56_2b64_c24a,
                0xbfe5_ac00_7989_d22f,
                0x3fc3_bc81_d6f6_f3ef,
                0xbff5_eb94_19f3_fe3d,
                0xbfe0_d91e_c575_9fd2,
                0x3fe6_2751_8beb_49fd,
                0x3ff4_1d4f_92ac_1c2e,
                0x3fb0_c52e_9f80_4eb5,
            ]
        );
    }

    #[test]
    fn fails_when_the_declared_normal_budget_is_exhausted() {
        let mut stream = PhiloxPathStream::new(7, 11, 13, 1);
        sample_standard_normal(&mut stream).unwrap();
        assert_eq!(
            sample_standard_normal(&mut stream).unwrap_err(),
            NormalError::BudgetExhausted
        );
    }
}
