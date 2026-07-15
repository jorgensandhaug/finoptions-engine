//! Independent Haug-Haug-Lewis one-cash-dividend call reference.

use std::f64::consts::{PI, SQRT_2};

use finoptions_core::DividendPolicy;
use thiserror::Error;

const NORMAL_INTEGRATION_LIMIT: f64 = 12.0;
const ABSOLUTE_TOLERANCE: f64 = 2.0e-11;
const RELATIVE_TOLERANCE: f64 = 2.0e-11;
const MAXIMUM_RECURSION_DEPTH: u32 = 30;

/// A non-production one-dividend reference request is invalid or did not converge.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum OneDividendReferenceError {
    /// A named scalar lies outside the published reference domain.
    #[error("one-dividend reference field {field} is invalid: {observed}")]
    InvalidInput {
        /// Stable field identity.
        field: &'static str,
        /// Rejected scalar.
        observed: f64,
    },

    /// The cash event is not strictly between valuation and expiry.
    #[error(
        "one-dividend reference time must be strictly between zero and maturity {maturity}; observed {dividend_time}"
    )]
    InvalidDividendTime {
        /// Rejected cash-event time.
        dividend_time: f64,
        /// Declared expiry.
        maturity: f64,
    },

    /// The one-pre-dividend-exercise theorem is not admitted at a negative rate.
    #[error("one-dividend American-call reference requires a nonnegative rate; observed {rate}")]
    NegativeRate {
        /// Rejected continuously compounded rate.
        rate: f64,
    },

    /// The independent adaptive integration failed its explicit error budget.
    #[error("one-dividend adaptive integration did not converge within depth {depth}")]
    IntegrationDidNotConverge {
        /// Exhausted recursion depth.
        depth: u32,
    },

    /// Finite inputs produced non-finite arithmetic.
    #[error("one-dividend reference produced a non-finite value")]
    NonFiniteResult,

    /// A future core policy variant has no independent implementation here.
    #[error("one-dividend reference does not implement the requested dividend policy")]
    UnsupportedDividendPolicy,
}

#[derive(Clone, Copy)]
enum SupportedDividendPolicy {
    Liquidator,
    Survivor,
}

/// Independently evaluates the one-cash-dividend American-call formula.
///
/// Under a nonnegative rate and no proportional yield, exercise is considered immediately before
/// the single strictly pre-expiry cash event. Continuation after the explicitly selected cash
/// policy is a Black-Scholes European call. The outer standard-normal expectation is evaluated by
/// adaptive Simpson integration, independently of the production event grid and Gauss-Hermite
/// implementation.
///
/// # Errors
///
/// Returns a typed domain, convergence, or arithmetic error. No input or tolerance is repaired.
#[allow(clippy::too_many_arguments)]
pub fn one_dividend_american_call(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    maturity: f64,
    dividend_time: f64,
    dividend_amount: f64,
    policy: DividendPolicy,
) -> Result<f64, OneDividendReferenceError> {
    validate_inputs(
        spot,
        strike,
        rate,
        volatility,
        maturity,
        dividend_time,
        dividend_amount,
    )?;
    if rate < 0.0 {
        return Err(OneDividendReferenceError::NegativeRate { rate });
    }
    let policy = match policy {
        DividendPolicy::Liquidator => SupportedDividendPolicy::Liquidator,
        DividendPolicy::Survivor => SupportedDividendPolicy::Survivor,
        _ => return Err(OneDividendReferenceError::UnsupportedDividendPolicy),
    };
    let variance = volatility * volatility * dividend_time;
    let log_drift = rate.mul_add(dividend_time, -0.5 * variance);
    let diffusion = volatility * dividend_time.sqrt();
    let remaining = maturity - dividend_time;
    let integrand = |normal: f64| {
        let pre_dividend_spot = spot * diffusion.mul_add(normal, log_drift).exp();
        let post_dividend_spot = post_dividend_spot(pre_dividend_spot, dividend_amount, policy);
        let continuation = european_call(post_dividend_spot, strike, rate, volatility, remaining);
        let exercise = (pre_dividend_spot - strike).max(0.0);
        exercise.max(continuation) * standard_normal_density(normal)
    };
    let mut boundaries = vec![-NORMAL_INTEGRATION_LIMIT, NORMAL_INTEGRATION_LIMIT];
    for stock_level in [dividend_amount, strike, dividend_amount + strike] {
        if stock_level > 0.0 {
            let normal = ((stock_level / spot).ln() - log_drift) / diffusion;
            if (-NORMAL_INTEGRATION_LIMIT..NORMAL_INTEGRATION_LIMIT).contains(&normal) {
                boundaries.push(normal);
            }
        }
    }
    boundaries.sort_unstable_by(f64::total_cmp);
    boundaries.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let segments = u32::try_from(boundaries.len() - 1)
        .map_err(|_| OneDividendReferenceError::NonFiniteResult)?;
    let segment_absolute_tolerance = ABSOLUTE_TOLERANCE / f64::from(segments);
    let mut expectation = 0.0;
    for pair in boundaries.windows(2) {
        expectation += adaptive_simpson(
            &integrand,
            pair[0],
            pair[1],
            segment_absolute_tolerance,
            RELATIVE_TOLERANCE,
            MAXIMUM_RECURSION_DEPTH,
        )?;
    }
    let result = (-rate * dividend_time).exp() * expectation;
    if !result.is_finite() || result < 0.0 {
        return Err(OneDividendReferenceError::NonFiniteResult);
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    maturity: f64,
    dividend_time: f64,
    dividend_amount: f64,
) -> Result<(), OneDividendReferenceError> {
    for (field, observed, strictly_positive) in [
        ("spot", spot, true),
        ("strike", strike, true),
        ("rate", rate, false),
        ("volatility", volatility, true),
        ("maturity", maturity, true),
        ("dividend_amount", dividend_amount, false),
    ] {
        let invalid = !observed.is_finite()
            || (strictly_positive && observed <= 0.0)
            || (!strictly_positive
                && field == "dividend_amount"
                && (observed < 0.0 || (observed == 0.0 && observed.is_sign_negative())));
        if invalid {
            return Err(OneDividendReferenceError::InvalidInput { field, observed });
        }
    }
    if !dividend_time.is_finite() || dividend_time <= 0.0 || dividend_time >= maturity {
        return Err(OneDividendReferenceError::InvalidDividendTime {
            dividend_time,
            maturity,
        });
    }
    Ok(())
}

fn post_dividend_spot(
    pre_dividend_spot: f64,
    dividend_amount: f64,
    policy: SupportedDividendPolicy,
) -> f64 {
    match policy {
        SupportedDividendPolicy::Liquidator => (pre_dividend_spot - dividend_amount).max(0.0),
        SupportedDividendPolicy::Survivor => {
            if pre_dividend_spot > dividend_amount {
                pre_dividend_spot - dividend_amount
            } else {
                pre_dividend_spot
            }
        }
    }
}

fn normal_cdf(value: f64) -> f64 {
    0.5 * libm::erfc(-value / SQRT_2)
}

fn standard_normal_density(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (2.0 * PI).sqrt()
}

fn european_call(spot: f64, strike: f64, rate: f64, volatility: f64, maturity: f64) -> f64 {
    if spot == 0.0 {
        return 0.0;
    }
    let sigma_time = volatility * maturity.sqrt();
    let d1 =
        ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * maturity) / sigma_time;
    let d2 = d1 - sigma_time;
    spot.mul_add(
        normal_cdf(d1),
        -strike * (-rate * maturity).exp() * normal_cdf(d2),
    )
}

#[derive(Clone, Copy)]
struct SimpsonInterval {
    left: f64,
    midpoint: f64,
    right: f64,
    left_value: f64,
    midpoint_value: f64,
    right_value: f64,
    estimate: f64,
}

impl SimpsonInterval {
    fn new(function: &impl Fn(f64) -> f64, left: f64, right: f64) -> Self {
        let midpoint = 0.5 * (left + right);
        let left_value = function(left);
        let midpoint_value = function(midpoint);
        let right_value = function(right);
        let estimate = simpson(left, right, left_value, midpoint_value, right_value);
        Self {
            left,
            midpoint,
            right,
            left_value,
            midpoint_value,
            right_value,
            estimate,
        }
    }
}

fn simpson(left: f64, right: f64, left_value: f64, midpoint_value: f64, right_value: f64) -> f64 {
    (right - left) * (left_value + 4.0 * midpoint_value + right_value) / 6.0
}

fn adaptive_simpson(
    function: &impl Fn(f64) -> f64,
    left: f64,
    right: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    depth: u32,
) -> Result<f64, OneDividendReferenceError> {
    adaptive_simpson_interval(
        function,
        SimpsonInterval::new(function, left, right),
        absolute_tolerance,
        relative_tolerance,
        depth,
    )
}

fn adaptive_simpson_interval(
    function: &impl Fn(f64) -> f64,
    interval: SimpsonInterval,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    depth: u32,
) -> Result<f64, OneDividendReferenceError> {
    let left_midpoint = 0.5 * (interval.left + interval.midpoint);
    let right_midpoint = 0.5 * (interval.midpoint + interval.right);
    let left_midpoint_value = function(left_midpoint);
    let right_midpoint_value = function(right_midpoint);
    let left_estimate = simpson(
        interval.left,
        interval.midpoint,
        interval.left_value,
        left_midpoint_value,
        interval.midpoint_value,
    );
    let right_estimate = simpson(
        interval.midpoint,
        interval.right,
        interval.midpoint_value,
        right_midpoint_value,
        interval.right_value,
    );
    let refined = left_estimate + right_estimate;
    let correction = (refined - interval.estimate) / 15.0;
    let tolerance = absolute_tolerance + relative_tolerance * refined.abs();
    if correction.abs() <= tolerance {
        let result = refined + correction;
        return if result.is_finite() {
            Ok(result)
        } else {
            Err(OneDividendReferenceError::NonFiniteResult)
        };
    }
    if depth == 0 {
        return Err(OneDividendReferenceError::IntegrationDidNotConverge { depth });
    }
    let left_interval = SimpsonInterval {
        left: interval.left,
        midpoint: left_midpoint,
        right: interval.midpoint,
        left_value: interval.left_value,
        midpoint_value: left_midpoint_value,
        right_value: interval.midpoint_value,
        estimate: left_estimate,
    };
    let right_interval = SimpsonInterval {
        left: interval.midpoint,
        midpoint: right_midpoint,
        right: interval.right,
        left_value: interval.midpoint_value,
        midpoint_value: right_midpoint_value,
        right_value: interval.right_value,
        estimate: right_estimate,
    };
    let next_absolute = 0.5 * absolute_tolerance;
    Ok(adaptive_simpson_interval(
        function,
        left_interval,
        next_absolute,
        relative_tolerance,
        depth - 1,
    )? + adaptive_simpson_interval(
        function,
        right_interval,
        next_absolute,
        relative_tolerance,
        depth - 1,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_published_liquidator_values() {
        for (strike, dividend_time, expected) in [
            (100.0, 0.0001, 10.5806),
            (100.0, 0.5000, 11.6564),
            (100.0, 0.9999, 14.7162),
            (130.0, 0.5000, 3.4595),
            (70.0, 0.5000, 32.4608),
        ] {
            let observed = one_dividend_american_call(
                100.0,
                strike,
                0.06,
                0.30,
                1.0,
                dividend_time,
                7.0,
                DividendPolicy::Liquidator,
            )
            .unwrap_or_else(|error| {
                panic!("strike={strike}, dividend_time={dividend_time}: {error}")
            });
            assert!((observed - expected).abs() <= 6.0e-4, "observed={observed}");
        }
    }

    #[test]
    fn rejects_negative_rate_for_the_one_exercise_formula() {
        assert_eq!(
            one_dividend_american_call(
                100.0,
                100.0,
                -0.01,
                0.30,
                1.0,
                0.5,
                7.0,
                DividendPolicy::Liquidator,
            ),
            Err(OneDividendReferenceError::NegativeRate { rate: -0.01 })
        );
    }
}
