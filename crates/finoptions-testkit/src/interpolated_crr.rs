//! Independent Vellekoop-Nieuwenhuis-style interpolated cash-dividend tree.

use finoptions_core::{
    ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve,
};
use thiserror::Error;

const MAXIMUM_STEPS: usize = 4_294_967_295;

/// A non-production interpolated-tree request is invalid or numerically unrepresentable.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum InterpolatedCrrError {
    /// The schedule and discount curve do not use one exact grid.
    #[error("CRR schedule and log-discount curve use different time grids")]
    DiscountTimeGridMismatch,

    /// The schedule and proportional-yield curve do not use one exact grid.
    #[error("CRR schedule and cumulative dividend-yield curve use different time grids")]
    DividendYieldTimeGridMismatch,

    /// Initial spot is not finite and strictly positive.
    #[error("CRR initial spot must be finite and strictly positive; observed {observed}")]
    InvalidInitialSpot {
        /// Rejected initial spot.
        observed: f64,
    },

    /// Strike is not finite and strictly positive.
    #[error("CRR strike must be finite and strictly positive; observed {observed}")]
    InvalidStrike {
        /// Rejected strike.
        observed: f64,
    },

    /// The tree has too few time steps.
    #[error("interpolated CRR requires at least four steps; observed {observed}")]
    TooFewSteps {
        /// Rejected step count.
        observed: usize,
    },

    /// The tree exceeds the exact integer-to-f64 step domain.
    #[error("interpolated CRR supports at most {maximum} steps; observed {observed}")]
    TooManySteps {
        /// Largest admitted step count.
        maximum: usize,
        /// Rejected step count.
        observed: usize,
    },

    /// One exact event time does not coincide with a requested tree step.
    #[error("CRR event node {node} at time {time} does not align with {steps} steps")]
    EventOffGrid {
        /// Event node.
        node: usize,
        /// Exact model time.
        time: f64,
        /// Requested tree step count.
        steps: usize,
    },

    /// An event interval contains no tree transition.
    #[error("CRR event interval {interval} contains no tree step")]
    EmptyEventInterval {
        /// Schedule interval.
        interval: usize,
    },

    /// The declared step size does not admit an arbitrage-free CRR probability.
    #[error("CRR interval {interval} has invalid risk-neutral probability {probability}")]
    InvalidProbability {
        /// Schedule interval.
        interval: usize,
        /// Rejected probability.
        probability: f64,
    },

    /// A caller-sized tree buffer could not be allocated.
    #[error("could not reserve {elements} elements for CRR buffer {buffer}")]
    Allocation {
        /// Stable buffer identity.
        buffer: &'static str,
        /// Requested element count.
        elements: usize,
    },

    /// A future dividend-policy variant is not implemented independently.
    #[error("interpolated CRR does not implement the requested dividend policy")]
    UnsupportedDividendPolicy,

    /// Finite inputs produced an invalid tree state or result.
    #[error("interpolated CRR produced a non-finite or negative value at step {step}")]
    InvalidValue {
        /// Tree step at failure.
        step: usize,
    },
}

#[derive(Clone, Copy)]
enum SupportedDividendPolicy {
    Liquidator,
    Survivor,
}

/// Prices the exact finite schedule with an independently implemented interpolated CRR tree.
///
/// Every schedule node must align exactly with the requested uniform tree. Fixed cash jumps are
/// handled by linear interpolation of the post-event value function on the unshifted recombining
/// lattice. Exercise is applied to the pre-dividend state, and terminal cash precedes payoff.
///
/// # Errors
///
/// Returns a typed domain, alignment, probability, allocation, or arithmetic error. No step,
/// event, probability, or policy is changed.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn interpolated_crr_call(
    initial_spot: f64,
    strike: f64,
    schedule: &EventSchedule,
    curve: &LogDiscountCurve,
    dividend_yield_curve: &CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
    dividend_policy: DividendPolicy,
    steps: usize,
) -> Result<f64, InterpolatedCrrError> {
    validate_inputs(
        initial_spot,
        strike,
        schedule,
        curve,
        dividend_yield_curve,
        steps,
    )?;
    let policy = match dividend_policy {
        DividendPolicy::Liquidator => SupportedDividendPolicy::Liquidator,
        DividendPolicy::Survivor => SupportedDividendPolicy::Survivor,
        _ => return Err(InterpolatedCrrError::UnsupportedDividendPolicy),
    };
    let maturity = schedule.grid().as_slice()[schedule.grid().len() - 1];
    let dt = maturity / exact_f64(steps);
    let (event_steps, node_steps) = aligned_event_steps(schedule, dt, steps)?;
    let up = (volatility.get() * dt.sqrt()).exp();
    let down = 1.0 / up;
    let up_over_down = up / down;
    let mut probabilities = allocated_f64("probabilities", steps)?;
    let mut discounts = allocated_f64("discounts", steps)?;
    for interval in 0..node_steps.len() - 1 {
        let start_step = node_steps[interval];
        let end_step = node_steps[interval + 1];
        let step_count = end_step - start_step;
        if step_count == 0 {
            return Err(InterpolatedCrrError::EmptyEventInterval { interval });
        }
        let from = schedule
            .grid()
            .node_id(interval)
            .map_err(|_| InterpolatedCrrError::InvalidValue { step: start_step })?;
        let to = schedule
            .grid()
            .node_id(interval + 1)
            .map_err(|_| InterpolatedCrrError::InvalidValue { step: end_step })?;
        let step_log_discount = curve
            .interval_log_discount(from, to)
            .map_err(|_| InterpolatedCrrError::InvalidValue { step: start_step })?
            / exact_f64(step_count);
        let step_yield = dividend_yield_curve
            .interval_yield(from, to)
            .map_err(|_| InterpolatedCrrError::InvalidValue { step: start_step })?
            / exact_f64(step_count);
        let growth = (-step_log_discount - step_yield).exp();
        let probability = (growth - down) / (up - down);
        if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
            return Err(InterpolatedCrrError::InvalidProbability {
                interval,
                probability,
            });
        }
        probabilities[start_step..end_step].fill(probability);
        discounts[start_step..end_step].fill(step_log_discount.exp());
    }

    let mut stock_values = lattice_spots(initial_spot, down, up_over_down, steps)?;
    let expiry_phases = schedule
        .phases(schedule.expiry())
        .map_err(|_| InterpolatedCrrError::InvalidValue { step: steps })?;
    let terminal_cash = expiry_phases.dividend.map(|cash| cash.amount());
    let mut values = allocated_f64("option values", steps + 1)?;
    for (value, &pre_event_spot) in values.iter_mut().zip(&stock_values) {
        let post_event_spot = apply_cash(pre_event_spot, terminal_cash, policy);
        *value = (post_event_spot - strike).max(0.0);
    }
    let mut mapped = allocated_f64("cash interpolation", steps + 1)?;

    for step in (0..steps).rev() {
        let probability = probabilities[step];
        let discount = discounts[step];
        for node in 0..=step {
            values[node] = discount
                * (probability.mul_add(values[node + 1], (1.0 - probability) * values[node]));
        }
        values.truncate(step + 1);

        let Some(schedule_node) = event_steps[step] else {
            continue;
        };
        stock_values = lattice_spots(initial_spot, down, up_over_down, step)?;
        let node_id = schedule
            .grid()
            .node_id(schedule_node)
            .map_err(|_| InterpolatedCrrError::InvalidValue { step })?;
        let phases = schedule
            .phases(node_id)
            .map_err(|_| InterpolatedCrrError::InvalidValue { step })?;
        if let Some(dividend) = phases.dividend {
            mapped.resize(step + 1, 0.0);
            for (&pre_event_spot, output) in stock_values.iter().zip(&mut mapped) {
                let post_event_spot = apply_cash(pre_event_spot, Some(dividend.amount()), policy);
                *output = interpolate_lattice(&stock_values, &values, post_event_spot);
                if !output.is_finite() || *output < 0.0 {
                    return Err(InterpolatedCrrError::InvalidValue { step });
                }
            }
            std::mem::swap(&mut values, &mut mapped);
        }
        if phases.exercise {
            for (&spot, value) in stock_values.iter().zip(&mut values) {
                *value = value.max((spot - strike).max(0.0));
            }
        }
    }
    let result = values[0];
    if !result.is_finite() || result < 0.0 {
        return Err(InterpolatedCrrError::InvalidValue { step: 0 });
    }
    Ok(result)
}

fn validate_inputs(
    initial_spot: f64,
    strike: f64,
    schedule: &EventSchedule,
    curve: &LogDiscountCurve,
    dividend_yield_curve: &CumulativeDividendYieldCurve,
    steps: usize,
) -> Result<(), InterpolatedCrrError> {
    if !initial_spot.is_finite() || initial_spot <= 0.0 {
        return Err(InterpolatedCrrError::InvalidInitialSpot {
            observed: initial_spot,
        });
    }
    if !strike.is_finite() || strike <= 0.0 {
        return Err(InterpolatedCrrError::InvalidStrike { observed: strike });
    }
    if steps < 4 {
        return Err(InterpolatedCrrError::TooFewSteps { observed: steps });
    }
    if steps > MAXIMUM_STEPS {
        return Err(InterpolatedCrrError::TooManySteps {
            maximum: MAXIMUM_STEPS,
            observed: steps,
        });
    }
    if schedule.grid() != curve.grid() {
        return Err(InterpolatedCrrError::DiscountTimeGridMismatch);
    }
    if schedule.grid() != dividend_yield_curve.grid() {
        return Err(InterpolatedCrrError::DividendYieldTimeGridMismatch);
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Nearest is finite, nonnegative, integral, and bounded by validated u32 steps.
fn aligned_event_steps(
    schedule: &EventSchedule,
    dt: f64,
    steps: usize,
) -> Result<(Vec<Option<usize>>, Vec<usize>), InterpolatedCrrError> {
    let mut event_steps = vec![None; steps + 1];
    let mut node_steps = Vec::new();
    node_steps
        .try_reserve_exact(schedule.grid().len())
        .map_err(|_| InterpolatedCrrError::Allocation {
            buffer: "event steps",
            elements: schedule.grid().len(),
        })?;
    for (node, &time) in schedule.grid().as_slice().iter().enumerate() {
        let position = time / dt;
        let nearest = position.round();
        let tolerance = 32.0 * f64::EPSILON * position.abs().max(1.0);
        if (position - nearest).abs() > tolerance || nearest < 0.0 || nearest > exact_f64(steps) {
            return Err(InterpolatedCrrError::EventOffGrid { node, time, steps });
        }
        let step = nearest as usize;
        if event_steps[step].replace(node).is_some() {
            return Err(InterpolatedCrrError::EventOffGrid { node, time, steps });
        }
        node_steps.push(step);
    }
    Ok((event_steps, node_steps))
}

fn allocated_f64(buffer: &'static str, elements: usize) -> Result<Vec<f64>, InterpolatedCrrError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| InterpolatedCrrError::Allocation { buffer, elements })?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn lattice_spots(
    initial_spot: f64,
    down: f64,
    up_over_down: f64,
    step: usize,
) -> Result<Vec<f64>, InterpolatedCrrError> {
    let mut spots = allocated_f64("stock lattice", step + 1)?;
    let mut spot = initial_spot * down.powf(exact_f64(step));
    for value in &mut spots {
        if !spot.is_finite() || spot < 0.0 {
            return Err(InterpolatedCrrError::InvalidValue { step });
        }
        *value = spot;
        spot *= up_over_down;
    }
    Ok(spots)
}

fn exact_f64(value: usize) -> f64 {
    match u32::try_from(value) {
        Ok(exact) => f64::from(exact),
        Err(_) => unreachable!("validated CRR step counts are within u32"),
    }
}

fn apply_cash(pre_event_spot: f64, amount: Option<f64>, policy: SupportedDividendPolicy) -> f64 {
    let Some(amount) = amount else {
        return pre_event_spot;
    };
    match policy {
        SupportedDividendPolicy::Liquidator => (pre_event_spot - amount).max(0.0),
        SupportedDividendPolicy::Survivor => {
            if pre_event_spot > amount {
                pre_event_spot - amount
            } else {
                pre_event_spot
            }
        }
    }
}

fn interpolate_lattice(spots: &[f64], values: &[f64], target: f64) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let upper = spots.partition_point(|spot| *spot < target);
    if upper == 0 {
        return values[0] * target / spots[0];
    }
    if upper >= spots.len() {
        return values[values.len() - 1];
    }
    let lower = upper - 1;
    let fraction = (target - spots[lower]) / (spots[upper] - spots[lower]);
    (values[upper] - values[lower]).mul_add(fraction, values[lower])
}
