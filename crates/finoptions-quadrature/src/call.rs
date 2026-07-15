//! Backward direct integration for a fixed-strike call.

use std::sync::atomic::{AtomicUsize, Ordering};

use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve,
};
use finoptions_runtime::CpuExecutor;
use finoptions_sim::GbmStep;
use rayon::prelude::*;

use crate::QuadratureError;
use crate::rule::{SUPPORTED_NORMAL_ORDERS, StandardNormalRule};

const MIN_STOCK_POINTS: usize = 1_001;
const MAX_STOCK_POINTS: usize = 4_294_967_295;

/// Explicit numerical identity of one event-aligned direct-integration method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventAlignedQuadratureSpec {
    upper_stock: f64,
    stock_points: usize,
    normal_order: usize,
}

impl EventAlignedQuadratureSpec {
    /// Validates fixed numerical controls without choosing or refining them.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid boundary, a stock grid below the admitted reference
    /// domain, or a normal order outside the validated Golub-Welsch range.
    pub fn try_new(
        upper_stock: f64,
        stock_points: usize,
        normal_order: usize,
    ) -> Result<Self, QuadratureError> {
        if !upper_stock.is_finite() || upper_stock <= 0.0 {
            return Err(QuadratureError::InvalidUpperStockBoundary {
                observed: upper_stock,
            });
        }
        if stock_points < MIN_STOCK_POINTS {
            return Err(QuadratureError::TooFewStockPoints {
                minimum: MIN_STOCK_POINTS,
                observed: stock_points,
            });
        }
        if stock_points > MAX_STOCK_POINTS {
            return Err(QuadratureError::TooManyStockPoints {
                maximum: MAX_STOCK_POINTS,
                observed: stock_points,
            });
        }
        if ![32, 64, 128, 256, 512, 1_024, 2_048].contains(&normal_order) {
            return Err(QuadratureError::UnsupportedQuadratureOrder {
                observed: normal_order,
                supported: SUPPORTED_NORMAL_ORDERS,
            });
        }
        Ok(Self {
            upper_stock,
            stock_points,
            normal_order,
        })
    }

    /// Returns the exact upper stock boundary.
    #[must_use]
    pub const fn upper_stock(self) -> f64 {
        self.upper_stock
    }

    /// Returns the uniform stock-grid point count, including both boundaries.
    #[must_use]
    pub const fn stock_points(self) -> usize {
        self.stock_points
    }

    /// Returns the standard-normal Gauss-Hermite order.
    #[must_use]
    pub const fn normal_order(self) -> usize {
        self.normal_order
    }
}

/// Reusable validated quadrature rule and stock-grid geometry.
#[derive(Clone, Debug)]
pub struct EventAlignedQuadrature {
    spec: EventAlignedQuadratureSpec,
    normal_rule: StandardNormalRule,
}

/// Price plus the largest discontinuity between the computed upper grid value and affine tail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadratureCallResult {
    price: f64,
    maximum_upper_boundary_gap: f64,
}

impl QuadratureCallResult {
    /// Returns the finite nonnegative call value.
    #[must_use]
    pub const fn price(self) -> f64 {
        self.price
    }

    /// Returns the largest absolute upper-grid/affine-tail gap across backward event nodes.
    #[must_use]
    pub const fn maximum_upper_boundary_gap(self) -> f64 {
        self.maximum_upper_boundary_gap
    }
}

impl EventAlignedQuadrature {
    /// Constructs and validates the requested standard-normal quadrature rule once.
    ///
    /// # Errors
    ///
    /// Returns a typed error if Golub-Welsch does not produce a valid rule at the declared order.
    pub fn try_new(spec: EventAlignedQuadratureSpec) -> Result<Self, QuadratureError> {
        let normal_rule = StandardNormalRule::try_new(spec.normal_order)?;
        Ok(Self { spec, normal_rule })
    }

    /// Returns the exact immutable numerical specification.
    #[must_use]
    pub const fn spec(&self) -> EventAlignedQuadratureSpec {
        self.spec
    }

    /// Prices the caller-declared finite call exercise schedule by backward direct integration.
    ///
    /// Exercise occurs on the pre-dividend state. A same-node cash jump follows exercise. At the
    /// terminal node, an admitted cash jump precedes expiry payoff. The method never changes a
    /// grid, event, boundary, order, policy, or execution backend.
    ///
    /// # Errors
    ///
    /// Returns a typed financial, grid, allocation, transition, quadrature, or arithmetic error.
    #[allow(
        clippy::cast_precision_loss,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn price_call(
        &self,
        executor: &CpuExecutor,
        initial_spot: f64,
        strike: f64,
        schedule: &EventSchedule,
        curve: &LogDiscountCurve,
        dividend_yield_curve: &CumulativeDividendYieldCurve,
        volatility: ConstantVolatility,
        dividend_policy: DividendPolicy,
    ) -> Result<QuadratureCallResult, QuadratureError> {
        validate_request(
            self.spec,
            initial_spot,
            strike,
            schedule,
            curve,
            dividend_yield_curve,
        )?;
        let steps = build_steps(curve, dividend_yield_curve, volatility)?;
        let mut values = allocated_f64("values", self.spec.stock_points)?;
        let mut continuation = allocated_f64("continuation", self.spec.stock_points)?;
        let spacing = self.spec.upper_stock / (self.spec.stock_points - 1) as f64;
        let expiry_phases = schedule.phases(schedule.expiry())?;
        let terminal_dividend = expiry_phases.dividend.copied();
        for (index, value) in values.iter_mut().enumerate() {
            let pre_event_spot = spacing * index as f64;
            let post_event_spot =
                apply_dividend(terminal_dividend, dividend_policy, pre_event_spot)?;
            *value = (post_event_spot - strike).max(0.0);
        }
        let mut upper_affine = AffineTail {
            slope: 1.0,
            intercept: -strike - terminal_dividend.map_or(0.0, CashDividend::amount),
        };
        upper_affine.validate_at(self.spec.upper_stock, schedule.expiry().get())?;
        let mut maximum_gap = upper_boundary_gap(&values, upper_affine, self.spec.upper_stock);
        let mut exponents = allocated_f64("normal exponents", self.normal_rule.nodes().len())?;
        let mut multipliers =
            allocated_f64("transition multipliers", self.normal_rule.nodes().len())?;

        for interval in (0..steps.len()).rev() {
            let step = steps[interval];
            for (output, &normal) in exponents.iter_mut().zip(self.normal_rule.nodes()) {
                *output = step.diffusion().mul_add(normal, step.log_drift());
            }
            executor
                .math_backend()
                .exp_f64(&exponents, &mut multipliers)?;
            if let Some(normal_node) = multipliers
                .iter()
                .position(|multiplier| !multiplier.is_finite() || *multiplier <= 0.0)
            {
                return Err(QuadratureError::NonFiniteTransition {
                    interval,
                    stock_point: 0,
                    normal_node,
                });
            }

            let from = schedule.grid().node_id(interval)?;
            let to = schedule.grid().node_id(interval + 1)?;
            let interval_discount = curve.interval_log_discount(from, to)?.exp();
            let interval_payout = (-dividend_yield_curve.interval_yield(from, to)?).exp();
            let post_event_affine =
                upper_affine.transition(interval, interval_discount, interval_payout)?;
            transition_grid(
                executor,
                interval,
                spacing,
                self.spec.upper_stock,
                &values,
                &mut continuation,
                upper_affine,
                &multipliers,
                self.normal_rule.weights(),
                interval_discount,
            )?;

            let phases = schedule.phases(from)?;
            let mut pre_event_affine = post_event_affine;
            if let Some(dividend) = phases.dividend.copied() {
                map_cash_event(
                    spacing,
                    self.spec.upper_stock,
                    &continuation,
                    &mut values,
                    post_event_affine,
                    dividend,
                    dividend_policy,
                )?;
                pre_event_affine = post_event_affine.after_cash(interval, dividend.amount())?;
            } else {
                std::mem::swap(&mut values, &mut continuation);
            }
            if phases.exercise {
                for (index, value) in values.iter_mut().enumerate() {
                    let spot = spacing * index as f64;
                    *value = value.max((spot - strike).max(0.0));
                }
                pre_event_affine = pre_event_affine.max_with_call_intrinsic(strike);
            }
            pre_event_affine.validate_at(self.spec.upper_stock, from.get())?;
            maximum_gap = maximum_gap.max(upper_boundary_gap(
                &values,
                pre_event_affine,
                self.spec.upper_stock,
            ));
            upper_affine = pre_event_affine;
        }

        let price = interpolate_uniform(
            initial_spot,
            spacing,
            self.spec.upper_stock,
            &values,
            upper_affine,
        );
        if !price.is_finite() || price < 0.0 {
            return Err(QuadratureError::InvalidResult { observed: price });
        }
        Ok(QuadratureCallResult {
            price,
            maximum_upper_boundary_gap: maximum_gap,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct AffineTail {
    slope: f64,
    intercept: f64,
}

impl AffineTail {
    fn value(self, spot: f64) -> f64 {
        self.slope.mul_add(spot, self.intercept)
    }

    fn transition(
        self,
        interval: usize,
        discount: f64,
        payout: f64,
    ) -> Result<Self, QuadratureError> {
        let transitioned = Self {
            slope: self.slope * payout,
            intercept: self.intercept * discount,
        };
        if !transitioned.slope.is_finite() || !transitioned.intercept.is_finite() {
            return Err(QuadratureError::NonFiniteValue {
                interval,
                stock_point: 0,
            });
        }
        Ok(transitioned)
    }

    fn after_cash(self, interval: usize, amount: f64) -> Result<Self, QuadratureError> {
        let intercept = (-self.slope).mul_add(amount, self.intercept);
        if !intercept.is_finite() {
            return Err(QuadratureError::NonFiniteValue {
                interval,
                stock_point: 0,
            });
        }
        Ok(Self {
            slope: self.slope,
            intercept,
        })
    }

    #[allow(clippy::float_cmp)] // Exact slope equality selects between two affine asymptotes.
    fn max_with_call_intrinsic(self, strike: f64) -> Self {
        if self.slope > 1.0 || (self.slope == 1.0 && self.intercept >= -strike) {
            self
        } else {
            Self {
                slope: 1.0,
                intercept: -strike,
            }
        }
    }

    fn validate_at(self, upper: f64, node: u32) -> Result<(), QuadratureError> {
        let value = self.value(upper);
        if !value.is_finite() || value < 0.0 {
            return Err(QuadratureError::InvalidAffineBoundary { node, upper, value });
        }
        Ok(())
    }
}

fn validate_request(
    spec: EventAlignedQuadratureSpec,
    initial_spot: f64,
    strike: f64,
    schedule: &EventSchedule,
    curve: &LogDiscountCurve,
    dividend_yield_curve: &CumulativeDividendYieldCurve,
) -> Result<(), QuadratureError> {
    if !initial_spot.is_finite()
        || initial_spot < 0.0
        || (initial_spot == 0.0 && initial_spot.is_sign_negative())
    {
        return Err(QuadratureError::InvalidInitialSpot {
            observed: initial_spot,
        });
    }
    if !strike.is_finite() || strike <= 0.0 {
        return Err(QuadratureError::InvalidStrike { observed: strike });
    }
    if spec.upper_stock <= initial_spot.max(strike) {
        return Err(QuadratureError::UpperStockBoundaryTooLow {
            upper: spec.upper_stock,
            initial_spot,
            strike,
        });
    }
    if schedule.grid() != curve.grid() {
        return Err(QuadratureError::DiscountTimeGridMismatch);
    }
    if schedule.grid() != dividend_yield_curve.grid() {
        return Err(QuadratureError::DividendYieldTimeGridMismatch);
    }
    Ok(())
}

fn build_steps(
    curve: &LogDiscountCurve,
    dividend_yield_curve: &CumulativeDividendYieldCurve,
    volatility: ConstantVolatility,
) -> Result<Vec<GbmStep>, QuadratureError> {
    let grid = curve.grid();
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(grid.len() - 1)
        .map_err(|_| QuadratureError::Allocation {
            buffer: "GBM steps",
            elements: grid.len() - 1,
        })?;
    for index in 1..grid.len() {
        steps.push(GbmStep::try_new(
            curve,
            dividend_yield_curve,
            grid.node_id(index - 1)?,
            grid.node_id(index)?,
            volatility,
        )?);
    }
    Ok(steps)
}

fn allocated_f64(buffer: &'static str, elements: usize) -> Result<Vec<f64>, QuadratureError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| QuadratureError::Allocation { buffer, elements })?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn apply_dividend(
    dividend: Option<CashDividend>,
    policy: DividendPolicy,
    pre_event_spot: f64,
) -> Result<f64, QuadratureError> {
    dividend.map_or(Ok(pre_event_spot), |cash| {
        cash.apply(policy, pre_event_spot).map_err(Into::into)
    })
}

#[allow(clippy::too_many_arguments)]
fn transition_grid(
    executor: &CpuExecutor,
    interval: usize,
    spacing: f64,
    upper: f64,
    future_values: &[f64],
    output: &mut [f64],
    future_affine: AffineTail,
    multipliers: &[f64],
    weights: &[f64],
    discount: f64,
) -> Result<(), QuadratureError> {
    let failure = AtomicUsize::new(usize::MAX);
    executor.dispatch_mut(
        output,
        |output| {
            for (stock_point, value) in output.iter_mut().enumerate() {
                match transition_value(
                    interval,
                    stock_point,
                    spacing,
                    upper,
                    future_values,
                    future_affine,
                    multipliers,
                    weights,
                    discount,
                ) {
                    Ok(observed) => *value = observed,
                    Err(_) => {
                        failure.fetch_min(stock_point, Ordering::Relaxed);
                    }
                }
            }
        },
        |output| {
            output.par_iter_mut().enumerate().for_each(
                |(stock_point, value)| match transition_value(
                    interval,
                    stock_point,
                    spacing,
                    upper,
                    future_values,
                    future_affine,
                    multipliers,
                    weights,
                    discount,
                ) {
                    Ok(observed) => *value = observed,
                    Err(_) => {
                        failure.fetch_min(stock_point, Ordering::Relaxed);
                    }
                },
            );
        },
    );
    let first_failure = failure.load(Ordering::Relaxed);
    if first_failure == usize::MAX {
        return Ok(());
    }
    transition_value(
        interval,
        first_failure,
        spacing,
        upper,
        future_values,
        future_affine,
        multipliers,
        weights,
        discount,
    )
    .map(|_| ())
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn transition_value(
    interval: usize,
    stock_point: usize,
    spacing: f64,
    upper: f64,
    future_values: &[f64],
    future_affine: AffineTail,
    multipliers: &[f64],
    weights: &[f64],
    discount: f64,
) -> Result<f64, QuadratureError> {
    let spot = spacing * stock_point as f64;
    let mut expectation = 0.0;
    for (normal_node, (&multiplier, &weight)) in multipliers.iter().zip(weights).enumerate() {
        let future_spot = spot * multiplier;
        if !future_spot.is_finite() {
            return Err(QuadratureError::NonFiniteTransition {
                interval,
                stock_point,
                normal_node,
            });
        }
        let future_value =
            interpolate_uniform(future_spot, spacing, upper, future_values, future_affine);
        expectation = weight.mul_add(future_value, expectation);
    }
    let value = discount * expectation;
    if !value.is_finite() || value < 0.0 {
        return Err(QuadratureError::NonFiniteValue {
            interval,
            stock_point,
        });
    }
    Ok(value)
}

#[allow(clippy::cast_precision_loss)] // Stock-point indices are validated within u32 in the spec.
fn map_cash_event(
    spacing: f64,
    upper: f64,
    post_event_values: &[f64],
    pre_event_values: &mut [f64],
    post_event_affine: AffineTail,
    dividend: CashDividend,
    dividend_policy: DividendPolicy,
) -> Result<(), QuadratureError> {
    for (index, value) in pre_event_values.iter_mut().enumerate() {
        let pre_event_spot = spacing * index as f64;
        let post_event_spot = dividend.apply(dividend_policy, pre_event_spot)?;
        *value = interpolate_uniform(
            post_event_spot,
            spacing,
            upper,
            post_event_values,
            post_event_affine,
        );
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // The finite position is proven in [0, values.len() - 1] before conversion.
fn interpolate_uniform(
    spot: f64,
    spacing: f64,
    upper: f64,
    values: &[f64],
    affine: AffineTail,
) -> f64 {
    if spot > upper {
        return affine.value(spot);
    }
    if spot <= 0.0 {
        return values[0];
    }
    let position = spot / spacing;
    let lower = (position.floor() as usize).min(values.len() - 2);
    let fraction = position - lower as f64;
    (values[lower + 1] - values[lower]).mul_add(fraction, values[lower])
}

fn upper_boundary_gap(values: &[f64], affine: AffineTail, upper: f64) -> f64 {
    (values[values.len() - 1] - affine.value(upper)).abs()
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use finoptions_core::{CashDividend, NodeId, TimeGrid};
    use finoptions_testkit::{interpolated_crr_call, one_dividend_american_call};

    use super::*;

    fn method() -> &'static EventAlignedQuadrature {
        static METHOD: OnceLock<EventAlignedQuadrature> = OnceLock::new();
        METHOD.get_or_init(|| {
            EventAlignedQuadrature::try_new(
                EventAlignedQuadratureSpec::try_new(600.0, 20_001, 2_048).unwrap(),
            )
            .unwrap()
        })
    }

    fn one_dividend_method() -> &'static EventAlignedQuadrature {
        static METHOD: OnceLock<EventAlignedQuadrature> = OnceLock::new();
        METHOD.get_or_init(|| {
            EventAlignedQuadrature::try_new(
                EventAlignedQuadratureSpec::try_new(600.0, 20_001, 1_024).unwrap(),
            )
            .unwrap()
        })
    }

    fn schedule(times: &[f64], dividends: &[(usize, f64)], exercises: &[usize]) -> EventSchedule {
        let grid = TimeGrid::try_new(times.to_vec()).unwrap();
        let cash = dividends
            .iter()
            .map(|&(node, amount)| {
                CashDividend::try_new(grid.node_id(node).unwrap(), amount).unwrap()
            })
            .collect::<Vec<_>>();
        let exercise_nodes = exercises
            .iter()
            .map(|&node| grid.node_id(node).unwrap())
            .collect::<Vec<NodeId>>();
        let expiry = grid.node_id(times.len() - 1).unwrap();
        EventSchedule::try_new(grid, cash, exercise_nodes, expiry).unwrap()
    }

    fn model_curves(
        schedule: &EventSchedule,
        rate: f64,
        yield_rate: f64,
    ) -> (LogDiscountCurve, CumulativeDividendYieldCurve) {
        let log_discounts = schedule
            .grid()
            .as_slice()
            .iter()
            .enumerate()
            .map(|(node, time)| if node == 0 { 0.0 } else { -rate * time })
            .collect::<Vec<_>>();
        let cumulative_yields = schedule
            .grid()
            .as_slice()
            .iter()
            .map(|time| yield_rate * time)
            .collect::<Vec<_>>();
        (
            LogDiscountCurve::try_new(schedule.grid(), log_discounts).unwrap(),
            CumulativeDividendYieldCurve::try_new(schedule.grid(), cumulative_yields).unwrap(),
        )
    }

    fn price(
        quadrature: &EventAlignedQuadrature,
        strike: f64,
        schedule: &EventSchedule,
        rate: f64,
        yield_rate: f64,
        volatility: f64,
        threads: usize,
    ) -> QuadratureCallResult {
        let (curve, dividend_yield_curve) = model_curves(schedule, rate, yield_rate);
        quadrature
            .price_call(
                &CpuExecutor::try_new(threads).unwrap(),
                100.0,
                strike,
                schedule,
                &curve,
                &dividend_yield_curve,
                ConstantVolatility::try_new(volatility).unwrap(),
                DividendPolicy::Liquidator,
            )
            .unwrap()
    }

    #[test]
    fn european_reductions_match_black_scholes_with_and_without_yield() {
        let european = schedule(&[0.0, 1.0], &[], &[]);
        let without_yield = price(method(), 100.0, &european, 0.06, 0.0, 0.30, 1);
        let with_yield = price(method(), 100.0, &european, 0.05, 0.02, 0.20, 1);
        assert!(
            (without_yield.price() - 14.717_072_420_289_284).abs() <= 2.0e-4,
            "{without_yield:?}",
        );
        assert!(
            (with_yield.price() - 9.227_005_508_154_036).abs() <= 2.0e-4,
            "{with_yield:?}",
        );
    }

    #[test]
    fn no_dividend_american_call_matches_european_in_the_nonnegative_rate_domain() {
        let european = schedule(&[0.0, 0.25, 0.5, 0.75, 1.0], &[], &[]);
        let american = schedule(&[0.0, 0.25, 0.5, 0.75, 1.0], &[], &[0, 1, 2, 3]);
        let european_value = price(method(), 100.0, &european, 0.06, 0.0, 0.30, 1).price();
        let american_value = price(method(), 100.0, &american, 0.06, 0.0, 0.30, 1).price();
        assert!((american_value - european_value).abs() <= 2.0e-4);
    }

    #[test]
    fn exercise_precedes_the_published_one_dividend_liquidator_jump() {
        let one_dividend = schedule(&[0.0, 0.5, 1.0], &[(1, 7.0)], &[1]);
        let observed = price(
            one_dividend_method(),
            100.0,
            &one_dividend,
            0.06,
            0.0,
            0.30,
            1,
        );
        let independent = one_dividend_american_call(
            100.0,
            100.0,
            0.06,
            0.30,
            1.0,
            0.5,
            7.0,
            DividendPolicy::Liquidator,
        )
        .unwrap();
        assert!(
            (observed.price() - independent).abs() <= 2.0e-4,
            "{observed:?}",
        );
        assert!(observed.maximum_upper_boundary_gap() <= 2.0e-4);
    }

    #[test]
    fn terminal_liquidator_cash_precedes_expiry_payoff() {
        let terminal_cash = schedule(&[0.0, 1.0], &[(1, 7.0)], &[]);
        let no_cash = schedule(&[0.0, 1.0], &[], &[]);
        let observed = price(method(), 100.0, &terminal_cash, 0.06, 0.0, 0.30, 1);
        let effective_strike = price(method(), 107.0, &no_cash, 0.06, 0.0, 0.30, 1);
        assert_eq!(observed, effective_strike);
    }

    #[test]
    fn terminal_cash_bermudan_schedule_agrees_with_the_independent_interpolated_tree() {
        let intervals = 256_u32;
        let intervals_usize = usize::try_from(intervals).unwrap();
        let times = (0..=intervals)
            .map(|node| f64::from(node) / f64::from(intervals))
            .collect::<Vec<_>>();
        let exercises = (0..intervals_usize).collect::<Vec<_>>();
        let terminal_cash = schedule(&times, &[(intervals_usize, 4.0)], &exercises);
        let (curve, dividend_yield_curve) = model_curves(&terminal_cash, 0.03, 0.0);
        let quadrature = EventAlignedQuadrature::try_new(
            EventAlignedQuadratureSpec::try_new(400.0, 10_001, 512).unwrap(),
        )
        .unwrap();
        let observed = quadrature
            .price_call(
                &CpuExecutor::try_new(4).unwrap(),
                100.0,
                100.0,
                &terminal_cash,
                &curve,
                &dividend_yield_curve,
                ConstantVolatility::try_new(0.25).unwrap(),
                DividendPolicy::Liquidator,
            )
            .unwrap();
        let independent = interpolated_crr_call(
            100.0,
            100.0,
            &terminal_cash,
            &curve,
            &dividend_yield_curve,
            ConstantVolatility::try_new(0.25).unwrap(),
            DividendPolicy::Liquidator,
            16_384,
        )
        .unwrap();
        assert!(
            (observed.price() - independent).abs() <= 2.0e-3,
            "quadrature={observed:?}, tree={independent}",
        );
    }

    #[test]
    fn serial_and_dedicated_pool_results_are_bitwise_identical() {
        let one_dividend = schedule(&[0.0, 0.5, 1.0], &[(1, 7.0)], &[1]);
        assert_eq!(
            price(
                one_dividend_method(),
                100.0,
                &one_dividend,
                0.06,
                0.0,
                0.30,
                1,
            ),
            price(
                one_dividend_method(),
                100.0,
                &one_dividend,
                0.06,
                0.0,
                0.30,
                4,
            ),
        );
    }

    #[test]
    fn independent_interpolated_tree_converges_to_the_one_dividend_reference() {
        let one_dividend = schedule(&[0.0, 0.5, 1.0], &[(1, 7.0)], &[1]);
        let (curve, dividend_yield_curve) = model_curves(&one_dividend, 0.06, 0.0);
        let tree = interpolated_crr_call(
            100.0,
            100.0,
            &one_dividend,
            &curve,
            &dividend_yield_curve,
            ConstantVolatility::try_new(0.30).unwrap(),
            DividendPolicy::Liquidator,
            16_000,
        )
        .unwrap();
        let independent = one_dividend_american_call(
            100.0,
            100.0,
            0.06,
            0.30,
            1.0,
            0.5,
            7.0,
            DividendPolicy::Liquidator,
        )
        .unwrap();
        assert!((tree - independent).abs() <= 5.0e-5, "tree={tree}");
    }
}
