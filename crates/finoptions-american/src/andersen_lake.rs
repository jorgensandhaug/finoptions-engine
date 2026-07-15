//! Grouped Andersen-Lake-Offengenden pricing for continuous-yield American calls.
//!
//! The fixed-point boundary is evaluated at Chebyshev collocation nodes and reconstructed with a
//! shape-preserving piecewise cubic in the transformed boundary coordinate. This implementation
//! therefore does not rely on a clipped Chebyshev interpolant.

use std::f64::consts::{PI, SQRT_2};
use std::ops::Range;

use finoptions_runtime::CpuExecutor;
use rayon::prelude::*;

use crate::error::AmericanOptionError;
use crate::legendre::LegendreRule;

const MIN_COLLOCATION_NODES: usize = 2;
const MAX_COLLOCATION_NODES: usize = 64;
const MAX_ITERATIONS: usize = 64;
const SHARDS_PER_THREAD: usize = 4;

/// Fixed numerical identity and strict convergence gate for one Andersen-Lake method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AndersenLakeSpec {
    boundary_quadrature_order: usize,
    iterations: usize,
    collocation_nodes: usize,
    price_quadrature_order: usize,
    maximum_relative_boundary_residual: f64,
    maximum_relative_obstacle_projection: f64,
}

impl AndersenLakeSpec {
    /// Validates every fixed numerical control without selecting or refining one.
    ///
    /// `collocation_nodes` counts positive-time Chebyshev nodes; the exact right limit at zero is
    /// an additional node. Both quadrature orders must identify checked-in SciPy-generated
    /// Gauss-Legendre rules. The implementation performs exactly `iterations` ordinary system-B
    /// fixed-point steps and errors if the final nodewise residual exceeds the declared maximum.
    /// The separate obstacle-projection limit bounds any finite-precision restoration of the exact
    /// stopping lower bound at the valuation origin; zero disables that projection.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an unsupported count, rule, or residual tolerance.
    pub fn try_new(
        boundary_quadrature_order: usize,
        iterations: usize,
        collocation_nodes: usize,
        price_quadrature_order: usize,
        maximum_relative_boundary_residual: f64,
        maximum_relative_obstacle_projection: f64,
    ) -> Result<Self, AmericanOptionError> {
        if iterations == 0 || iterations > MAX_ITERATIONS {
            return Err(AmericanOptionError::InvalidIterations {
                maximum: MAX_ITERATIONS,
                observed: iterations,
            });
        }
        if !(MIN_COLLOCATION_NODES..=MAX_COLLOCATION_NODES).contains(&collocation_nodes) {
            return Err(AmericanOptionError::InvalidCollocationNodes {
                minimum: MIN_COLLOCATION_NODES,
                maximum: MAX_COLLOCATION_NODES,
                observed: collocation_nodes,
            });
        }
        LegendreRule::try_new(boundary_quadrature_order)?;
        LegendreRule::try_new(price_quadrature_order)?;
        if !maximum_relative_boundary_residual.is_finite()
            || maximum_relative_boundary_residual <= 0.0
        {
            return Err(AmericanOptionError::InvalidBoundaryResidualTolerance {
                observed: maximum_relative_boundary_residual,
            });
        }
        if !maximum_relative_obstacle_projection.is_finite()
            || maximum_relative_obstacle_projection < 0.0
        {
            return Err(AmericanOptionError::InvalidObstacleProjectionTolerance {
                observed: maximum_relative_obstacle_projection,
            });
        }
        Ok(Self {
            boundary_quadrature_order,
            iterations,
            collocation_nodes,
            price_quadrature_order,
            maximum_relative_boundary_residual,
            maximum_relative_obstacle_projection,
        })
    }

    /// Returns the Gauss-Legendre order inside each boundary fixed-point evaluation.
    #[must_use]
    pub const fn boundary_quadrature_order(self) -> usize {
        self.boundary_quadrature_order
    }

    /// Returns the exact fixed-point iteration count.
    #[must_use]
    pub const fn iterations(self) -> usize {
        self.iterations
    }

    /// Returns the positive-time Chebyshev collocation-node count.
    #[must_use]
    pub const fn collocation_nodes(self) -> usize {
        self.collocation_nodes
    }

    /// Returns the Gauss-Legendre order for converting one boundary into option prices.
    #[must_use]
    pub const fn price_quadrature_order(self) -> usize {
        self.price_quadrature_order
    }

    /// Returns the maximum admitted put-strike-normalized boundary residual.
    #[must_use]
    pub const fn maximum_relative_boundary_residual(self) -> f64 {
        self.maximum_relative_boundary_residual
    }

    /// Returns the maximum admitted origin obstacle projection divided by original call spot.
    #[must_use]
    pub const fn maximum_relative_obstacle_projection(self) -> f64 {
        self.maximum_relative_obstacle_projection
    }
}

/// Zero-copy Arrow-style grouped call inputs under constant Black-Scholes parameters.
///
/// One group owns a scalar spot, maturity, continuously compounded risk-free rate, continuous
/// dividend yield, and volatility plus one or more strikes. Group offsets may start at a nonzero
/// base exactly as for a sliced Arrow `LargeList`; the supplied strike slice must span only the
/// referenced child range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupedContinuousYieldCalls<'a> {
    offsets: &'a [i64],
    spots: &'a [f64],
    maturities: &'a [f64],
    rates: &'a [f64],
    dividend_yields: &'a [f64],
    volatilities: &'a [f64],
    strikes: &'a [f64],
    base: usize,
}

impl<'a> GroupedContinuousYieldCalls<'a> {
    /// Validates the complete supported financial and structural domain without copying values.
    ///
    /// The current method deliberately supports only nonnegative continuous call yields. This
    /// includes the recovered study domain and ensures that call-put symmetry maps every boundary
    /// solve to a single-boundary American put with nonnegative rate. Negative call yields are not
    /// reinterpreted, clamped, or routed to another method.
    ///
    /// # Errors
    ///
    /// Returns a typed offset, shape, empty-group, or financial-input error.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        offsets: &'a [i64],
        spots: &'a [f64],
        maturities: &'a [f64],
        rates: &'a [f64],
        dividend_yields: &'a [f64],
        volatilities: &'a [f64],
        strikes: &'a [f64],
    ) -> Result<Self, AmericanOptionError> {
        if offsets.len() < 2 {
            return Err(AmericanOptionError::InsufficientOffsets {
                observed: offsets.len(),
            });
        }
        for (index, &offset) in offsets.iter().enumerate() {
            if offset < 0 {
                return Err(AmericanOptionError::NegativeOffset {
                    index,
                    observed: offset,
                });
            }
            if index > 0 && offsets[index - 1] > offset {
                return Err(AmericanOptionError::DecreasingOffset {
                    previous_index: index - 1,
                    previous: offsets[index - 1],
                    index,
                    observed: offset,
                });
            }
        }
        let first = offsets[0];
        let last_index = offsets.len() - 1;
        let last = offsets[last_index];
        let base =
            usize::try_from(first).map_err(|_| AmericanOptionError::OffsetIndexOverflow {
                index: 0,
                observed: first,
            })?;
        let end = usize::try_from(last).map_err(|_| AmericanOptionError::OffsetIndexOverflow {
            index: last_index,
            observed: last,
        })?;
        let offset_strikes =
            end.checked_sub(base)
                .ok_or(AmericanOptionError::DecreasingOffset {
                    previous_index: 0,
                    previous: first,
                    index: last_index,
                    observed: last,
                })?;
        if offset_strikes != strikes.len() {
            return Err(AmericanOptionError::OffsetSpanMismatch {
                first,
                last,
                offset_strikes,
                buffer_strikes: strikes.len(),
            });
        }
        let groups = offsets.len() - 1;
        for (column, observed) in [
            ("spots", spots.len()),
            ("maturities", maturities.len()),
            ("rates", rates.len()),
            ("dividend_yields", dividend_yields.len()),
            ("volatilities", volatilities.len()),
        ] {
            if observed != groups {
                return Err(AmericanOptionError::GroupColumnLengthMismatch {
                    column,
                    expected: groups,
                    observed,
                });
            }
        }
        let inputs = Self {
            offsets,
            spots,
            maturities,
            rates,
            dividend_yields,
            volatilities,
            strikes,
            base,
        };
        inputs.validate_financial_domain()?;
        Ok(inputs)
    }

    /// Returns the natural option-group count.
    #[must_use]
    pub const fn groups(self) -> usize {
        self.spots.len()
    }

    /// Returns the total strike/contract count.
    #[must_use]
    pub const fn strikes(self) -> usize {
        self.strikes.len()
    }

    fn validate_financial_domain(self) -> Result<(), AmericanOptionError> {
        for group in 0..self.groups() {
            let strike_range = self.strike_range(group);
            if strike_range.is_empty() {
                return Err(AmericanOptionError::EmptyStrikeGroup { group });
            }
            let spot = self.spots[group];
            if !spot.is_finite() || spot <= 0.0 {
                return Err(AmericanOptionError::InvalidSpot {
                    group,
                    observed: spot,
                });
            }
            let maturity = self.maturities[group];
            if !maturity.is_finite() || maturity <= 0.0 {
                return Err(AmericanOptionError::InvalidMaturity {
                    group,
                    observed: maturity,
                });
            }
            let rate = self.rates[group];
            if !rate.is_finite() {
                return Err(AmericanOptionError::InvalidRate {
                    group,
                    observed: rate,
                });
            }
            let dividend_yield = self.dividend_yields[group];
            if !dividend_yield.is_finite()
                || dividend_yield < 0.0
                || (dividend_yield == 0.0 && dividend_yield.is_sign_negative())
            {
                return Err(AmericanOptionError::InvalidDividendYield {
                    group,
                    observed: dividend_yield,
                });
            }
            let volatility = self.volatilities[group];
            if !volatility.is_finite() || volatility <= 0.0 {
                return Err(AmericanOptionError::InvalidVolatility {
                    group,
                    observed: volatility,
                });
            }
            for (strike, &observed) in self.strikes[strike_range].iter().enumerate() {
                if !observed.is_finite() || observed <= 0.0 {
                    return Err(AmericanOptionError::InvalidStrike {
                        group,
                        strike,
                        observed,
                    });
                }
            }
        }
        Ok(())
    }

    fn strike_range(self, group: usize) -> Range<usize> {
        let start = usize::try_from(self.offsets[group]).expect("validated nonnegative offset");
        let end = usize::try_from(self.offsets[group + 1]).expect("validated nonnegative offset");
        start - self.base..end - self.base
    }

    fn group(self, group: usize) -> CallGroup<'a> {
        let range = self.strike_range(group);
        CallGroup {
            spot: self.spots[group],
            maturity: self.maturities[group],
            rate: self.rates[group],
            dividend_yield: self.dividend_yields[group],
            volatility: self.volatilities[group],
            strikes: &self.strikes[range],
        }
    }
}

#[derive(Clone, Copy)]
struct CallGroup<'a> {
    spot: f64,
    maturity: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    strikes: &'a [f64],
}

/// Per-group numerical identity and convergence evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AndersenLakeGroupDiagnostics {
    maximum_relative_boundary_residual: f64,
    maximum_relative_obstacle_projection: f64,
    used_european_identity: bool,
}

impl AndersenLakeGroupDiagnostics {
    /// Returns the maximum collocation-node fixed-point residual divided by transformed put strike.
    #[must_use]
    pub const fn maximum_relative_boundary_residual(self) -> f64 {
        self.maximum_relative_boundary_residual
    }

    /// Returns the largest bounded obstacle projection divided by original call spot.
    #[must_use]
    pub const fn maximum_relative_obstacle_projection(self) -> f64 {
        self.maximum_relative_obstacle_projection
    }

    /// Returns whether the exact no-early-exercise theorem avoided a boundary solve.
    #[must_use]
    pub const fn used_european_identity(self) -> bool {
        self.used_european_identity
    }
}

/// Contract-ordered prices plus group-ordered deterministic diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct AndersenLakeBatchResult {
    prices: Vec<f64>,
    exercised_at_origin: Vec<bool>,
    group_diagnostics: Vec<AndersenLakeGroupDiagnostics>,
}

impl AndersenLakeBatchResult {
    /// Borrows contract prices in child-strike order.
    #[must_use]
    pub fn prices(&self) -> &[f64] {
        &self.prices
    }

    /// Borrows contract-ordered exact stopping decisions at the valuation origin.
    #[must_use]
    pub fn exercised_at_origin(&self) -> &[bool] {
        &self.exercised_at_origin
    }

    /// Borrows one diagnostic record per natural group.
    #[must_use]
    pub fn group_diagnostics(&self) -> &[AndersenLakeGroupDiagnostics] {
        &self.group_diagnostics
    }

    /// Transfers contract prices without copying.
    #[must_use]
    pub fn into_prices(self) -> Vec<f64> {
        self.prices
    }

    /// Transfers every result axis without copying.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, Vec<bool>, Vec<AndersenLakeGroupDiagnostics>) {
        (
            self.prices,
            self.exercised_at_origin,
            self.group_diagnostics,
        )
    }
}

/// Reusable version-pinned quadrature rules and normalized Chebyshev collocation geometry.
#[derive(Clone, Debug)]
pub struct AndersenLake {
    spec: AndersenLakeSpec,
    boundary_rule: LegendreRule,
    price_rule: LegendreRule,
    chebyshev_nodes: Box<[f64]>,
}

impl AndersenLake {
    /// Constructs all immutable quadrature and interpolation resources once.
    ///
    /// # Errors
    ///
    /// Returns a typed specification, checked-in-rule, or allocation error.
    pub fn try_new(spec: AndersenLakeSpec) -> Result<Self, AmericanOptionError> {
        let boundary_rule = LegendreRule::try_new(spec.boundary_quadrature_order)?;
        let price_rule = LegendreRule::try_new(spec.price_quadrature_order)?;
        let node_count = spec.collocation_nodes + 1;
        let mut chebyshev_nodes = allocated_f64("Chebyshev nodes", node_count)?;
        let denominator = exact_f64(spec.collocation_nodes);
        for (index, node) in chebyshev_nodes.iter_mut().enumerate() {
            *node = -(exact_f64(index) * PI / denominator).cos();
        }
        chebyshev_nodes[0] = -1.0;
        chebyshev_nodes[spec.collocation_nodes] = 1.0;
        Ok(Self {
            spec,
            boundary_rule,
            price_rule,
            chebyshev_nodes: chebyshev_nodes.into_boxed_slice(),
        })
    }

    /// Returns the complete immutable numerical identity.
    #[must_use]
    pub const fn spec(&self) -> AndersenLakeSpec {
        self.spec
    }

    /// Prices every group with exact call-put symmetry and outer group parallelism.
    ///
    /// For a continuous-yield American call, the implementation applies
    /// `C_A(S,K,r,q)=P_A(K,S,q,r)` and normalizes the transformed put strike to one. A single
    /// exercise boundary is therefore shared by every strike in one natural input group. When
    /// `q=0` and `r>=0`, the exact no-early-exercise theorem routes directly to the European
    /// Black-Scholes formula. All other admitted groups use fixed-point system B from
    /// Andersen-Lake-Offengenden at Chebyshev collocation nodes, with shape-preserving piecewise
    /// cubic reconstruction of the transformed boundary. The exact stopping payoff is used inside
    /// the solved exercise region. A quadrature value infinitesimally below that Snell-envelope
    /// obstacle is projected only within the caller-declared normalized limit and reported per
    /// group; a larger gap fails. No alternate solver, adaptive refinement, or retry is performed.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, boundary, residual, interpolation, or price error. Parallel
    /// execution deterministically returns the earliest failing group.
    pub fn price_call_groups(
        &self,
        executor: &CpuExecutor,
        inputs: GroupedContinuousYieldCalls<'_>,
    ) -> Result<AndersenLakeBatchResult, AmericanOptionError> {
        let groups = inputs.groups();
        let shard_count = groups.min(executor.threads().saturating_mul(SHARDS_PER_THREAD));
        let ranges = shard_ranges(groups, shard_count)?;
        let shards = executor.dispatch(
            || {
                ranges
                    .iter()
                    .map(|range| self.price_shard(inputs, range.clone()))
                    .collect::<Vec<_>>()
            },
            || {
                ranges
                    .par_iter()
                    .map(|range| self.price_shard(inputs, range.clone()))
                    .collect::<Vec<_>>()
            },
        );
        let mut successful = Vec::new();
        successful.try_reserve_exact(shards.len()).map_err(|_| {
            AmericanOptionError::Allocation {
                buffer: "successful shards",
                elements: shards.len(),
            }
        })?;
        for shard in shards {
            successful.push(shard?);
        }
        let mut prices = allocated_f64("American call prices", inputs.strikes())?;
        let mut exercised_at_origin = Vec::new();
        exercised_at_origin
            .try_reserve_exact(inputs.strikes())
            .map_err(|_| AmericanOptionError::Allocation {
                buffer: "American call origin decisions",
                elements: inputs.strikes(),
            })?;
        let mut diagnostics = Vec::new();
        diagnostics
            .try_reserve_exact(groups)
            .map_err(|_| AmericanOptionError::Allocation {
                buffer: "American call group diagnostics",
                elements: groups,
            })?;
        let mut price_cursor = 0;
        for shard in successful {
            let end = price_cursor + shard.prices.len();
            prices[price_cursor..end].copy_from_slice(&shard.prices);
            price_cursor = end;
            exercised_at_origin.extend(shard.exercised_at_origin);
            diagnostics.extend(shard.diagnostics);
        }
        if price_cursor != inputs.strikes() {
            return Err(AmericanOptionError::ShardShapeMismatch {
                field: "price",
                expected: inputs.strikes(),
                observed: price_cursor,
            });
        }
        if diagnostics.len() != groups {
            return Err(AmericanOptionError::ShardShapeMismatch {
                field: "diagnostic",
                expected: groups,
                observed: diagnostics.len(),
            });
        }
        if exercised_at_origin.len() != inputs.strikes() {
            return Err(AmericanOptionError::ShardShapeMismatch {
                field: "origin decision",
                expected: inputs.strikes(),
                observed: exercised_at_origin.len(),
            });
        }
        Ok(AndersenLakeBatchResult {
            prices,
            exercised_at_origin,
            group_diagnostics: diagnostics,
        })
    }

    fn price_shard(
        &self,
        inputs: GroupedContinuousYieldCalls<'_>,
        groups: Range<usize>,
    ) -> Result<ShardResult, AmericanOptionError> {
        let strikes = groups
            .clone()
            .map(|group| inputs.group(group).strikes.len())
            .sum();
        let mut prices = Vec::new();
        prices
            .try_reserve_exact(strikes)
            .map_err(|_| AmericanOptionError::Allocation {
                buffer: "American call shard prices",
                elements: strikes,
            })?;
        let mut exercised_at_origin = Vec::new();
        exercised_at_origin
            .try_reserve_exact(strikes)
            .map_err(|_| AmericanOptionError::Allocation {
                buffer: "American call shard origin decisions",
                elements: strikes,
            })?;
        let mut diagnostics = Vec::new();
        diagnostics.try_reserve_exact(groups.len()).map_err(|_| {
            AmericanOptionError::Allocation {
                buffer: "American call shard diagnostics",
                elements: groups.len(),
            }
        })?;
        let mut workspace = KernelWorkspace::try_new(self.spec.collocation_nodes)?;
        for group_index in groups {
            let group = inputs.group(group_index);
            let diagnostic = self.price_group(
                group_index,
                group,
                &mut workspace,
                &mut prices,
                &mut exercised_at_origin,
            )?;
            diagnostics.push(diagnostic);
        }
        Ok(ShardResult {
            prices,
            exercised_at_origin,
            diagnostics,
        })
    }

    fn price_group(
        &self,
        group_index: usize,
        group: CallGroup<'_>,
        workspace: &mut KernelWorkspace,
        prices: &mut Vec<f64>,
        exercised_at_origin: &mut Vec<bool>,
    ) -> Result<AndersenLakeGroupDiagnostics, AmericanOptionError> {
        if group.dividend_yield == 0.0 && group.rate >= 0.0 {
            for (strike_index, &strike) in group.strikes.iter().enumerate() {
                let price = european_call(
                    group.spot,
                    strike,
                    group.maturity,
                    group.rate,
                    0.0,
                    group.volatility,
                );
                validate_call_price(group_index, strike_index, group.spot, strike, price)?;
                prices.push(price);
                exercised_at_origin.push(false);
            }
            return Ok(AndersenLakeGroupDiagnostics {
                maximum_relative_boundary_residual: 0.0,
                maximum_relative_obstacle_projection: 0.0,
                used_european_identity: true,
            });
        }

        let transformed_put_rate = group.dividend_yield;
        let transformed_put_yield = group.rate;
        let origin_limit = if transformed_put_rate >= transformed_put_yield {
            1.0
        } else {
            transformed_put_rate / transformed_put_yield
        };
        let residual = self.solve_boundary(
            group_index,
            group.maturity,
            transformed_put_rate,
            transformed_put_yield,
            group.volatility,
            origin_limit,
            workspace,
        )?;
        let exercise_boundary = workspace.boundary[self.spec.collocation_nodes];
        let mut maximum_obstacle_projection = 0.0_f64;
        for (strike_index, &strike) in group.strikes.iter().enumerate() {
            let normalized_put_spot = strike / group.spot;
            let boundary_exercises = normalized_put_spot <= exercise_boundary;
            let (price, exercises, obstacle_projection) = if boundary_exercises {
                ((group.spot - strike).max(0.0), true, 0.0)
            } else {
                let normalized_put = self.price_normalized_put(
                    group_index,
                    group.maturity,
                    transformed_put_rate,
                    transformed_put_yield,
                    group.volatility,
                    normalized_put_spot,
                    origin_limit,
                    &workspace.transformed_boundary,
                    &workspace.interpolation_slopes,
                )?;
                apply_origin_obstacle(
                    group_index,
                    strike_index,
                    group.spot,
                    strike,
                    group.spot * normalized_put,
                    self.spec.maximum_relative_obstacle_projection,
                )?
            };
            maximum_obstacle_projection = maximum_obstacle_projection.max(obstacle_projection);
            validate_call_price(group_index, strike_index, group.spot, strike, price)?;
            prices.push(price);
            exercised_at_origin.push(exercises);
        }
        Ok(AndersenLakeGroupDiagnostics {
            maximum_relative_boundary_residual: residual,
            maximum_relative_obstacle_projection: maximum_obstacle_projection,
            used_european_identity: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_boundary(
        &self,
        group: usize,
        maturity: f64,
        rate: f64,
        dividend_yield: f64,
        volatility: f64,
        origin_limit: f64,
        workspace: &mut KernelWorkspace,
    ) -> Result<f64, AmericanOptionError> {
        let nodes = self.spec.collocation_nodes;
        for index in 0..=nodes {
            let normalized_root_time = (self.chebyshev_nodes[index] + 1.0) * 0.5;
            workspace.times[index] = maturity * normalized_root_time * normalized_root_time;
            workspace.boundary[index] = origin_limit;
        }
        workspace.times[0] = 0.0;
        workspace.times[nodes] = maturity;

        for iteration in 0..self.spec.iterations {
            transform_boundary(
                group,
                origin_limit,
                &workspace.boundary,
                &mut workspace.transformed_boundary,
            )?;
            prepare_shape_preserving_slopes(
                &self.chebyshev_nodes,
                &workspace.transformed_boundary,
                &mut workspace.interpolation_slopes,
            );
            workspace.next_boundary[0] = origin_limit;
            for node in 1..=nodes {
                let fixed_point = self.fixed_point_b(
                    group,
                    workspace.times[node],
                    maturity,
                    rate,
                    dividend_yield,
                    volatility,
                    workspace.boundary[node],
                    origin_limit,
                    &workspace.transformed_boundary,
                    &workspace.interpolation_slopes,
                )?;
                let observed = fixed_point;
                validate_boundary(group, iteration + 1, node, observed, origin_limit)?;
                workspace.next_boundary[node] = observed;
            }
            std::mem::swap(&mut workspace.boundary, &mut workspace.next_boundary);
        }

        transform_boundary(
            group,
            origin_limit,
            &workspace.boundary,
            &mut workspace.transformed_boundary,
        )?;
        prepare_shape_preserving_slopes(
            &self.chebyshev_nodes,
            &workspace.transformed_boundary,
            &mut workspace.interpolation_slopes,
        );
        let mut maximum_residual = 0.0_f64;
        for node in 1..=nodes {
            let fixed_point = self.fixed_point_b(
                group,
                workspace.times[node],
                maturity,
                rate,
                dividend_yield,
                volatility,
                workspace.boundary[node],
                origin_limit,
                &workspace.transformed_boundary,
                &workspace.interpolation_slopes,
            )?;
            maximum_residual = maximum_residual.max((workspace.boundary[node] - fixed_point).abs());
        }
        if !maximum_residual.is_finite()
            || maximum_residual > self.spec.maximum_relative_boundary_residual
        {
            return Err(AmericanOptionError::BoundaryResidualExceeded {
                group,
                observed: maximum_residual,
                tolerance: self.spec.maximum_relative_boundary_residual,
            });
        }
        Ok(maximum_residual)
    }

    #[allow(clippy::too_many_arguments)]
    fn fixed_point_b(
        &self,
        group: usize,
        time: f64,
        maturity: f64,
        rate: f64,
        dividend_yield: f64,
        volatility: f64,
        boundary_at_time: f64,
        origin_limit: f64,
        transformed_boundary: &[f64],
        interpolation_slopes: &[f64],
    ) -> Result<f64, AmericanOptionError> {
        let variance = volatility * volatility;
        let root_time = time.sqrt();
        let denominator = volatility * root_time;
        let log_moneyness = boundary_at_time.ln();
        let d_minus_at_strike =
            (log_moneyness + (rate - dividend_yield - 0.5 * variance) * time) / denominator;
        let d_plus_at_strike = d_minus_at_strike + denominator;
        let mut rate_integral = CompensatedSum::new();
        let mut yield_integral = CompensatedSum::new();
        for &(node, weight) in self.boundary_rule.nodes_and_weights() {
            let one_plus = 1.0 + node;
            let delta = time * 0.25 * one_plus * one_plus;
            let u = time - delta;
            let boundary_at_u = self.interpolate_boundary(
                group,
                u,
                maturity,
                origin_limit,
                transformed_boundary,
                interpolation_slopes,
            )?;
            let root_delta = delta.sqrt();
            let scaled_volatility = volatility * root_delta;
            let relative_log = (boundary_at_time / boundary_at_u).ln();
            let common = relative_log + (rate - dividend_yield) * delta;
            let d_minus = (common - 0.5 * variance * delta) / scaled_volatility;
            let d_plus = d_minus + scaled_volatility;
            let jacobian = time * 0.5 * one_plus * weight;
            rate_integral.add(jacobian * (rate * u).exp() * normal_cdf(d_minus));
            yield_integral.add(jacobian * (dividend_yield * u).exp() * normal_cdf(d_plus));
        }
        let numerator = normal_cdf(d_minus_at_strike) + rate * rate_integral.total();
        let denominator = normal_cdf(d_plus_at_strike) + dividend_yield * yield_integral.total();
        Ok((-(rate - dividend_yield) * time).exp() * numerator / denominator)
    }

    #[allow(clippy::too_many_arguments)]
    fn price_normalized_put(
        &self,
        group: usize,
        maturity: f64,
        rate: f64,
        dividend_yield: f64,
        volatility: f64,
        spot: f64,
        origin_limit: f64,
        transformed_boundary: &[f64],
        interpolation_slopes: &[f64],
    ) -> Result<f64, AmericanOptionError> {
        let variance = volatility * volatility;
        let root_maturity = maturity.sqrt();
        let scaled_volatility = volatility * root_maturity;
        let d_minus =
            (spot.ln() + (rate - dividend_yield - 0.5 * variance) * maturity) / scaled_volatility;
        let d_plus = d_minus + scaled_volatility;
        let european = (-rate * maturity).exp() * normal_cdf(-d_minus)
            - spot * (-dividend_yield * maturity).exp() * normal_cdf(-d_plus);
        let mut premium = CompensatedSum::new();
        for &(node, weight) in self.price_rule.nodes_and_weights() {
            let one_plus = 1.0 + node;
            let delta = maturity * 0.25 * one_plus * one_plus;
            let u = maturity - delta;
            let boundary_at_u = self.interpolate_boundary(
                group,
                u,
                maturity,
                origin_limit,
                transformed_boundary,
                interpolation_slopes,
            )?;
            let scaled_interval_volatility = volatility * delta.sqrt();
            let common = (spot / boundary_at_u).ln() + (rate - dividend_yield) * delta;
            let interval_d_minus = (common - 0.5 * variance * delta) / scaled_interval_volatility;
            let interval_d_plus = interval_d_minus + scaled_interval_volatility;
            let integrand = rate * (-rate * delta).exp() * normal_cdf(-interval_d_minus)
                - dividend_yield
                    * spot
                    * (-dividend_yield * delta).exp()
                    * normal_cdf(-interval_d_plus);
            premium.add(maturity * 0.5 * one_plus * weight * integrand);
        }
        Ok(european + premium.total())
    }

    fn interpolate_boundary(
        &self,
        group: usize,
        time: f64,
        maturity: f64,
        origin_limit: f64,
        transformed_boundary: &[f64],
        interpolation_slopes: &[f64],
    ) -> Result<f64, AmericanOptionError> {
        if time == 0.0 {
            return Ok(origin_limit);
        }
        let normalized = time / maturity;
        let z = normalized.sqrt().mul_add(2.0, -1.0);
        let transformed = shape_preserving_cubic_interpolate(
            z,
            &self.chebyshev_nodes,
            transformed_boundary,
            interpolation_slopes,
        );
        if !transformed.is_finite() || transformed < 0.0 {
            return Err(AmericanOptionError::InvalidBoundaryInterpolation {
                group,
                time,
                observed: transformed,
            });
        }
        let boundary = origin_limit * (-transformed.sqrt()).exp();
        if !boundary.is_finite() || boundary <= 0.0 || boundary > origin_limit {
            return Err(AmericanOptionError::InvalidBoundaryInterpolation {
                group,
                time,
                observed: transformed,
            });
        }
        Ok(boundary)
    }
}

struct ShardResult {
    prices: Vec<f64>,
    exercised_at_origin: Vec<bool>,
    diagnostics: Vec<AndersenLakeGroupDiagnostics>,
}

struct KernelWorkspace {
    times: Vec<f64>,
    boundary: Vec<f64>,
    next_boundary: Vec<f64>,
    transformed_boundary: Vec<f64>,
    interpolation_slopes: Vec<f64>,
}

impl KernelWorkspace {
    fn try_new(collocation_nodes: usize) -> Result<Self, AmericanOptionError> {
        let elements = collocation_nodes + 1;
        Ok(Self {
            times: allocated_f64("boundary times", elements)?,
            boundary: allocated_f64("boundary values", elements)?,
            next_boundary: allocated_f64("next boundary values", elements)?,
            transformed_boundary: allocated_f64("transformed boundary values", elements)?,
            interpolation_slopes: allocated_f64("boundary interpolation slopes", elements)?,
        })
    }
}

fn transform_boundary(
    group: usize,
    origin_limit: f64,
    boundary: &[f64],
    transformed: &mut [f64],
) -> Result<(), AmericanOptionError> {
    for (node, (&observed, output)) in boundary.iter().zip(transformed).enumerate() {
        validate_boundary(group, 0, node, observed, origin_limit)?;
        let log_ratio = (observed / origin_limit).ln();
        *output = log_ratio * log_ratio;
    }
    Ok(())
}

fn validate_boundary(
    group: usize,
    iteration: usize,
    node: usize,
    observed: f64,
    origin_limit: f64,
) -> Result<(), AmericanOptionError> {
    if !observed.is_finite() || observed <= 0.0 || observed > origin_limit {
        return Err(AmericanOptionError::InvalidBoundary {
            group,
            iteration,
            node,
            observed,
            origin_limit,
        });
    }
    Ok(())
}

fn prepare_shape_preserving_slopes(nodes: &[f64], values: &[f64], slopes: &mut [f64]) {
    let intervals = values.len() - 1;
    for index in 1..intervals {
        let previous_width = nodes[index] - nodes[index - 1];
        let next_width = nodes[index + 1] - nodes[index];
        let previous = (values[index] - values[index - 1]) / previous_width;
        let next = (values[index + 1] - values[index]) / next_width;
        if previous == 0.0 || next == 0.0 || previous.signum() != next.signum() {
            slopes[index] = 0.0;
        } else {
            let first_weight = 2.0 * next_width + previous_width;
            let second_weight = next_width + 2.0 * previous_width;
            slopes[index] =
                (first_weight + second_weight) / (first_weight / previous + second_weight / next);
        }
    }
    let first_width = nodes[1] - nodes[0];
    let second_width = nodes[2] - nodes[1];
    let first_secant = (values[1] - values[0]) / first_width;
    let second_secant = (values[2] - values[1]) / second_width;
    slopes[0] = endpoint_slope(first_width, second_width, first_secant, second_secant);

    let last_width = nodes[intervals] - nodes[intervals - 1];
    let penultimate_width = nodes[intervals - 1] - nodes[intervals - 2];
    let last_secant = (values[intervals] - values[intervals - 1]) / last_width;
    let penultimate_secant = (values[intervals - 1] - values[intervals - 2]) / penultimate_width;
    slopes[intervals] = endpoint_slope(
        last_width,
        penultimate_width,
        last_secant,
        penultimate_secant,
    );
}

fn endpoint_slope(first_width: f64, second_width: f64, first: f64, second: f64) -> f64 {
    let candidate = ((2.0 * first_width + second_width) * first - first_width * second)
        / (first_width + second_width);
    if first == 0.0 || candidate.signum() != first.signum() {
        0.0
    } else if first.signum() != second.signum() && candidate.abs() > 3.0 * first.abs() {
        3.0 * first
    } else {
        candidate
    }
}

fn shape_preserving_cubic_interpolate(
    z: f64,
    nodes: &[f64],
    values: &[f64],
    slopes: &[f64],
) -> f64 {
    if z <= nodes[0] {
        return values[0];
    }
    if z >= nodes[nodes.len() - 1] {
        return values[values.len() - 1];
    }
    let upper = nodes.partition_point(|&node| node < z);
    let lower = upper - 1;
    let width = nodes[upper] - nodes[lower];
    let fraction = (z - nodes[lower]) / width;
    let squared = fraction * fraction;
    let cubed = squared * fraction;
    let lower_basis = 2.0 * cubed - 3.0 * squared + 1.0;
    let lower_slope_basis = cubed - 2.0 * squared + fraction;
    let upper_basis = -2.0 * cubed + 3.0 * squared;
    let upper_slope_basis = cubed - squared;
    lower_basis.mul_add(
        values[lower],
        (lower_slope_basis * width).mul_add(
            slopes[lower],
            upper_basis.mul_add(values[upper], upper_slope_basis * width * slopes[upper]),
        ),
    )
}

fn validate_call_price(
    group: usize,
    strike: usize,
    spot: f64,
    strike_value: f64,
    observed: f64,
) -> Result<(), AmericanOptionError> {
    let lower_bound = (spot - strike_value).max(0.0);
    let upper_bound = spot;
    let tolerance = 2_048.0 * f64::EPSILON * spot.max(strike_value).max(1.0);
    if !observed.is_finite()
        || observed < 0.0
        || observed < lower_bound - tolerance
        || observed > upper_bound + tolerance
    {
        return Err(AmericanOptionError::InvalidPrice {
            group,
            strike,
            observed,
            lower_bound,
            upper_bound,
        });
    }
    Ok(())
}

fn apply_origin_obstacle(
    group: usize,
    strike: usize,
    spot: f64,
    strike_value: f64,
    raw_price: f64,
    maximum_relative_projection: f64,
) -> Result<(f64, bool, f64), AmericanOptionError> {
    let intrinsic = (spot - strike_value).max(0.0);
    if !raw_price.is_finite() || raw_price >= intrinsic {
        return Ok((raw_price, false, 0.0));
    }
    let relative_projection = (intrinsic - raw_price) / spot;
    if !relative_projection.is_finite() || relative_projection > maximum_relative_projection {
        return Err(AmericanOptionError::ObstacleProjectionExceeded {
            group,
            strike,
            observed: relative_projection,
            tolerance: maximum_relative_projection,
        });
    }
    Ok((intrinsic, true, relative_projection))
}

fn european_call(
    spot: f64,
    strike: f64,
    maturity: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
) -> f64 {
    let scaled_volatility = volatility * maturity.sqrt();
    let d_one = ((spot / strike).ln()
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity)
        / scaled_volatility;
    let d_two = d_one - scaled_volatility;
    spot * (-dividend_yield * maturity).exp() * normal_cdf(d_one)
        - strike * (-rate * maturity).exp() * normal_cdf(d_two)
}

#[inline]
fn normal_cdf(value: f64) -> f64 {
    0.5 * libm::erfc(-value / SQRT_2)
}

#[derive(Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    const fn new() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let updated = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - updated) + value
        } else {
            (value - updated) + self.sum
        };
        self.sum = updated;
    }

    const fn total(self) -> f64 {
        self.sum + self.correction
    }
}

fn shard_ranges(groups: usize, shards: usize) -> Result<Vec<Range<usize>>, AmericanOptionError> {
    if groups == 0 {
        return Ok(Vec::new());
    }
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(shards)
        .map_err(|_| AmericanOptionError::Allocation {
            buffer: "American call shard ranges",
            elements: shards,
        })?;
    let width = groups / shards;
    let remainder = groups % shards;
    let mut start = 0;
    for shard in 0..shards {
        let end = start + width + usize::from(shard < remainder);
        ranges.push(start..end);
        start = end;
    }
    Ok(ranges)
}

fn allocated_f64(buffer: &'static str, elements: usize) -> Result<Vec<f64>, AmericanOptionError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| AmericanOptionError::Allocation { buffer, elements })?;
    values.resize(elements, 0.0);
    Ok(values)
}

#[allow(clippy::cast_precision_loss)]
fn exact_f64(value: usize) -> f64 {
    debug_assert!(value <= MAX_COLLOCATION_NODES);
    value as f64
}

#[cfg(test)]
mod tests {
    use finoptions_core::{
        ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
        LogDiscountCurve, TimeGrid,
    };
    use finoptions_testkit::interpolated_crr_call;

    use super::*;

    fn medium_method(maximum_residual: f64) -> AndersenLake {
        AndersenLake::try_new(
            AndersenLakeSpec::try_new(21, 32, 10, 41, maximum_residual, 1.0e-8).unwrap(),
        )
        .unwrap()
    }

    fn one_group<'a>(
        offsets: &'a [i64],
        spot: &'a [f64],
        maturity: &'a [f64],
        rate: &'a [f64],
        dividend_yield: &'a [f64],
        volatility: &'a [f64],
        strikes: &'a [f64],
    ) -> GroupedContinuousYieldCalls<'a> {
        GroupedContinuousYieldCalls::try_new(
            offsets,
            spot,
            maturity,
            rate,
            dividend_yield,
            volatility,
            strikes,
        )
        .unwrap()
    }

    fn independent_crr_call(
        spot: f64,
        strike: f64,
        maturity: f64,
        rate: f64,
        dividend_yield: f64,
        volatility: f64,
        steps: u32,
    ) -> f64 {
        let times = (0..=steps)
            .map(|node| maturity * f64::from(node) / f64::from(steps))
            .collect::<Vec<_>>();
        let grid = TimeGrid::try_new(times).unwrap();
        let expiry = grid.node_id(steps as usize).unwrap();
        let exercise_nodes = (0..steps as usize)
            .map(|node| grid.node_id(node).unwrap())
            .collect::<Vec<_>>();
        let schedule = EventSchedule::try_new(grid, [], exercise_nodes, expiry).unwrap();
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
            .map(|time| dividend_yield * time)
            .collect::<Vec<_>>();
        let curve = LogDiscountCurve::try_new(schedule.grid(), log_discounts).unwrap();
        let yield_curve =
            CumulativeDividendYieldCurve::try_new(schedule.grid(), cumulative_yields).unwrap();
        interpolated_crr_call(
            spot,
            strike,
            &schedule,
            &curve,
            &yield_curve,
            ConstantVolatility::try_new(volatility).unwrap(),
            DividendPolicy::Liquidator,
            steps as usize,
        )
        .unwrap()
    }

    #[test]
    fn exact_zero_yield_theorem_matches_european_black_scholes() {
        let inputs = one_group(
            &[0, 3],
            &[100.0],
            &[0.75],
            &[0.04],
            &[0.0],
            &[0.2],
            &[80.0, 100.0, 120.0],
        );
        let result = medium_method(1.0e-6)
            .price_call_groups(&CpuExecutor::try_new(1).unwrap(), inputs)
            .unwrap();
        for (&observed, strike) in result.prices().iter().zip([80.0, 100.0, 120.0]) {
            let expected = european_call(100.0, strike, 0.75, 0.04, 0.0, 0.2);
            assert_eq!(observed.to_bits(), expected.to_bits());
        }
        assert_eq!(
            result.group_diagnostics(),
            &[AndersenLakeGroupDiagnostics {
                maximum_relative_boundary_residual: 0.0,
                maximum_relative_obstacle_projection: 0.0,
                used_european_identity: true,
            }]
        );
    }

    #[test]
    fn grouped_strikes_share_one_converged_boundary_and_preserve_bounds() {
        let inputs = one_group(
            &[7, 11],
            &[100.0],
            &[1.0],
            &[0.03],
            &[0.05],
            &[0.25],
            &[70.0, 90.0, 100.0, 120.0],
        );
        let result = medium_method(1.0e-5)
            .price_call_groups(&CpuExecutor::try_new(1).unwrap(), inputs)
            .unwrap();
        assert!(result.group_diagnostics()[0].maximum_relative_boundary_residual() <= 1.0e-5);
        assert!(!result.group_diagnostics()[0].used_european_identity());
        for window in result.prices().windows(2) {
            assert!(window[0] >= window[1]);
        }
        for (&price, strike) in result.prices().iter().zip([70.0, 90.0, 100.0, 120.0]) {
            assert!(price >= (100.0_f64 - strike).max(0.0));
            assert!(price <= 100.0);
        }
    }

    #[test]
    fn reproduces_published_three_year_fp_b_put_benchmarks_via_symmetry() {
        let inputs = one_group(
            &[0, 3],
            &[100.0],
            &[3.0],
            &[0.04],
            &[0.04],
            &[0.20],
            &[80.0, 100.0, 120.0],
        );
        let observed = medium_method(2.0e-5)
            .price_call_groups(&CpuExecutor::try_new(1).unwrap(), inputs)
            .unwrap();
        for (&price, expected) in observed
            .prices()
            .iter()
            .zip([23.228_34, 12.605_21, 6.482_425])
        {
            assert!((price - expected).abs() <= 4.0e-5, "price={price}");
        }
    }

    #[test]
    fn negative_rate_zero_yield_uses_boundary_path_without_fallback() {
        let inputs = one_group(
            &[0, 1],
            &[100.0],
            &[0.5],
            &[-0.01],
            &[0.0],
            &[0.3],
            &[100.0],
        );
        let result = medium_method(1.0e-5)
            .price_call_groups(&CpuExecutor::try_new(1).unwrap(), inputs)
            .unwrap();
        assert!(!result.group_diagnostics()[0].used_european_identity());
        assert!(result.prices()[0].is_finite());
    }

    #[test]
    fn serial_and_parallel_group_execution_are_bitwise_identical() {
        let offsets = [0, 2, 4, 6, 8];
        let spots = [100.0, 90.0, 110.0, 75.0];
        let maturities = [0.25, 0.5, 0.75, 1.0];
        let rates = [0.03, -0.005, 0.02, 0.08];
        let yields = [0.04, 0.0, 0.06, 0.03];
        let volatilities = [0.2, 0.3, 0.4, 0.15];
        let strikes = [90.0, 110.0, 80.0, 100.0, 100.0, 120.0, 65.0, 85.0];
        let inputs = one_group(
            &offsets,
            &spots,
            &maturities,
            &rates,
            &yields,
            &volatilities,
            &strikes,
        );
        let method = medium_method(2.0e-5);
        let serial = method
            .price_call_groups(&CpuExecutor::try_new(1).unwrap(), inputs)
            .unwrap();
        let parallel = method
            .price_call_groups(&CpuExecutor::try_new(4).unwrap(), inputs)
            .unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn fixed_point_prices_agree_with_independent_dense_american_trees() {
        let cases = [
            (100.0, 100.0, 1.0, 0.03, 0.05, 0.25),
            (100.0, 100.0, 0.5, -0.01, 0.0, 0.30),
            (100.0, 90.0, 0.75, 0.08, 0.03, 0.20),
            (100.0, 120.0, 0.25, 0.01, 0.10, 0.50),
        ];
        let method = medium_method(1.0e-5);
        for (spot, strike, maturity, rate, yield_rate, volatility) in cases {
            let spots = [spot];
            let maturities = [maturity];
            let rates = [rate];
            let yields = [yield_rate];
            let volatilities = [volatility];
            let strikes = [strike];
            let inputs = one_group(
                &[0, 1],
                &spots,
                &maturities,
                &rates,
                &yields,
                &volatilities,
                &strikes,
            );
            let observed = method
                .price_call_groups(&CpuExecutor::try_new(1).unwrap(), inputs)
                .unwrap()
                .prices()[0];
            let independent =
                independent_crr_call(spot, strike, maturity, rate, yield_rate, volatility, 8_192);
            assert!(
                (observed - independent).abs() <= 3.0e-4,
                "observed={observed}, independent={independent}, case={:?}",
                (spot, strike, maturity, rate, yield_rate, volatility),
            );
        }
    }

    #[test]
    fn solved_stopping_region_returns_exact_intrinsic_and_reports_the_decision() {
        let inputs = one_group(&[0, 1], &[100.0], &[1.0], &[0.03], &[0.04], &[0.20], &[1.0]);
        let result = medium_method(2.0e-5)
            .price_call_groups(&CpuExecutor::try_new(1).unwrap(), inputs)
            .unwrap();
        assert_eq!(result.prices()[0].to_bits(), 99.0_f64.to_bits());
        assert_eq!(result.exercised_at_origin(), &[true]);
    }

    #[test]
    fn shape_preserving_cubic_stays_inside_nonmonotone_nonnegative_intervals() {
        let nodes = [-1.0, -0.6, 0.0, 0.3, 1.0];
        let values = [0.0, 0.04, 0.01, 0.09, 0.02];
        let mut slopes = [0.0; 5];
        prepare_shape_preserving_slopes(&nodes, &values, &mut slopes);
        for interval in 0..nodes.len() - 1 {
            let lower = values[interval].min(values[interval + 1]);
            let upper = values[interval].max(values[interval + 1]);
            for step in 0..=1_024 {
                let fraction = f64::from(step) / 1_024.0;
                let z = (nodes[interval + 1] - nodes[interval]).mul_add(fraction, nodes[interval]);
                let observed = shape_preserving_cubic_interpolate(z, &nodes, &values, &slopes);
                assert!(
                    observed >= lower && observed <= upper,
                    "observed={observed}"
                );
            }
        }
    }

    #[test]
    fn origin_obstacle_projection_is_bounded_reported_and_never_implicit() {
        let (price, exercised, projection) =
            apply_origin_obstacle(2, 3, 100.0, 80.0, 19.999_999_9, 2.0e-9).unwrap();
        assert_eq!(price.to_bits(), 20.0_f64.to_bits());
        assert!(exercised);
        assert!((projection - 1.0e-9).abs() <= 8.0 * f64::EPSILON);
        assert!(matches!(
            apply_origin_obstacle(2, 3, 100.0, 80.0, 19.999_999_9, 5.0e-10),
            Err(AmericanOptionError::ObstacleProjectionExceeded {
                group: 2,
                strike: 3,
                ..
            })
        ));
    }

    #[test]
    fn malformed_domains_fail_before_pricing() {
        assert!(matches!(
            GroupedContinuousYieldCalls::try_new(
                &[0, 1],
                &[100.0],
                &[1.0],
                &[0.02],
                &[-0.01],
                &[0.2],
                &[100.0],
            ),
            Err(AmericanOptionError::InvalidDividendYield { group: 0, .. })
        ));
        assert!(matches!(
            AndersenLakeSpec::try_new(20, 6, 10, 41, 1.0e-6, 1.0e-8),
            Err(AmericanOptionError::UnsupportedLegendreOrder {
                observed: 20,
                supported: crate::legendre::SUPPORTED_LEGENDRE_ORDERS,
            })
        ));
    }
}
