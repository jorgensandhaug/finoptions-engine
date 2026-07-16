"""Public grouped Andersen-Lake API and fail-fast domain tests."""

from __future__ import annotations

import numpy as np
import pytest
from finoptions import (
    AmericanOptionError,
    AndersenLake,
    CpuExecutor,
    InputError,
    price_andersen_lake_call_grouped,
)

MAXIMUM_RESIDUAL = 2.0e-5
MAXIMUM_OBSTACLE_PROJECTION = 1.0e-8


def method(*, residual: float = MAXIMUM_RESIDUAL) -> AndersenLake:
    """Return the fixed medium-precision configuration used by API tests."""
    return AndersenLake(
        boundary_quadrature_order=21,
        iterations=32,
        collocation_nodes=10,
        price_quadrature_order=41,
        maximum_relative_boundary_residual=residual,
        maximum_relative_obstacle_projection=MAXIMUM_OBSTACLE_PROJECTION,
    )


def test_published_put_benchmark_is_reproduced_through_call_put_symmetry() -> None:
    """One group must share one boundary across all three published spots/strikes."""
    result = price_andersen_lake_call_grouped(
        group_offsets=np.array([0, 3], dtype=np.int64),
        group_spots=np.array([100.0], dtype=np.float64),
        group_maturities=np.array([3.0], dtype=np.float64),
        group_constant_continuously_compounded_rates=np.array([0.04], dtype=np.float64),
        group_annualized_dividend_yields=np.array([0.04], dtype=np.float64),
        group_constant_volatilities=np.array([0.20], dtype=np.float64),
        strikes=np.array([80.0, 100.0, 120.0], dtype=np.float64),
        executor=CpuExecutor(threads=2),
        method=method(),
    )
    np.testing.assert_allclose(
        result.prices,
        np.array([23.228_34, 12.605_21, 6.482_425], dtype=np.float64),
        rtol=0.0,
        atol=4.0e-5,
    )
    assert result.maximum_relative_boundary_residuals.shape == (1,)
    assert result.maximum_relative_boundary_residuals[0] <= MAXIMUM_RESIDUAL
    assert result.maximum_relative_obstacle_projections[0] <= MAXIMUM_OBSTACLE_PROJECTION
    assert not result.used_european_identity[0]
    assert result.exercised_at_origin.tolist() == [False, False, False]
    assert result.rows == result.prices.size
    assert result.groups == result.used_european_identity.size
    for values in (
        result.prices,
        result.exercised_at_origin,
        result.maximum_relative_boundary_residuals,
        result.maximum_relative_obstacle_projections,
        result.used_european_identity,
    ):
        assert not values.flags.writeable


def test_solved_exercise_region_returns_exact_intrinsic_without_clamping() -> None:
    """A contract inside the solved stopping region must take the exact stopping payoff."""
    result = price_andersen_lake_call_grouped(
        group_offsets=np.array([0, 1], dtype=np.int64),
        group_spots=np.array([100.0], dtype=np.float64),
        group_maturities=np.array([1.0], dtype=np.float64),
        group_constant_continuously_compounded_rates=np.array([0.03], dtype=np.float64),
        group_annualized_dividend_yields=np.array([0.04], dtype=np.float64),
        group_constant_volatilities=np.array([0.20], dtype=np.float64),
        strikes=np.array([1.0], dtype=np.float64),
        executor=CpuExecutor(threads=1),
        method=method(),
    )
    assert result.prices.tolist() == [99.0]
    assert result.exercised_at_origin.tolist() == [True]


def test_obstacle_projection_limit_is_explicit_and_fail_fast() -> None:
    """A negative projection limit must fail during method construction."""
    with pytest.raises(AmericanOptionError, match="must be finite and nonnegative"):
        AndersenLake(
            boundary_quadrature_order=21,
            iterations=32,
            collocation_nodes=10,
            price_quadrature_order=41,
            maximum_relative_boundary_residual=MAXIMUM_RESIDUAL,
            maximum_relative_obstacle_projection=-1.0,
        )


def test_negative_method_count_is_a_typed_public_input_error() -> None:
    """PyO3 unsigned conversion must not escape the public typed error boundary."""
    with pytest.raises(InputError, match="nonnegative platform-representable integers"):
        AndersenLake(
            boundary_quadrature_order=-1,
            iterations=32,
            collocation_nodes=10,
            price_quadrature_order=41,
            maximum_relative_boundary_residual=MAXIMUM_RESIDUAL,
            maximum_relative_obstacle_projection=MAXIMUM_OBSTACLE_PROJECTION,
        )


def test_zero_yield_nonnegative_rate_uses_exact_european_identity() -> None:
    """The theorem route must be explicit in group diagnostics."""
    result = price_andersen_lake_call_grouped(
        group_offsets=np.array([0, 2], dtype=np.int64),
        group_spots=np.array([100.0], dtype=np.float64),
        group_maturities=np.array([0.5], dtype=np.float64),
        group_constant_continuously_compounded_rates=np.array([0.03], dtype=np.float64),
        group_annualized_dividend_yields=np.array([0.0], dtype=np.float64),
        group_constant_volatilities=np.array([0.25], dtype=np.float64),
        strikes=np.array([90.0, 110.0], dtype=np.float64),
        executor=CpuExecutor(threads=1),
        method=method(),
    )
    assert result.used_european_identity.tolist() == [True]
    assert result.maximum_relative_boundary_residuals.tolist() == [0.0]


def test_array_and_financial_domains_fail_before_pricing() -> None:
    """No dtype conversion or negative-yield routing is inferred."""
    with pytest.raises(InputError, match="group_offsets must have dtype int64"):
        price_andersen_lake_call_grouped(
            group_offsets=np.array([0, 1], dtype=np.int32),  # type: ignore[arg-type]
            group_spots=np.array([100.0], dtype=np.float64),
            group_maturities=np.array([0.5], dtype=np.float64),
            group_constant_continuously_compounded_rates=np.array([0.03], dtype=np.float64),
            group_annualized_dividend_yields=np.array([0.02], dtype=np.float64),
            group_constant_volatilities=np.array([0.25], dtype=np.float64),
            strikes=np.array([100.0], dtype=np.float64),
            executor=CpuExecutor(threads=1),
            method=method(),
        )
    with pytest.raises(InputError, match="dividend yield must be finite and nonnegative"):
        price_andersen_lake_call_grouped(
            group_offsets=np.array([0, 1], dtype=np.int64),
            group_spots=np.array([100.0], dtype=np.float64),
            group_maturities=np.array([0.5], dtype=np.float64),
            group_constant_continuously_compounded_rates=np.array([0.03], dtype=np.float64),
            group_annualized_dividend_yields=np.array([-0.01], dtype=np.float64),
            group_constant_volatilities=np.array([0.25], dtype=np.float64),
            strikes=np.array([100.0], dtype=np.float64),
            executor=CpuExecutor(threads=1),
            method=method(),
        )


def test_declared_residual_gate_never_refines_or_retries() -> None:
    """An intentionally impossible tolerance must return the numerical failure."""
    with pytest.raises(AmericanOptionError, match="boundary residual"):
        price_andersen_lake_call_grouped(
            group_offsets=np.array([0, 1], dtype=np.int64),
            group_spots=np.array([100.0], dtype=np.float64),
            group_maturities=np.array([1.0], dtype=np.float64),
            group_constant_continuously_compounded_rates=np.array([0.03], dtype=np.float64),
            group_annualized_dividend_yields=np.array([0.05], dtype=np.float64),
            group_constant_volatilities=np.array([0.25], dtype=np.float64),
            strikes=np.array([100.0], dtype=np.float64),
            executor=CpuExecutor(threads=1),
            method=method(residual=1.0e-15),
        )
