"""Public package contract tests."""

from __future__ import annotations

import finoptions
import pytest


def test_build_identity_is_available() -> None:
    """A loaded extension exposes the identities needed by future manifests."""
    info = finoptions.build_info()
    assert info.package_version == finoptions.__version__
    assert info.package_version == "0.1.0"
    assert info.rustc_version.startswith("rustc 1.97.0")


def test_event_schedule_is_validated_without_hidden_reordering() -> None:
    """The public facade preserves explicit event identity and phase order."""
    dividend = finoptions.CashDividend(node=1, amount=2.0)
    schedule = finoptions.EventSchedule(
        times=(0.0, 0.5, 1.0),
        dividends=(dividend,),
        exercise_nodes=(1,),
        expiry_node=2,
    )
    assert schedule.dividends == (dividend,)

    with pytest.raises(finoptions.InputError, match="strictly increasing"):
        finoptions.EventSchedule(
            times=(0.0, 0.25, 0.5, 1.0),
            dividends=(
                finoptions.CashDividend(node=2, amount=2.0),
                finoptions.CashDividend(node=1, amount=1.0),
            ),
            exercise_nodes=(),
            expiry_node=3,
        )


def test_input_types_and_canonical_zero_fail_early() -> None:
    """No implicit numeric narrowing, mutable schedules, or negative-zero identity is accepted."""
    with pytest.raises(finoptions.InputError, match="plain float"):
        finoptions.CashDividend(node=1, amount=2)
    with pytest.raises(finoptions.InputError, match="finite and nonnegative"):
        finoptions.CashDividend(node=1, amount=-0.0)
    with pytest.raises(finoptions.InputError, match="immutable tuple"):
        finoptions.EventSchedule(
            times=[0.0, 1.0],  # ty: ignore[invalid-argument-type]
            dividends=(),
            exercise_nodes=(),
            expiry_node=1,
        )


def test_curve_supports_negative_rates_but_rejects_bad_origin() -> None:
    """Curve validation allows discount factors above one without weakening canonical identity."""
    expected_final_log_discount = 0.005
    curve = finoptions.LogDiscountCurve(
        times=(0.0, 0.5, 1.0),
        log_discounts=(0.0, -0.01, expected_final_log_discount),
    )
    assert curve.log_discounts[-1] == expected_final_log_discount
    with pytest.raises(finoptions.InputError, match=r"exactly 0\.0"):
        finoptions.LogDiscountCurve(times=(0.0, 1.0), log_discounts=(-0.0, -0.01))
