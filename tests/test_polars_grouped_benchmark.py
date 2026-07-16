"""Smoke tests for the compact grouped Polars boundary benchmark."""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks.polars_grouped_boundary import main

_GROUPS = 3
_STRIKE_ROWS = 8
_ITERATIONS = 2


@pytest.mark.parametrize(
    ("dividends", "events", "native_request_bytes"),
    [
        ("continuous_yield", 0, 200),
        ("cash", 6, 304),
    ],
)
def test_grouped_boundary_report_verifies_both_compact_axes(
    dividends: str,
    events: int,
    native_request_bytes: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both dividend layouts emit one exact canonical compact request report."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "polars_grouped_boundary.py",
            "--groups",
            str(_GROUPS),
            "--strike-rows",
            str(_STRIKE_ROWS),
            "--dividends",
            dividends,
            "--iterations",
            str(_ITERATIONS),
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == "finoptions-polars-grouped-boundary-benchmark-v1"
    assert report["groups"] == _GROUPS
    assert report["strike_rows"] == _STRIKE_ROWS
    assert report["state_rows_avoided"] == _STRIKE_ROWS - _GROUPS
    assert report["events"] == events
    assert report["native_request_bytes"] == native_request_bytes
    assert report["scalar_and_offset_zero_copy_verified"] is True
