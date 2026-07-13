"""Public package contract tests."""

from __future__ import annotations

import finoptions


def test_build_identity_is_available() -> None:
    """A loaded extension exposes the identities needed by future manifests."""
    info = finoptions.build_info()
    assert info.package_version == finoptions.__version__
    assert info.package_version == "0.1.0"
    assert info.rustc_version.startswith("rustc 1.97.0")
