# SPDX-License-Identifier: MIT
"""Tests for execution.risk.core module."""

from __future__ import annotations

import pytest

try:
    from execution.risk.core import RiskError
except ImportError:
    pytestmark = pytest.mark.skip("execution.risk.core not importable")
    RiskError = None


class TestRiskError:
    def test_is_runtime_error(self):
        err = RiskError("limit breached")
        assert isinstance(err, RuntimeError)
        assert str(err) == "limit breached"

    def test_can_be_caught_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise RiskError("test")
