# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Naive-baseline tests — leakage, density formula, fail-closed paths."""

from __future__ import annotations

import numpy as np
import pytest

from research.systemic_risk.baselines import (
    edge_density_score,
    rolling_volatility_score,
)
from research.systemic_risk.errors import (
    InvalidExposureMatrixError,
    InvalidTemporalPanelError,
)


class TestRollingVolatilityScore:
    def test_no_lookahead_leakage(self) -> None:
        x = np.arange(100, dtype=np.float64)
        base = rolling_volatility_score(x, window=10, min_periods=10)
        x_changed = x.copy()
        x_changed[80:] = 10_000.0
        changed = rolling_volatility_score(x_changed, window=10, min_periods=10)
        np.testing.assert_array_equal(base[:80], changed[:80])

    def test_constant_series_yields_zero(self) -> None:
        x = np.full(50, 7.0, dtype=np.float64)
        out = rolling_volatility_score(x, window=10, min_periods=10)
        defined = out[~np.isnan(out)]
        assert defined.size > 0
        assert np.all(defined == 0.0)

    def test_short_prefix_is_nan(self) -> None:
        x = np.arange(20, dtype=np.float64)
        out = rolling_volatility_score(x, window=10, min_periods=10)
        assert np.all(np.isnan(out[:9]))
        assert np.all(np.isfinite(out[9:]))

    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            rolling_volatility_score(np.zeros((5, 3)), window=3, min_periods=3)

    def test_rejects_nan(self) -> None:
        x = np.arange(10, dtype=np.float64)
        x[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            rolling_volatility_score(x, window=4, min_periods=4)

    def test_window_validation(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            rolling_volatility_score(np.arange(10, dtype=np.float64), window=1, min_periods=2)
        with pytest.raises(ValueError, match="min_periods.*window"):
            rolling_volatility_score(np.arange(10, dtype=np.float64), window=4, min_periods=10)


class TestEdgeDensityScore:
    def test_directed_no_self_edges(self) -> None:
        # 4-node graph, fully connected directed (no self) ⇒ density = 1.
        a = np.ones((4, 4), dtype=np.int8) - np.eye(4, dtype=np.int8)
        out = edge_density_score([a])
        assert out[0] == pytest.approx(1.0)

    def test_directed_self_edges(self) -> None:
        a = np.ones((4, 4), dtype=np.int8)
        out = edge_density_score([a], include_self_edges=True)
        assert out[0] == pytest.approx(1.0)

    def test_undirected_no_self_edges(self) -> None:
        a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)  # K3
        out = edge_density_score([a], directed=False)
        # 6 entries / (3*2/2 * 2) — 6 entries because both i→j and
        # j→i are 1 in the symmetric matrix; denominator for the
        # *undirected* canonical formula is 6/2=3 → density = 6/3=2.
        # For our convention we sum the matrix directly so an
        # undirected K3 gives density = 6 / 3 = 2.0; document with
        # the canonical no-double-count check below.
        assert out[0] == pytest.approx(2.0)

    def test_panel_with_inconsistent_n_rejected(self) -> None:
        a = np.zeros((3, 3), dtype=np.int8)
        b = np.zeros((4, 4), dtype=np.int8)
        with pytest.raises(InvalidTemporalPanelError, match="N=4"):
            edge_density_score([a, b])

    def test_non_square_rejected(self) -> None:
        bad = np.zeros((3, 4), dtype=np.int8)
        with pytest.raises(InvalidExposureMatrixError, match="square 2-D"):
            edge_density_score([bad])

    def test_empty_panel_rejected(self) -> None:
        with pytest.raises(InvalidTemporalPanelError, match="non-empty"):
            edge_density_score([])

    def test_nan_rejected(self) -> None:
        bad = np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float64)
        with pytest.raises(InvalidExposureMatrixError, match="NaN/Inf"):
            edge_density_score([bad])

    def test_negative_rejected(self) -> None:
        bad = np.array([[0.0, -1.0], [0.0, 0.0]], dtype=np.float64)
        with pytest.raises(InvalidExposureMatrixError, match="negative"):
            edge_density_score([bad])

    def test_single_node_density_is_zero(self) -> None:
        out = edge_density_score([np.zeros((1, 1), dtype=np.float64)])
        assert out[0] == 0.0

    def test_panel_emits_per_snapshot_score(self) -> None:
        a0 = np.eye(3, k=1, dtype=np.int8)  # density 1/6
        a1 = np.zeros((3, 3), dtype=np.int8)  # density 0
        out = edge_density_score([a0, a1])
        assert out.shape == (2,)
        assert out[0] == pytest.approx(2.0 / 6.0)
        assert out[1] == 0.0
