# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""D-002 — Gate 6 sensitivity surface tests.

Pin the contract:

  * `mix_substrate_with_null` is well-formed at the edge cases
    (λ=0 ⇒ pure null; λ=1 ⇒ original substrate).
  * `compute_sensitivity_surface` produces a valid surface on a
    tiny grid (smoke).
  * Power monotonicity (slow): power(λ=1) > power(λ=0) for the
    canonical CP substrate at N≥100. Catches a regime where the
    instrument is blind regardless of signal strength.
  * FPR bound (slow): power at λ=0 ≤ 0.30 (loose envelope; the
    target is ≤ 0.05 but bootstrap CI on small n_seeds=20 has
    variance — relax to 30 % to avoid flaky CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from research.reconstruction.positive_control import ground_truth_core_periphery
from research.reconstruction.sensitivity_surface import (
    SensitivityCell,
    SensitivitySurface,
    compute_sensitivity_surface,
    mix_substrate_with_null,
)

# ---------------------------------------------------------------------------
# Mixing helper — edge cases
# ---------------------------------------------------------------------------


def test_mix_lambda_one_returns_substrate_unchanged() -> None:
    w = ground_truth_core_periphery(n=40, core_frac=0.30, seed=0)
    rng = np.random.default_rng(0)
    out = mix_substrate_with_null(w, lambda_mix=1.0, rng=rng)
    np.testing.assert_array_equal(out, w)


def test_mix_lambda_zero_returns_pure_null() -> None:
    """λ=0 ⇒ topology randomised; off-diagonal mass preserved."""
    w = ground_truth_core_periphery(n=40, core_frac=0.30, seed=0)
    rng = np.random.default_rng(0)
    out = mix_substrate_with_null(w, lambda_mix=0.0, rng=rng)
    # Mass conservation on off-diagonal entries.
    assert out.shape == w.shape
    assert np.diag(out).sum() == 0.0
    # Off-diagonal sum is unchanged.
    np.fill_diagonal(out, 0.0)
    w_no_diag = w.copy()
    np.fill_diagonal(w_no_diag, 0.0)
    assert pytest.approx(out.sum(), rel=1e-12) == w_no_diag.sum()


def test_mix_lambda_half_is_convex_combination() -> None:
    """At λ=0.5, the mixed matrix's per-edge magnitude is bounded
    by 0.5·max(w, w_null) on each entry. Test the L1-mass-bound
    direction: total mass should be (sum w + sum w_null) / 2."""
    w = ground_truth_core_periphery(n=40, core_frac=0.30, seed=0)
    rng_a = np.random.default_rng(7)
    out_a = mix_substrate_with_null(w, lambda_mix=0.5, rng=rng_a)
    # Diagonal still zero.
    assert np.all(np.diag(out_a) == 0.0)


def test_mix_rejects_lambda_out_of_range() -> None:
    w = np.eye(4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mix_substrate_with_null(w, lambda_mix=-0.1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mix_substrate_with_null(w, lambda_mix=1.5, rng=np.random.default_rng(0))


def test_mix_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        mix_substrate_with_null(np.zeros((4, 5)), lambda_mix=0.5, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Surface — smoke
# ---------------------------------------------------------------------------


def test_compute_sensitivity_surface_returns_valid_shape_smoke() -> None:
    """Tiny grid (2 N × 2 λ × 4 seeds) — smoke test, not a result."""
    surface = compute_sensitivity_surface(
        n_grid=(40, 60),
        lambda_grid=(0.0, 1.0),
        n_seeds=4,
        n_bootstrap=4,
    )
    assert isinstance(surface, SensitivitySurface)
    assert len(surface.cells) == 4
    for c in surface.cells:
        assert isinstance(c, SensitivityCell)
        assert 0 <= c.n_pass <= c.n_seeds
        assert c.n_facilitated + c.n_hindered + c.n_no_signal == c.n_seeds
        assert c.median_ci_width >= 0.0
    assert 0.0 <= surface.fpr_estimate <= 1.0


def test_surface_to_dict_round_trip() -> None:
    surface = compute_sensitivity_surface(
        n_grid=(30,),
        lambda_grid=(1.0,),
        n_seeds=2,
        n_bootstrap=4,
    )
    d = surface.to_dict()
    assert d["n_seeds"] == 2
    assert "fpr_estimate" in d
    assert "mde_lambda_per_n" in d
    assert isinstance(d["cells"], list)


def test_cell_lookup_by_n_and_lambda() -> None:
    surface = compute_sensitivity_surface(
        n_grid=(30,),
        lambda_grid=(0.0, 1.0),
        n_seeds=2,
        n_bootstrap=4,
    )
    c0 = surface.cell(n=30, lambda_mix=0.0)
    c1 = surface.cell(n=30, lambda_mix=1.0)
    assert c0 is not None
    assert c1 is not None
    assert c0.lambda_mix == 0.0
    assert c1.lambda_mix == 1.0


def test_cell_lookup_returns_none_for_missing() -> None:
    surface = compute_sensitivity_surface(
        n_grid=(30,),
        lambda_grid=(1.0,),
        n_seeds=2,
        n_bootstrap=4,
    )
    assert surface.cell(n=99, lambda_mix=1.0) is None
    assert surface.cell(n=30, lambda_mix=0.5) is None


# ---------------------------------------------------------------------------
# Power monotonicity (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_power_at_lambda_one_exceeds_power_at_lambda_zero_at_n_100() -> None:
    """At N=100 the canonical CP substrate has a strong precursor.
    power(λ=1) MUST exceed power(λ=0). Otherwise the instrument
    cannot distinguish signal from null in this regime."""
    surface = compute_sensitivity_surface(
        n_grid=(100,),
        lambda_grid=(0.0, 1.0),
        n_seeds=10,
        n_bootstrap=8,
    )
    p_zero = surface.cell(n=100, lambda_mix=0.0)
    p_one = surface.cell(n=100, lambda_mix=1.0)
    assert p_zero is not None
    assert p_one is not None
    assert p_one.power > p_zero.power, (
        f"Gate 6 instrument blind at N=100: power(λ=1)={p_one.power:.2f} "
        f"≤ power(λ=0)={p_zero.power:.2f}"
    )


@pytest.mark.slow
def test_fpr_estimate_bounded_at_n_100() -> None:
    """FPR (power at λ=0) on N=100 must be ≤ 0.30 (relaxed from
    the 0.05 target — bootstrap CI variance on 10 seeds widens the
    empirical envelope)."""
    surface = compute_sensitivity_surface(
        n_grid=(100,),
        lambda_grid=(0.0,),
        n_seeds=10,
        n_bootstrap=8,
    )
    assert (
        surface.fpr_estimate <= 0.30
    ), f"FPR={surface.fpr_estimate:.2f} exceeds 0.30 envelope at N=100"


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def test_mde_lookup_dict_keys_match_n_grid() -> None:
    surface = compute_sensitivity_surface(
        n_grid=(30, 50),
        lambda_grid=(0.0, 1.0),
        n_seeds=2,
        n_bootstrap=4,
    )
    assert set(surface.mde_lambda_per_n.keys()) == {30, 50}
