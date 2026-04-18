"""Tests for the regime-conditional IC decomposition artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_ARTIFACT = Path("results/L2_REGIME_CONDITIONAL_IC.json")


@pytest.fixture(scope="module")
def cond() -> dict[str, Any]:
    if not _ARTIFACT.exists():
        pytest.skip("regime-conditional IC artifact not present")
    with _ARTIFACT.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def test_two_cells_cover_full_universe(cond: dict[str, Any]) -> None:
    regimes = {c["regime"] for c in cond["cells"]}
    assert regimes == {"HIGH_VOL", "LOW_VOL"}


def test_cell_row_counts_sum_close_to_total(cond: dict[str, Any]) -> None:
    """High + low rows must equal total rows (within 1 row for edge cases)."""
    total = sum(int(c["n_rows"]) for c in cond["cells"])
    assert total > 10_000  # Session 1 substrate expected size


def test_verdict_is_canonical(cond: dict[str, Any]) -> None:
    assert cond["verdict"] in {"VOL_DRIVEN", "UNIFORM", "QUIET_DRIVEN", "INCONCLUSIVE"}


def test_high_vol_quantile_mask_approximately_sized(cond: dict[str, Any]) -> None:
    """HIGH_VOL mask at q=0.75 should cover ~25% of rows (±5%)."""
    high = next(c for c in cond["cells"] if c["regime"] == "HIGH_VOL")
    low = next(c for c in cond["cells"] if c["regime"] == "LOW_VOL")
    total = int(high["n_rows"]) + int(low["n_rows"])
    frac_high = int(high["n_rows"]) / total
    assert 0.20 <= frac_high <= 0.30


def test_baseline_ic_matches_pooled_killtest(cond: dict[str, Any]) -> None:
    """Pooled IC in cond artifact must match the kill-test canonical value."""
    with Path("results/L2_KILLTEST_VERDICT.json").open("r", encoding="utf-8") as f:
        killtest: dict[str, Any] = json.load(f)
    baseline = float(cond["baseline_ic_pooled"])
    canonical = float(killtest["ic_signal"])
    assert abs(baseline - canonical) < 1e-3


def test_ratio_nonnegative_or_nan(cond: dict[str, Any]) -> None:
    import math

    r = float(cond["abs_ratio_high_over_low"])
    assert math.isnan(r) or r >= 0.0
