"""Tests for the leave-one-symbol-out ablation artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_ARTIFACT = Path("results/L2_SYMBOL_ABLATION.json")


@pytest.fixture(scope="module")
def ablation() -> dict[str, Any]:
    if not _ARTIFACT.exists():
        pytest.skip("symbol-ablation artifact not present")
    with _ARTIFACT.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def test_cells_cover_entire_symbol_universe(ablation: dict[str, Any]) -> None:
    """Each of the 10 default symbols is removed once."""
    removed = {c["removed_symbol"] for c in ablation["cells"]}
    assert removed == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "POLUSDT",
    }
    assert len(ablation["cells"]) == 10


def test_every_leave_one_out_cell_keeps_ic_positive(ablation: dict[str, Any]) -> None:
    """Removing ANY single symbol must not collapse the edge to ≤ 0."""
    cells = ablation["cells"]
    negatives = [c for c in cells if float(c["ic_point"]) <= 0.0]
    assert not negatives, f"edge collapses when removing: {negatives}"


def test_min_ic_exceeds_noise_floor(ablation: dict[str, Any]) -> None:
    """Even the worst-case leave-one-out retains material IC (> 0.04)."""
    assert float(ablation["min_ic"]) > 0.04


def test_verdict_is_one_of_canonical(ablation: dict[str, Any]) -> None:
    assert ablation["verdict"] in {"ROBUST", "MIXED", "CONCENTRATED"}


def test_baseline_ic_matches_killtest_within_tolerance(ablation: dict[str, Any]) -> None:
    """Baseline IC computed here should match the killtest canonical value to 3dp."""
    with Path("results/L2_KILLTEST_VERDICT.json").open("r", encoding="utf-8") as f:
        killtest: dict[str, Any] = json.load(f)
    baseline = float(ablation["baseline_ic"])
    canonical = float(killtest["ic_signal"])
    assert abs(baseline - canonical) < 1e-3


def test_cells_count_n_symbols_remaining_is_nine(ablation: dict[str, Any]) -> None:
    for cell in ablation["cells"]:
        assert int(cell["n_symbols_remaining"]) == 9


def test_thresholds_present_and_canonical(ablation: dict[str, Any]) -> None:
    assert float(ablation["robust_rel_drop_threshold"]) == 0.30
    assert float(ablation["concentrated_rel_drop_threshold"]) == 0.60
